import os
from typing import List, Any
import requests
from langdetect import detect, LangDetectException
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document, BaseRetriever
from langchain.llms.base import BaseLLM
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentProcessor:
    @staticmethod
    def download_pdf(url: str = os.getenv("PDF_URL"), output_path: str = "document.pdf") -> str:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path

    @staticmethod
    def load_and_split(pdf_path: str) -> List[Document]:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

class VectorStoreManager:
    @staticmethod
    def create_store(chunks: List[Document]) -> Chroma:
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase --multilingual-MiniLM-L12-v2")
        )
        chroma_db_dir = os.getenv("CHROMA_DB_DIR", "danfoss_db")
        if os.path.exists(chroma_db_dir):
            return Chroma(
                persist_directory=chroma_db_dir,
                embedding_function=embeddings
            )
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_db_dir
        )
        vector_store.persist()
        return vector_store

class LLMFactory:
    @staticmethod
    def create_llm() -> Ollama:
        return Ollama(
            model=os.getenv("LLM_MODEL_NAME", "phi3:mini"),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            num_predict=int(os.getenv("NUM_PREDICT", "2000")),
            verbose=False
        )

class RerankingRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever = Field(description="Base retriever for initial document retrieval")
    llm: BaseLLM = Field(description="Language model for reranking")
    k: int = Field(default=5, description="Number of initial documents to retrieve")
    m: int = Field(default=3, description="Number of documents to return after reranking")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        initial_docs = self.base_retriever.get_relevant_documents(query)[:self.k]
        scores = []
        for doc in initial_docs:
            prompt = f"Question: {query}\nDocument: {doc.page_content}\nOn a scale of 0 to 10, how relevant is this document to answering the question? Only provide the number."
            response = self.llm(prompt)
            try:
                score = float(response.strip())
            except ValueError:
                score = 0
            scores.append((doc, score))
        sorted_docs = sorted(scores, key=lambda x: x[1], reverse=True)[:self.m]
        return [doc for doc, _ in sorted_docs]

    def with_config(self, **kwargs) -> "RerankingRetriever":
        return self.__class__(
            base_retriever=self.base_retriever,
            llm=self.llm,
            k=self.k,
            m=self.m
        )

class RAGSystem:
    PROMPT_TEMPLATE = """
    You are an expert assistant for Danfoss Product Catalogue 2025.
    Answer the question in the same language as the question using the following context.
    Include references to the page numbers where the information can be found.
    If you don't know the answer, say you don't know based on the available documentation.
    
    Answer language: {language}
    Context:
    {context}
    
    Question: {input}
    
    Answer:
    """
    
    LANGUAGE_MAPPING = {
        "id": "Indonesian",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German"
    }

    def __init__(self):
        self.rag_chain = self._setup_system()

    def _setup_system(self) -> Any:
        pdf_path = DocumentProcessor.download_pdf()
        chunks = DocumentProcessor.load_and_split(pdf_path)
        vector_store = VectorStoreManager.create_store(chunks)
        return self._create_chain(vector_store)

    def _create_chain(self, vector_store: Chroma) -> Any:
        llm = LLMFactory.create_llm()
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": int(os.getenv("RETRIEVER_K", "5"))}
        )
        reranking_retriever = RerankingRetriever(base_retriever=base_retriever, llm=llm, k=5, m=3)
        document_prompt = PromptTemplate.from_template("Page {page}: {page_content}")
        prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        document_chain = create_stuff_documents_chain(
            llm,
            prompt,
            document_prompt=document_prompt
        )
        return create_retrieval_chain(reranking_retriever, document_chain)

    def _detect_language(self, text: str) -> str:
        try:
            lang_code = detect(text)
            return self.LANGUAGE_MAPPING.get(
                lang_code,
                os.getenv("DEFAULT_LANGUAGE", "English")
            )
        except LangDetectException:
            return os.getenv("DEFAULT_LANGUAGE", "English")

    def query(self, question: str) -> str:
        try:
            detected_lang = self._detect_language(question)
            response = self.rag_chain.invoke({
                "input": question,
                "language": detected_lang
            })
            return response["answer"]
        except Exception as e:
            return f"Error processing query: {str(e)}"

class ChatInterface:
    @staticmethod
    def run():
        try:
            rag_system = RAGSystem()
            while True:
                user_input = input("\nQuestion: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input:
                    continue
                answer = rag_system.query(user_input)
                print(f"Answer: {answer}")
        except Exception as e:
            print(f"System error: {str(e)}")
            
if __name__ == "__main__":
    ChatInterface.run()