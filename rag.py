"""
Multilingual RAG System for Danfoss ECL Comfort Controller
"""

import os
from typing import List, Dict, Any
import requests
from langdetect import detect, LangDetectException
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
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
        """Create or load vector store"""
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
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
            model=os.getenv("LLM_MODEL_NAME", "llama3.1:8b-instruct-q3_K_L"),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            num_predict=int(os.getenv("NUM_PREDICT", "2000")),
            verbose=False
        )

class RAGSystem:
    PROMPT_TEMPLATE = """
    You are an expert assistant for Danfoss Product Catalogue 2025.
    Answer the question in the same language as the question using the following context.
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
        """Set up RAG components"""
        pdf_path = DocumentProcessor.download_pdf()
        chunks = DocumentProcessor.load_and_split(pdf_path)
        vector_store = VectorStoreManager.create_store(chunks)
        return self._create_chain(vector_store)

    def _create_chain(self, vector_store: Chroma) -> Any:
        """Create RAG chain"""
        llm = LLMFactory.create_llm()
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": int(os.getenv("RETRIEVER_K", "5"))}
        )
        prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        document_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, document_chain)

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