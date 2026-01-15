"""
RAG Chain Module

Creates and manages RAG (Retrieval-Augmented Generation) chains.
Supports multiple chain types: retrieval chain, LCEL chain, and conversational chain.
"""
from typing import List, Optional, Dict, Any, Callable
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.rag.llm_manager import LLMManager
from app.rag.vector_store import VectorStoreManager
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class RAGChainManager:
    """
    Manages RAG chains for question-answering.
    
    Supports:
    - Standard retrieval chain (create_retrieval_chain)
    - LCEL-based chain (LangChain Expression Language)
    - Conversational chain with history-aware retriever
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        llm_manager: Optional[LLMManager] = None,
        llm: Optional[BaseChatModel] = None,
        retriever: Optional[BaseRetriever] = None,
        k: int = 3,
        chain_type: str = "retrieval"
    ):
        """
        Initialize RAG chain manager.

        Args:
            vector_store: VectorStoreManager instance
            llm_manager: LLMManager instance (optional, creates default if None)
            llm: Direct LLM instance (optional, uses llm_manager if None)
            retriever: Custom retriever (optional, creates from vector_store if None)
            k: Number of documents to retrieve
            chain_type: Chain type ("retrieval", "lcel", or "conversational")
        """
        self.vector_store = vector_store
        
        # Initialize LLM
        if llm is not None:
            self.llm = llm
        elif llm_manager is not None:
            self.llm = llm_manager.get_llm()
        else:
            # Create default LLM manager
            self.llm_manager = LLMManager()
            self.llm = self.llm_manager.get_llm()
        
        # Create retriever
        if retriever is not None:
            self.retriever = retriever
        else:
            vector_store_instance = vector_store.get_vector_store()
            self.retriever = vector_store_instance.as_retriever(search_kwargs={"k": k})
        
        self.k = k
        self.chain_type = chain_type
        
        # Initialize chain based on type
        if chain_type == "retrieval":
            self.chain = self._create_retrieval_chain()
        elif chain_type == "lcel":
            self.chain = self._create_lcel_chain()
        elif chain_type == "conversational":
            self.chain = self._create_conversational_chain()
        else:
            raise ValueError(f"Unsupported chain_type: {chain_type}. Use 'retrieval', 'lcel', or 'conversational'")
        
        logger.info(
            f"RAGChainManager initialized | "
            f"Chain type: {chain_type} | "
            f"Retrieve k: {k}"
        )

    def _create_retrieval_chain(self):
        """Create standard retrieval chain using create_retrieval_chain"""
        # Create prompt template
        system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
        
        return retrieval_chain

    def _create_lcel_chain(self):
        """Create LCEL-based RAG chain"""
        # Create custom prompt
        custom_prompt = ChatPromptTemplate.from_template("""Use the following context to answer the question. 
If you don't know the answer based on the context, say you don't know.
Provide specific details from the context to support your answer.

Context:
{context}

Question: {question}

Answer:""")
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Build LCEL chain
        lcel_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | custom_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return lcel_chain

    def _create_conversational_chain(self):
        """Create conversational RAG chain with history-aware retriever"""
        # Create contextualize question prompt
        contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )
        
        # Create QA prompt with history
        qa_system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create document chain
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # Create conversational RAG chain
        conversational_rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )
        
        return conversational_rag_chain

    def invoke(
        self,
        query: str,
        chat_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        """
        Invoke the RAG chain with a query.

        Args:
            query: User query/question
            chat_history: Optional chat history for conversational chains

        Returns:
            Dictionary with 'answer' and 'context' (for retrieval chain) or just answer (for LCEL)
        """
        try:
            if self.chain_type == "conversational":
                if chat_history is None:
                    chat_history = []
                result = self.chain.invoke({
                    "chat_history": chat_history,
                    "input": query
                })
            elif self.chain_type == "lcel":
                # LCEL chain takes string directly
                answer = self.chain.invoke(query)
                # Get source documents separately
                docs = self.retriever.get_relevant_documents(query)
                result = {
                    "answer": answer,
                    "context": docs
                }
            else:
                # Standard retrieval chain
                result = self.chain.invoke({"input": query})
            
            logger.info(
                f"RAG chain invoked | "
                f"Chain type: {self.chain_type} | "
                f"Query: {query[:50]}..."
            )
            
            return result
        except Exception as e:
            logger.error(f"Error invoking RAG chain: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to invoke RAG chain: {str(e)}")

    def get_chain(self):
        """Get the underlying chain"""
        return self.chain

    def get_retriever(self) -> BaseRetriever:
        """Get the retriever"""
        return self.retriever


def create_rag_chain(
    vector_store: VectorStoreManager,
    llm_provider: str = "openai",
    llm_model_name: Optional[str] = None,
    temperature: float = 0.0,
    k: int = 3,
    chain_type: str = "retrieval",
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None
) -> RAGChainManager:
    """
    Create a RAG chain manager.

    Args:
        vector_store: VectorStoreManager instance
        llm_provider: LLM provider ("openai" or "groq")
        llm_model_name: Model name
        temperature: Sampling temperature
        k: Number of documents to retrieve
        chain_type: Chain type ("retrieval", "lcel", or "conversational")
        openai_api_key: OpenAI API key
        groq_api_key: GROQ API key

    Returns:
        RAGChainManager instance
    """
    llm_manager = LLMManager(
        provider=llm_provider,
        model_name=llm_model_name,
        temperature=temperature,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key
    )
    
    return RAGChainManager(
        vector_store=vector_store,
        llm_manager=llm_manager,
        k=k,
        chain_type=chain_type
    )
