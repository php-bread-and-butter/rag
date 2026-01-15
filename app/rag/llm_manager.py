"""
LLM Manager Module

Manages Language Model instances for RAG applications.
Supports OpenAI, GROQ, and other providers via LangChain.
"""
import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.chat_models.base import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.logging_config import get_logger
from app.core.config import settings

logger = get_logger(__name__)

# Supported LLM providers
SUPPORTED_PROVIDERS = ["openai", "groq"]

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-3.5-turbo",
    "groq": "gemma2-9b-it"
}

# Model information
OPENAI_MODELS = {
    "gpt-3.5-turbo": "GPT-3.5 Turbo - Fast and cost-effective",
    "gpt-4": "GPT-4 - More capable than GPT-3.5",
    "gpt-4-turbo": "GPT-4 Turbo - Latest GPT-4 model",
    "gpt-4o": "GPT-4o - Optimized GPT-4 model",
}

GROQ_MODELS = {
    "gemma2-9b-it": "Gemma 2 9B Instruct - Fast inference",
    "llama-3.1-8b-instant": "Llama 3.1 8B Instant - Fast inference",
    "llama-3.1-70b-versatile": "Llama 3.1 70B Versatile - More capable",
    "mixtral-8x7b-32768": "Mixtral 8x7B - High quality",
}


class LLMManager:
    """
    Manages Language Model instances for RAG applications.
    
    Supports:
    - OpenAI Chat Models (ChatOpenAI)
    - GROQ Chat Models (ChatGroq)
    - Generic models via init_chat_model
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the LLM manager.

        Args:
            provider: LLM provider ("openai" or "groq")
            model_name: Model name (defaults to provider's default)
            temperature: Sampling temperature (0.0 for deterministic)
            openai_api_key: OpenAI API key (uses env var if None)
            groq_api_key: GROQ API key (uses env var if None)
            **kwargs: Additional model parameters
        """
        self.provider = provider.lower()
        
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {SUPPORTED_PROVIDERS}")
        
        # Set default model if not provided
        if model_name is None:
            model_name = DEFAULT_MODELS[self.provider]
        
        self.model_name = model_name
        self.temperature = temperature
        
        # Get API keys
        if self.provider == "openai":
            self.api_key = openai_api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        elif self.provider == "groq":
            self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
            if not self.api_key:
                raise ValueError("GROQ API key is required. Set GROQ_API_KEY environment variable.")
        
        # Initialize LLM
        self.llm: BaseChatModel = self._create_llm(**kwargs)
        
        logger.info(
            f"LLMManager initialized | "
            f"Provider: {self.provider} | "
            f"Model: {self.model_name} | "
            f"Temperature: {self.temperature}"
        )

    def _create_llm(self, **kwargs) -> BaseChatModel:
        """Create LLM instance based on provider"""
        if self.provider == "openai":
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key,
                **kwargs
            )
        elif self.provider == "groq":
            return ChatGroq(
                model=self.model_name,
                temperature=self.temperature,
                groq_api_key=self.api_key,
                **kwargs
            )
        else:
            # Fallback to init_chat_model for other providers
            if init_chat_model is None:
                raise ValueError(f"Provider '{self.provider}' not directly supported and init_chat_model is not available")
            model_id = f"{self.provider}:{self.model_name}"
            return init_chat_model(model_id, temperature=self.temperature, **kwargs)

    def invoke(self, prompt: str) -> Any:
        """
        Invoke the LLM with a prompt.

        Args:
            prompt: Input prompt text

        Returns:
            LLM response
        """
        try:
            response = self.llm.invoke(prompt)
            logger.debug(f"LLM invoked | Provider: {self.provider} | Model: {self.model_name}")
            return response
        except Exception as e:
            logger.error(f"Error invoking LLM: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to invoke LLM: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "supported_providers": SUPPORTED_PROVIDERS
        }

    def get_llm(self) -> BaseChatModel:
        """Get the underlying LLM instance"""
        return self.llm


def get_llm_manager(
    provider: str = "openai",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    openai_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    **kwargs
) -> LLMManager:
    """
    Get or create an LLM manager instance.

    Args:
        provider: LLM provider
        model_name: Model name
        temperature: Sampling temperature
        openai_api_key: OpenAI API key
        groq_api_key: GROQ API key
        **kwargs: Additional model parameters

    Returns:
        LLMManager instance
    """
    return LLMManager(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        openai_api_key=openai_api_key,
        groq_api_key=groq_api_key,
        **kwargs
    )


def get_available_models() -> Dict[str, Dict[str, str]]:
    """Get all available models by provider"""
    return {
        "openai": OPENAI_MODELS,
        "groq": GROQ_MODELS
    }
