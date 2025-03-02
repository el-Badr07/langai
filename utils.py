import os
import json
import requests
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from enum import Enum
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """Enum for supported LLM providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    NVIDIA = "nvidia"
    ANTHROPIC = "anthropic"
    # Add more providers as needed

class LLMProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
    
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def generate_chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat response from messages"""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """Get embedding for text"""
        pass


class OpenAICompatibleProvider(LLMProvider):
    """Base provider implementation for OpenAI-compatible APIs"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, provider_name: str = "openai"):
        super().__init__(api_key, base_url)
        self.api_key = api_key
        self.base_url = base_url
        self.provider_name = provider_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, prompt: str, model: str = None, **kwargs) -> str:
        """Generate text using the completions endpoint"""
        if model is None:
            model = self.get_default_model("text")
            
        response = requests.post(
            f"{self.base_url}/completions",
            headers=self.headers,
            json={"model": model, "prompt": prompt, **kwargs}
        )
        response.raise_for_status()
        result = response.json()
        return result.get("choices", [{}])[0].get("text", "")
    
    def generate_chat(self, messages: List[Dict[str, str]], model: str = None, **kwargs) -> Dict[str, Any]:
        """Generate chat response using the chat completions endpoint"""
        if model is None:
            model = self.get_default_model("chat")
            
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={"model": model, "messages": messages, **kwargs}
        )
        response.raise_for_status()
        result = response.json()
        return {
            "content": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "role": result.get("choices", [{}])[0].get("message", {}).get("role", "assistant"),
            "finish_reason": result.get("choices", [{}])[0].get("finish_reason", "")
        }
    
    def get_embedding(self, text: str, model: str = None, **kwargs) -> List[float]:
        """Get embedding using the embeddings endpoint"""
        if model is None:
            model = self.get_default_model("embedding")
            
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json={"model": model, "input": text, **kwargs}
        )
        response.raise_for_status()
        return response.json().get("data", [{}])[0].get("embedding", [])
    
    def get_default_model(self, task_type: str) -> str:
        """Get the default model for a task"""
        if self.provider_name == "openai":
            if task_type == "text":
                return "gpt-3.5-turbo-instruct"
            elif task_type == "chat":
                return "gpt-3.5-turbo"
            else:  # embedding
                return "text-embedding-ada-002"
        elif self.provider_name == "groq":
            return "llama-3.1-8b-instant"
        elif self.provider_name == "deepseek":
            return "deepseek-llm-7b-chat"
        return "gpt-3.5-turbo"  # fallback


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        base_url = base_url or "https://api.openai.com/v1"
        super().__init__(api_key, base_url, provider_name="openai")


class GroqProvider(OpenAICompatibleProvider):
    """Groq provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        base_url = base_url or "https://api.groq.com/openai/v1"
        super().__init__(api_key, base_url, provider_name="groq")
    
    def get_default_model(self, task_type: str) -> str:
        """Get the default model based on task type"""
        if task_type in ["text", "chat"]:
            return "llama-3.1-8b-instant"
        return "llama-3.1-8b-instant"  # Default embedding model


class OllamaProvider(LLMProvider):
    """Ollama provider implementation (not OpenAI-compatible)"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key, base_url)
        self.base_url = base_url or "http://localhost:11434/api"
    
    def generate_text(self, prompt: str, model: str = "llama2", **kwargs) -> str:
        response = requests.post(
            f"{self.base_url}/generate",
            json={"model": model, "prompt": prompt, **kwargs}
        )
        response.raise_for_status()
        return response.json().get("response", "")
    
    def generate_chat(self, messages: List[Dict[str, str]], model: str = "llama2", **kwargs) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat",
            json={"model": model, "messages": messages, **kwargs}
        )
        response.raise_for_status()
        result = response.json()
        return {
            "content": result.get("message", {}).get("content", ""),
            "role": result.get("message", {}).get("role", "assistant"),
            "finish_reason": result.get("done", True)
        }
    
    def get_embedding(self, text: str, model: str = "llama2", **kwargs) -> List[float]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={"model": model, "prompt": text, **kwargs}
        )
        response.raise_for_status()
        return response.json().get("embedding", [])


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        base_url = base_url or "https://api.deepseek.com/v1"
        super().__init__(api_key, base_url, provider_name="deepseek")


class AnthropicProvider(OpenAICompatibleProvider):
    """Anthropic provider implementation"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        base_url = base_url or "https://api.anthropic.com/v1"
        super().__init__(api_key, base_url, provider_name="anthropic")


class LLMFactory:
    """Factory class to create LLM provider instances"""
    
    @staticmethod
    def create_provider(
        provider: Union[str, ProviderType],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> LLMProvider:
        """Create an LLM provider instance
        
        Args:
            provider: Provider name or ProviderType enum
            api_key: API key for the provider
            base_url: Base URL for the provider API
            
        Returns:
            LLMProvider: An instance of the requested provider
        """
        if isinstance(provider, str):
            try:
                provider = ProviderType(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported provider: {provider}")
        
        if provider == ProviderType.OPENAI:
            return OpenAIProvider(api_key=api_key, base_url=base_url)
        elif provider == ProviderType.OLLAMA:
            return OllamaProvider(api_key=api_key, base_url=base_url)
        elif provider == ProviderType.GROQ:
            return GroqProvider(api_key=api_key, base_url=base_url)
        elif provider == ProviderType.DEEPSEEK:
            return DeepSeekProvider(api_key=api_key, base_url=base_url)
        elif provider == ProviderType.ANTHROPIC:
            return AnthropicProvider(api_key=api_key, base_url=base_url)
        else:
            raise NotImplementedError(f"Provider {provider.value} is not implemented yet")


# Convenience function to get a provider
def get_provider(provider: Union[str, ProviderType], **kwargs) -> LLMProvider:
    """Get an LLM provider instance
    
    Args:
        provider: Provider name or ProviderType enum
        **kwargs: Additional arguments to pass to the provider constructor
        
    Returns:
        LLMProvider: An instance of the requested provider
    """
    return LLMFactory.create_provider(provider, **kwargs)


# Test the implementation with Groq
if __name__ == "__main__":
    try:
        # Replace with your actual Groq API key
        provider = get_provider(ProviderType.GROQ, api_key="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")
        
        # Test text generation
        print("Testing text generation...")
        text = provider.generate_text("Once upon a time", max_tokens=100)
        print(f"Generated text: {text}\n")
        
        # Test chat completion
        print("Testing chat completion...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I need help with my computer."}
        ]
        chat_response = provider.generate_chat(messages)
        print(f"Chat response: {chat_response}\n")
        
        # Test embeddings
        print("Testing embeddings...")
        embedding = provider.get_embedding("Hello, how are you?")
        print(f"Embedding (first 5 values): {embedding[:5]}\n")
        
    except Exception as e:
        print(f"Error: {str(e)}")
