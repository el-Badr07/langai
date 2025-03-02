from typing import Any, List, Dict, Optional, Union, Callable, TypeVar, Generic

T = TypeVar('T')
P = TypeVar('P')

class StrOutputParser:
    """
    Parser for extracting string output from various inputs.
    
    This class is commonly used in LLM processing pipelines to handle
    responses from language models and extract plain text.
    """
    
    def __init__(self, strip_whitespace: bool = True):
        """
        Initialize the string output parser.
        
        Args:
            strip_whitespace: Whether to strip whitespace from the output
        """
        self.strip_whitespace = strip_whitespace
    
    def parse(self, text: Any) -> str:
        """
        Parse the input into a string.
        
        Args:
            text: The input to parse, can be various types
            
        Returns:
            Parsed string
        """
        if text is None:
            return ""
            
        # Handle different input types
        if isinstance(text, dict):
            # Handle dictionary with common output fields
            if "output" in text:
                text = text["output"]
            elif "text" in text:
                text = text["text"]
            elif "content" in text:
                text = text["content"]
            elif "message" in text and isinstance(text["message"], dict):
                message = text["message"]
                if "content" in message:
                    text = message["content"]
            elif "result" in text:
                text = text["result"]
            elif "response" in text:
                text = text["response"]
            elif "choices" in text and isinstance(text["choices"], list) and len(text["choices"]) > 0:
                # Handle OpenAI API style responses
                choice = text["choices"][0]
                if isinstance(choice, dict):
                    if "message" in choice and "content" in choice["message"]:
                        text = choice["message"]["content"]
                    elif "text" in choice:
                        text = choice["text"]
            else:
                # Fall back to string representation 
                text = str(text)
        elif isinstance(text, list):
            if all(isinstance(item, str) for item in text):
                # Convert list of strings to a single string
                text = " ".join(text)
            elif len(text) > 0:
                # Take the first element if it exists
                return self.parse(text[0])
            else:
                return ""
            
        # Make sure we have a string
        text = str(text)
        
        # Strip whitespace if configured
        if self.strip_whitespace:
            text = text.strip()
            
        return text
    
    def __call__(self, text: Any) -> str:
        """
        Call method for functional usage.
        
        Args:
            text: The input to parse
            
        Returns:
            Parsed string
        """
        return self.parse(text)
    
    @staticmethod
    def get_format_instructions() -> str:
        """
        Get format instructions for the parser.
        
        Returns:
            Format instructions as a string
        """
        return "Your response should be a plain text string."
    
    def with_config(self, **kwargs) -> 'StrOutputParser':
        """
        Create a new parser with updated configuration.
        
        Args:
            **kwargs: Configuration options
            
        Returns:
            New parser instance
        """
        updated_parser = StrOutputParser(
            strip_whitespace=kwargs.get("strip_whitespace", self.strip_whitespace)
        )
        return updated_parser
    
    def pipe(self, func: Callable[[str], P]) -> 'PipelineParser[str, P]':
        """
        Create a pipeline with this parser and a function.
        
        Args:
            func: Function to apply to the parsed output
            
        Returns:
            Pipeline parser
        """
        return PipelineParser(self, func)


class PipelineParser(Generic[T, P]):
    """
    Parser that chains a parser with a transformation function.
    """
    
    def __init__(self, parser: Callable[[Any], T], func: Callable[[T], P]):
        """
        Initialize the pipeline parser.
        
        Args:
            parser: Initial parser
            func: Transformation function
        """
        self.parser = parser
        self.func = func
    
    def parse(self, input: Any) -> P:
        """
        Parse input by first applying the parser, then the function.
        
        Args:
            input: Input to parse
            
        Returns:
            Transformed output
        """
        parsed = self.parser(input) if callable(self.parser) else input
        return self.func(parsed)
    
    def __call__(self, input: Any) -> P:
        """
        Call method for functional usage.
        
        Args:
            input: Input to parse
            
        Returns:
            Transformed output
        """
        return self.parse(input)
    
    def pipe(self, func: Callable[[P], Any]) -> 'PipelineParser':
        """
        Extend the pipeline with another function.
        
        Args:
            func: Function to apply to the output
            
        Returns:
            Extended pipeline parser
        """
        return PipelineParser(self, func)

# Usage examples:
# parser = StrOutputParser()

# # Basic parsing
# result = parser.parse("Hello world")
# print(result)  # Output: "Hello world"
# # Parsing dictionary response
# result = parser.parse({"output": "Response text"})
# print(result)  # Output: "Response text"
# # Parsing list response
# result = parser.parse(["Response", "text"])
# print(result)  # Output: "Response text"
# # Parsing list of strings
# result = parser.parse(["Response", "text"])
# print(result)  # Output: "Response text"
# # Parsing nested dictionary
# result = parser.parse({"message": {"content": "AI response"}})
# print(result)  # Output: "AI response"
# # Parsing nested list
# result = parser.parse([{"message": {"content": "AI response"}}])
# print(result)  # Output: "AI response"
# # Parsing nested list of strings
# result = parser.parse([["AI", "response"]])
# print(result)  # Output: "AI response"
# # Parsing empty response
# result = parser.parse(None)
# print(result)  # Output: ""
# # Parsing empty list
# result = parser.parse([])
# print(result)  # Output: ""
# # Parsing empty dictionary
# result = parser.parse({})
# print(result)  # Output: ""
# # Parsing mixed list
# result = parser.parse([{"message": {"content": "AI response"}}, "Extra text"])
# print(result)  # Output: "AI response"
# # Parsing mixed dictionary
# result = parser.parse({"choices": [{"message": {"content": "AI response"}}]})
# print(result)  # Output: "AI response"

# # Parsing OpenAI style response
# result = parser.parse({
#     "choices": [{"message": {"content": "AI response"}}]
# })
# print(result)  # Output: "AI response"
# # Pipeline example
# def uppercase(text: str) -> str:
#     return text.upper()

# pipeline = parser.pipe(uppercase)
# result = pipeline.parse("hello")  # Returns "HELLO"
# print(result)  # Output: "HELLO"
# # Chaining multiple functions
# pipeline = parser.pipe(uppercase).pipe(lambda x: x + "!").pipe(lambda x: x * 2)
# result = pipeline.parse("hello")  # Returns "HELLO!HELLO!"
# print(result)  # Output: "HELLO!HELLO!"
# # Functional usage
# result = parser("Hello world")
# print(result)  # Output: "Hello world"
# # Format instructions
# instructions = StrOutputParser.get_format_instructions()
# print(instructions)  # Output: "Your response should be a plain text string."
# # Configuration example
# new_parser = parser.with_config(strip_whitespace=False)
# result = new_parser.parse("  Hello world  ")
# print(result)  # Output: "  Hello world  "



from typing import Any, List, Dict, Optional, Union, Callable, TypeVar, Generic

T = TypeVar('T')
P = TypeVar('P')

class StrOutputParser:
    """
    Parser for extracting string output from various inputs.
    
    This class is commonly used in LLM processing pipelines to handle
    responses from language models and extract plain text.
    """
    
    def __init__(self, strip_whitespace: bool = True):
        """
        Initialize the string output parser.
        
        Args:
            strip_whitespace: Whether to strip whitespace from the output
        """
        self.strip_whitespace = strip_whitespace
    
    def parse(self, text: Any) -> str:
        """
        Parse the input into a string.
        
        Args:
            text: The input to parse, can be various types
            
        Returns:
            Parsed string
        """
        if text is None:
            return ""
            
        # Handle different input types
        if isinstance(text, dict):
            # Handle dictionary with common output fields
            if "output" in text:
                text = text["output"]
            elif "text" in text:
                text = text["text"]
            elif "content" in text:
                text = text["content"]
            elif "message" in text and isinstance(text["message"], dict):
                message = text["message"]
                if "content" in message:
                    text = message["content"]
            elif "result" in text:
                text = text["result"]
            elif "response" in text:
                text = text["response"]
            elif "choices" in text and isinstance(text["choices"], list) and len(text["choices"]) > 0:
                # Handle OpenAI API style responses
                choice = text["choices"][0]
                if isinstance(choice, dict):
                    if "message" in choice and "content" in choice["message"]:
                        text = choice["message"]["content"]
                    elif "text" in choice:
                        text = choice["text"]
            else:
                # Fall back to string representation 
                text = str(text)
        elif isinstance(text, list):
            if all(isinstance(item, str) for item in text):
                # Convert list of strings to a single string
                text = " ".join(text)
            elif len(text) > 0:
                # Take the first element if it exists
                return self.parse(text[0])
            else:
                return ""
            
        # Make sure we have a string
        text = str(text)
        
        # Strip whitespace if configured
        if self.strip_whitespace:
            text = text.strip()
            
        return text
    
    def __call__(self, text: Any) -> str:
        """
        Call method for functional usage.
        
        Args:
            text: The input to parse
            
        Returns:
            Parsed string
        """
        return self.parse(text)
    
    @staticmethod
    def get_format_instructions() -> str:
        """
        Get format instructions for the parser.
        
        Returns:
            Format instructions as a string
        """
        return "Your response should be a plain text string."
    
    def with_config(self, **kwargs) -> 'StrOutputParser':
        """
        Create a new parser with updated configuration.
        
        Args:
            **kwargs: Configuration options
            
        Returns:
            New parser instance
        """
        updated_parser = StrOutputParser(
            strip_whitespace=kwargs.get("strip_whitespace", self.strip_whitespace)
        )
        return updated_parser
    
    def pipe(self, func: Callable[[str], P]) -> 'PipelineParser[str, P]':
        """
        Create a pipeline with this parser and a function.
        
        Args:
            func: Function to apply to the parsed output
            
        Returns:
            Pipeline parser
        """
        return PipelineParser(self, func)


class PipelineParser(Generic[T, P]):
    """
    Parser that chains a parser with a transformation function.
    """
    
    def __init__(self, parser: Callable[[Any], T], func: Callable[[T], P]):
        """
        Initialize the pipeline parser.
        
        Args:
            parser: Initial parser
            func: Transformation function
        """
        self.parser = parser
        self.func = func
    
    def parse(self, input: Any) -> P:
        """
        Parse input by first applying the parser, then the function.
        
        Args:
            input: Input to parse
            
        Returns:
            Transformed output
        """
        parsed = self.parser(input) if callable(self.parser) else input
        return self.func(parsed)
    
    def __call__(self, input: Any) -> P:
        """
        Call method for functional usage.
        
        Args:
            input: Input to parse
            
        Returns:
            Transformed output
        """
        return self.parse(input)
    
    def pipe(self, func: Callable[[P], Any]) -> 'PipelineParser':
        """
        Extend the pipeline with another function.
        
        Args:
            func: Function to apply to the output
            
        Returns:
            Extended pipeline parser
        """
        return PipelineParser(self, func)


import json
import requests
import time
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Iterator
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


class OllamaEmbeddings:
    """Embeddings implementation using Ollama API"""
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        timeout: Optional[float] = 120,
        options: Optional[Dict[str, Any]] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Initialize Ollama embeddings.
        
        Args:
            model: Name of the model to use
            base_url: Base URL of the Ollama API
            timeout: Request timeout in seconds
            options: Additional options to pass to the Ollama API
            dimensions: Optional dimension for embeddings
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.options = options or {}
        self.dimensions = dimensions
        
        if dimensions is not None:
            self.options["dimensions"] = dimensions
    
    def _make_embed_request(self, text: str) -> Dict[str, Any]:
        """Make a request to the Ollama embeddings API endpoint"""
        payload = {
            "model": self.model,
            "prompt": text,
        }
        
        if self.options:
            payload["options"] = self.options
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            try:
                if hasattr(e, "response") and e.response is not None:
                    error_details = e.response.json()
                    error_msg = f"{error_msg}: {error_details.get('error', '')}"
            except:
                pass
            raise RuntimeError(f"Ollama API embeddings request failed: {error_msg}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embeddings for each document
        """
        embeddings = []
        
        for text in texts:
            response = self._make_embed_request(text)
            embeddings.append(response.get("embedding", []))
            
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Get embedding for a query text.
        
        Args:
            text: Query text
            
        Returns:
            Embedding for the query
        """
        response = self._make_embed_request(text)
        return response.get("embedding", [])
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async version of embed_documents.
        
        Args:
            texts: List of document texts
            
        Returns:
            List of embeddings for each document
        """
        async def _embed_single(text):
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor, self._make_embed_request, text
                )
                return response.get("embedding", [])
        
        return await asyncio.gather(*[_embed_single(text) for text in texts])
    
    async def aembed_query(self, text: str) -> List[float]:
        """
        Async version of embed_query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding for the query
        """
        with ThreadPoolExecutor() as executor:
            response = await asyncio.get_event_loop().run_in_executor(
                executor, self._make_embed_request, text
            )
            return response.get("embedding", [])
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Dimension of the embeddings
        """
        # If dimensions were explicitly set, return that value
        if self.dimensions is not None:
            return self.dimensions
            
        # Otherwise, get a sample embedding to determine the dimension
        try:
            sample = self.embed_query("Sample text")
            return len(sample)
        except:
            # If we can't get a sample, return a default dimension
            return 4096  # Common dimension for many Ollama models


# Usage example:
# chat_model = ChatOllama(model="smollm")
# response = chat_model.chat([
#     {"role": "user", "content": "Hello, how are you?"}
# ])
# print(response["message"]["content"])

embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")





# # Time comparison between custom StrOutputParser and langchain's StrOutputParser
# # from langchain.output_parsers import StructuredOutputParser as LangchainParser
# from langchain.schema.output_parser import StrOutputParser as LangchainParser

# # Performance comparison between custom StrOutputParser and LangchainParser
# # to measure the efficiency of the custom implementation.
# import time

# # Test data
# test_cases = [
#     "Hello world",
#     {"output": "Response text"},
#     ["Response", "text"],
#     {"message": {"content": "AI response"}},
#     {"choices": [{"message": {"content": "AI response"}}]},
# ]

# # Custom parser
# custom_parser = StrOutputParser()
# start_time = time.time()
# for _ in range(1000000):
#     for case in test_cases:
#         _ = custom_parser.parse(case)
# custom_time = time.time() - start_time

# # Langchain parser
# lc_parser = LangchainParser()
# start_time = time.time()
# for _ in range(1000000):
#     for case in test_cases:
#         _ = lc_parser.parse(case)
# lc_time = time.time() - start_time

# print(f"Custom Parser Time: {custom_time:.8f} seconds")
# print(f"Langchain Parser Time: {lc_time:.8f} seconds")
# print(f"Custom Parser is {lc_time/custom_time:.3f}x faster than Langchain Parser")


# Custom Parser Time: 1.40135431 seconds
# Langchain Parser Time: 0.51000071 seconds
# Custom Parser is 0.364x faster than Langchain Parser



# # Usage examples of PipelineParser:
# # parser = StrOutputParser()
# pipe=PipelineParser(StrOutputParser(), lambda x: x.upper())
# # # Basic parsing
# # result = parser.parse("Hello world")
# # print(result)  # Output: "Hello world"
# print(pipe.parse("Hello world"))  # Output: "HELLO WORLD"
# # # Parsing dictionary response
# print(pipe({"output": "Response text"}))  # Output: "RESPONSE TEXT"
# # # Parsing list response
# print(pipe(["Response", "text"]))  # Output: "RESPONSE TEXT"
# # # Parsing nested dictionary
# print(pipe({"message": {"content": "AI response"}}))  # Output: "AI RESPONSE"
# # # Parsing nested list
# a=pipe.pipe(lambda x: x.upper()) # Output: "AI RESPONSE"
# print(a({"message": {"content": "AI response"}}))  # Output: "AI RESPONSE"



from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from enum import Enum
import re
from abc import ABC, abstractmethod


class RetrievalStrategy(str, Enum):
    """Enum for retrieval strategies"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents relevant to a query"""
        pass


class Retriever(BaseRetriever):
    """
    Retriever class for retrieving relevant documents based on queries.
    
    Supports multiple retrieval strategies:
    - semantic: Uses vector embeddings for similarity search
    - keyword: Uses keyword matching
    - hybrid: Combines semantic and keyword approaches
    
    Attributes:
        vector_store: The vector store for semantic search
        documents: List of documents with their metadata
        strategy: The retrieval strategy to use
        embedding_fn: Function to create embeddings from text
        rerank_fn: Optional function to rerank retrieved documents
    """
    
    def __init__(
        self,
        vector_store: Any = None,
        documents: List[Dict[str, Any]] = None,
        strategy: Union[str, RetrievalStrategy] = RetrievalStrategy.SEMANTIC,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        rerank_fn: Optional[Callable[[List[Dict[str, Any]], str], List[Dict[str, Any]]]] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store for semantic search
            documents: List of documents with their metadata
            strategy: The retrieval strategy to use
            embedding_fn: Function to create embeddings from text
            rerank_fn: Optional function to rerank retrieved documents
        """
        self.vector_store = vector_store
        self.documents = documents or []
        self.strategy = strategy if isinstance(strategy, RetrievalStrategy) else RetrievalStrategy(strategy)
        self.embedding_fn = embedding_fn
        self.rerank_fn = rerank_fn
        
        # Validate configuration based on strategy
        if self.strategy in [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID] and not vector_store:
            raise ValueError(f"Vector store is required for {self.strategy} retrieval strategy")
        
        if self.strategy in [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID] and not embedding_fn:
            raise ValueError(f"Embedding function is required for {self.strategy} retrieval strategy")

    def _semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: The query string
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with similarity scores
        """
        # Create embedding for the query
        query_embedding = self.embedding_fn(query)
        
        # Search the vector store
        results = self.vector_store.search(
            query_embedding, 
            top_k=top_k
        )
        
        return results

    def _keyword_search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            query: The query string
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with match scores
        """
        # Prepare keywords from the query
        keywords = re.findall(r'\w+', query.lower())
        keywords = [k for k in keywords if len(k) > 2]  # Filter out very short words
        
        results = []
        
        for doc in self.documents:
            # Calculate match score based on keyword frequency
            text = doc.get("content", "").lower()
            matches = 0
            for keyword in keywords:
                matches += text.count(keyword)
            
            if matches > 0:
                # Create a copy of the document with the score
                doc_copy = doc.copy()
                doc_copy["score"] = matches / len(text.split())  # Normalize by document length
                results.append(doc_copy)
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5, **kwargs) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword approaches.
        
        Args:
            query: The query string
            top_k: Number of results to return
            alpha: Weight for semantic search (1-alpha is the weight for keyword search)
            
        Returns:
            List of relevant documents with combined scores
        """
        # Get semantic results
        semantic_results = self._semantic_search(query, top_k=top_k*2)  # Get more results to combine
        semantic_dict = {doc["id"]: doc for doc in semantic_results}
        
        # Get keyword results
        keyword_results = self._keyword_search(query, top_k=top_k*2)
        keyword_dict = {doc["id"]: doc for doc in keyword_results}
        
        # Combine results
        combined_dict = {}
        
        # Process semantic results
        for doc_id, doc in semantic_dict.items():
            combined_dict[doc_id] = {
                **doc,
                "combined_score": doc["score"] * alpha
            }
        
        # Process keyword results
        for doc_id, doc in keyword_dict.items():
            if doc_id in combined_dict:
                combined_dict[doc_id]["combined_score"] += doc["score"] * (1-alpha)
            else:
                combined_dict[doc_id] = {
                    **doc,
                    "combined_score": doc["score"] * (1-alpha)
                }
        
        # Convert back to list and sort
        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined_results[:top_k]
    
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on the query using the configured strategy.
        
        Args:
            query: The query string
            top_k: Number of results to return
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of relevant documents
        """
        # Apply the selected strategy
        if self.strategy == RetrievalStrategy.SEMANTIC:
            results = self._semantic_search(query, top_k, **kwargs)
        elif self.strategy == RetrievalStrategy.KEYWORD:
            results = self._keyword_search(query, top_k, **kwargs)
        elif self.strategy == RetrievalStrategy.HYBRID:
            results = self._hybrid_search(query, top_k, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval strategy: {self.strategy}")
        
        # Apply reranking if available
        if self.rerank_fn and results:
            results = self.rerank_fn(results, query)
        
        return results[:top_k]
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retriever.
        
        Args:
            documents: List of documents to add
        """
        # Add to local document store
        self.documents.extend(documents)
        
        # If using semantic search, add to vector store
        if self.strategy in [RetrievalStrategy.SEMANTIC, RetrievalStrategy.HYBRID]:
            # Create embeddings and add to vector store
            for doc in documents:
                if "content" in doc and self.embedding_fn:
                    embedding = self.embedding_fn(doc["content"])
                    self.vector_store.add(
                        doc_id=doc.get("id"),
                        embedding=embedding,
                        metadata=doc
                    )



# Example of how to use the Retriever class

# First, set up a vector store and embedding function
# from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import uuid

# Simple vector store using FAISS
class SimpleVectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
        self.docs = {}
        
    def add(self, doc_id, embedding, metadata=None):
        if not doc_id:
            doc_id = str(uuid.uuid4())
        self.docs[doc_id] = metadata or {}
        self.index.add(np.array([embedding], dtype=np.float32))
        
    def search(self, query_embedding, top_k=5):
        D, I = self.index.search(np.array([query_embedding], dtype=np.float32), top_k)
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.docs):
                doc_id = list(self.docs.keys())[idx]
                doc = self.docs[doc_id].copy()
                doc["score"] = 1.0 / (1.0 + dist)  # Convert distance to similarity score
                doc["id"] = doc_id
                results.append(doc)
        return results


def test_retriever():
    # Set up embedding function
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model=embeddings_model
    embedding_fn = lambda text: model.embed_query(text)
    # embedding_fn = lambda text: model.encode(text)

    # Create vector store
    vector_store = SimpleVectorStore(dimension=model.get_dimension())  # MiniLM embedding size
    # Sample documents
    documents = [
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},
        {"id": "doc1", "content": "Python is a programming language.", "source": "wiki"},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence.", "source": "textbook"},
        {"id": "doc3", "content": "Neural networks are used in deep learning.", "source": "paper"},
        {"id": "doc4", "content": "Python is commonly used for data science and AI applications.", "source": "blog"},


    ]

    # Create retriever with hybrid strategy
    retriever = Retriever(
        vector_store=vector_store,
        documents=documents,
        strategy=RetrievalStrategy.HYBRID,
        embedding_fn=embedding_fn
    )

    # Add documents to the retriever
    retriever.add_documents(documents)

    # Retrieve documents
    results = retriever.retrieve("Python programming", top_k=4)
    for result in results:
        print(f"Document: {result['content']}")
        print(f"Score: {result.get('combined_score', result.get('score'))}")
        print("---")



from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from enum import Enum
import re
import uuid
from datetime import datetime

# Simulating the necessary libraries for a minimal working example
class SentenceTransformer:
    def encode(self, text):
        # Simplified embedding function that returns random vectors
        # In a real system, this would use an actual embedding model
        return np.random.rand(384).astype(np.float32)

class FaissIndex:
    def __init__(self, dimension):
        self.vectors = []
        self.dimension = dimension
        
    def add(self, vectors):
        self.vectors.extend(vectors)
        
    def search(self, query_vector, k):
        # Simple nearest neighbors search
        distances = []
        for vector in self.vectors:
            dist = np.linalg.norm(query_vector - vector)
            distances.append(dist)
        
        # Get indices of k smallest distances
        indices = np.argsort(distances)[:k]
        distances = [distances[i] for i in indices]
        
        return np.array([distances]), np.array([indices])

# Required classes from previous answers
class RetrievalStrategy(str, Enum):
    """Enum for retrieval strategies"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class ContextFormatting(str, Enum):
    """Enum for context formatting styles"""
    SIMPLE = "simple"
    MARKDOWN = "markdown"
    QA = "qa"
    STRUCTURED = "structured"

# Implementation of VectorStore
class VectorStore:
    """Vector store for semantic search"""
    
    def __init__(self, dimension=384):
        """Initialize vector store"""
        self.dimension = dimension
        self.index = FaissIndex(dimension)
        self.docs = {}
        
    def add(self, doc_id, embedding, metadata=None):
        """Add a document to the vector store"""
        if not doc_id:
            doc_id = str(uuid.uuid4())
        self.docs[doc_id] = metadata or {}
        self.index.add([embedding])
        return doc_id
        
    def search(self, query_embedding, top_k=5):
        """Search for similar documents"""
        D, I = self.index.search(query_embedding, top_k)
        results = []
        
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(self.docs):
                doc_id = list(self.docs.keys())[idx]
                doc = self.docs[doc_id].copy()
                doc["score"] = 1.0 / (1.0 + dist)  # Convert distance to similarity score
                doc["id"] = doc_id
                results.append(doc)
                
        return results

# Retriever class implementation
class Retriever:
    """Retriever for documents"""
    
    def __init__(
        self,
        vector_store=None,
        documents=None,
        strategy="hybrid",
        embedding_fn=None
    ):
        """Initialize retriever"""
        self.vector_store = vector_store
        self.documents = documents or []
        self.strategy = strategy
        self.embedding_fn = embedding_fn
        
    def _semantic_search(self, query, top_k=5):
        """Perform semantic search"""
        query_embedding = self.embedding_fn(query)
        return self.vector_store.search(query_embedding, top_k=top_k)
        
    def _keyword_search(self, query, top_k=5):
        """Perform keyword search"""
        keywords = re.findall(r'\w+', query.lower())
        keywords = [k for k in keywords if len(k) > 2]
        
        results = []
        for doc in self.documents:
            text = doc.get("content", "").lower()
            matches = sum(text.count(keyword) for keyword in keywords)
            
            if matches > 0:
                doc_copy = doc.copy()
                doc_copy["score"] = matches / max(1, len(text.split()))
                results.append(doc_copy)
                
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
        
    def _hybrid_search(self, query, top_k=5, alpha=0.5):
        """Perform hybrid search"""
        semantic_results = self._semantic_search(query, top_k*2)
        keyword_results = self._keyword_search(query, top_k*2)
        
        # Combine results
        combined_dict = {}
        
        for doc in semantic_results:
            doc_id = doc["id"]
            combined_dict[doc_id] = {**doc, "combined_score": doc["score"] * alpha}
            
        for doc in keyword_results:
            doc_id = doc["id"]
            if doc_id in combined_dict:
                combined_dict[doc_id]["combined_score"] += doc["score"] * (1-alpha)
            else:
                combined_dict[doc_id] = {**doc, "combined_score": doc["score"] * (1-alpha)}
                
        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined_results[:top_k]
        
    def retrieve(self, query, top_k=5):
        """Retrieve documents"""
        if self.strategy == "semantic":
            return self._semantic_search(query, top_k)
        elif self.strategy == "keyword":
            return self._keyword_search(query, top_k)
        elif self.strategy == "hybrid":
            return self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
    def add_documents(self, documents):
        """Add documents to retriever"""
        self.documents.extend(documents)
        
        if self.vector_store and self.embedding_fn:
            for doc in documents:
                if "content" in doc:
                    embedding = self.embedding_fn(doc["content"])
                    self.vector_store.add(doc.get("id"), embedding, doc)

# ContextGenerator class implementation
class ContextGenerator:
    """Context generator for LLMs"""
    
    def __init__(
        self,
        max_tokens=3000,
        formatting="markdown",
        token_counter=None,
        include_metadata=["source", "title"]
    ):
        """Initialize context generator"""
        self.max_tokens = max_tokens
        self.formatting = formatting
        self.include_metadata = include_metadata
        
        if token_counter is None:
            self.token_counter = lambda text: len(text.split())
        else:
            self.token_counter = token_counter
            
    def truncate_content(self, documents, query):
        """Truncate documents to fit token limit"""
        truncated_docs = []
        total_tokens = 0
        formatting_overhead = 100
        
        for doc in documents:
            doc_copy = {**doc}
            content = doc.get("content", "")
            content_tokens = self.token_counter(content)
            
            if total_tokens + content_tokens + formatting_overhead > self.max_tokens:
                # Truncate content
                words = content.split()
                remaining_tokens = self.max_tokens - total_tokens - formatting_overhead
                if remaining_tokens > 50:
                    truncated_words = words[:remaining_tokens]
                    truncated_content = " ".join(truncated_words) + "..."
                    doc_copy["content"] = truncated_content
                    truncated_docs.append(doc_copy)
                break
                
            doc_copy["content"] = content
            truncated_docs.append(doc_copy)
            total_tokens += content_tokens + formatting_overhead
            
        return truncated_docs
        
    def format_context(self, documents, query):
        """Format documents into a context string"""
        if not documents:
            return ""
            
        if self.formatting == "markdown":
            return self._format_markdown(documents)
        elif self.formatting == "qa":
            return self._format_qa(documents, query)
        else:
            return self._format_simple(documents)
            
    def _format_simple(self, documents):
        """Format as simple text"""
        parts = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            source = doc.get("source", f"Document {i+1}")
            parts.append(f"DOCUMENT {i+1} (Source: {source})")
            parts.append(content)
            parts.append("-" * 40)
        return "\n\n".join(parts)
        
    def _format_markdown(self, documents):
        """Format as markdown"""
        parts = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata_parts = []
            
            for field in self.include_metadata:
                if field in doc:
                    metadata_parts.append(f"**{field.title()}**: {doc[field]}")
                    
            parts.append(f"## Document {i+1}")
            if metadata_parts:
                parts.append("*" + " | ".join(metadata_parts) + "*")
            parts.append("")
            parts.append(content)
            parts.append("")
            
        return "\n".join(parts)
        
    def _format_qa(self, documents, query):
        """Format as QA"""
        parts = [f"Original Question: {query}\n"]
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            source = doc.get("source", f"Document {i+1}")
            parts.append(f"Information from {source}:")
            parts.append(content)
            
            metadata = []
            for field in self.include_metadata:
                if field in doc and field != "source":
                    metadata.append(f"{field}: {doc[field]}")
                    
            if metadata:
                parts.append("\n[Citation: " + "; ".join(metadata) + "]")
                
            parts.append("-" * 40)
            
        return "\n".join(parts)
        
    def generate_context(self, documents, query):
        """Generate context from documents"""
        truncated_docs = self.truncate_content(documents, query)
        return self.format_context(truncated_docs, query)

# Sample LLM interface
class SimpleLLM:
    def generate(self, prompt):
        # This would call an actual LLM API
        # For demo, just return a simple response
        if "Python" in prompt:
            return "Python is a programming language created by Guido van Rossum in 1991. It's known for its readability and versatility."
        else:
            return "I can answer questions based on the provided context."

# Main RAG Application
class RAGSystem:
    def __init__(self):
        # Set up sentence transformer (in reality, this would load a model)
        self.model = SentenceTransformer()
        self.embedding_fn = lambda text: self.model.encode(text)
        
        # Set up vector store
        self.vector_store = VectorStore(dimension=384)
        
        # Set up retriever
        self.retriever = Retriever(
            vector_store=self.vector_store,
            strategy="hybrid",
            embedding_fn=self.embedding_fn
        )
        
        # Set up context generator
        self.context_generator = ContextGenerator(
            max_tokens=500,
            formatting="markdown",
            include_metadata=["source", "title", "date"]
        )
        
        # Set up LLM
        self.llm = SimpleLLM()
        
    def add_documents(self, documents):
        """Add documents to the system"""
        # Process and add documents to the retriever
        self.retriever.add_documents(documents)
        print(f"Added {len(documents)} documents to the system")
        
    def answer_question(self, query, top_k=3):
        """Answer a question using RAG"""
        print(f"Question: {query}")
        print("Retrieving relevant documents...")
        
        # Retrieve documents
        docs = self.retriever.retrieve(query, top_k=top_k)
        print(f"Retrieved {len(docs)} documents")
        
        if not docs:
            return "No relevant information found to answer your question."
            
        # Generate context
        print("Generating context...")
        context = self.context_generator.generate_context(docs, query)
        
        # Create LLM prompt
        prompt = f"""Answer the following question based only on the provided context. 
If you cannot answer from the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate answer
        print("Generating answer...")
        answer = self.llm.generate(prompt)
        
        return answer


# Example usage
def main1():
    # Sample documents
    documents = [
        {
            "id": "doc1",
            "content": "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes code readability with its notable use of significant whitespace.",
            "source": "wikipedia.org",
            "title": "Python (programming language)",
            "date": "2023-05-20"
        },
        {
            "id": "doc2",
            "content": "Python 2.0 was released on October 16, 2000, with many new features. Python 3.0, released on December 3, 2008, was a major revision not completely backward-compatible with earlier versions.",
            "source": "python.org",
            "title": "History of Python",
            "date": "2022-10-05"
        },
        {
            "id": "doc3",
            "content": "JavaScript is a programming language that is one of the core technologies of the World Wide Web. JavaScript was invented by Brendan Eich in 1995.",
            "source": "webdev.com",
            "title": "JavaScript Basics",
            "date": "2023-01-15"
        },
        {
            "id": "doc4",
            "content": "Python has a large standard library, offering a range of facilities as indicated by its motto 'batteries included'. Python's simple, easy to learn syntax emphasizes readability, which reduces the cost of program maintenance.",
            "source": "python.org",
            "title": "Python Features",
            "date": "2023-02-10"
        }
    ]
    
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Add documents
    rag.add_documents(documents)
    
    # Answer a question
    query = "When was Python created and what are its main features?"
    answer = rag.answer_question(query, top_k=3)
    
    print("\nFinal Answer:")
    print(answer)
    
    # Try with different formatting
    print("\nTrying with QA formatting...")
    rag.context_generator.formatting = "qa"
    answer = rag.answer_question(query, top_k=3)
    
    print("\nFinal Answer (QA format):")
    print(answer)


from typing import List, Dict, Any, Optional
import litellm  # Make sure the litellm package is installed and configured

from typing import List, Dict, Any, Optional, Union
import litellm
import os

class LLMHandler:
    """
    Handles user choice of any LLM using litellm, including configuration options,
    text generation, and chat completions.

    Example usage:
        handler = LLMHandler(
            model="groq/llama3-70b-8192",
            api_key="gsk_your_api_key",
            config={"temperature": 0.7, "max_tokens": 150}
        )
        text = handler.generate_text("Tell me a joke.", max_tokens=100)
        chat = handler.generate_chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "How does a rocket work?"}
            ],
            max_tokens=200
        )
    """

    def __init__(
        self, 
        model: str, 
        api_key: Optional[str] = None, 
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LLMHandler.

        Args:
            model: The model name with provider prefix (e.g., "openai/gpt-3.5-turbo", "groq/llama3-70b-8192")
            api_key: API key for the provider
            config: Additional configuration options (e.g., temperature, max_tokens)
        """
        self.model = model
        
        # Set API key either from parameter or from environment variables
        if api_key:
            # Extract provider from model string
            provider = model.split('/')[0] if '/' in model else None
            if provider:
                os.environ[f"{provider.upper()}_API_KEY"] = api_key
            self.api_key = api_key
        
        self.config = config or {}

    def generate_text(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text based on a prompt using litellm's completion function.

        Args:
            prompt: The prompt string
            max_tokens: Optional override for maximum tokens
            temperature: Optional override for generation temperature

        Returns:
            The generated text as a string
        """
        params = self.config.copy()
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to text completion for models that don't support chat
            try:
                response = litellm.completion(
                    model=self.model,
                    prompt=prompt,
                    **params
                )
                return response.choices[0].text
            except Exception as e:
                raise Exception(f"Error generating text: {str(e)}")

    def generate_chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None, 
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a chat completion from a list of messages.

        Args:
            messages: A list of message dictionaries with keys like 'role' and 'content'
            max_tokens: Optional override for maximum tokens
            temperature: Optional override for generation temperature

        Returns:
            The chat response as a string
        """
        params = self.config.copy()
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            # If chat completion fails, try a fallback approach
            # by combining messages into a single prompt
            try:
                combined_prompt = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                    for msg in messages
                ])
                return self.generate_text(combined_prompt, max_tokens, temperature)
            except Exception as nested_e:
                raise Exception(f"Error generating chat: {str(e)}. Fallback error: {str(nested_e)}")

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the handler's configuration options.

        Args:
            new_config: A dictionary of configuration options to update
        """
        self.config.update(new_config)

def main1():
    # Initialize the handler with Groq as the provider.
    handler = LLMHandler(
        provider="groq",
        model="groq/llama-3.3-70b-versatile",  # Replace with the actual Groq model identifier
        api_key="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",
        config={"temperature": 0.7, "max_tokens": 150}
    )

    # Generate text using Groq.
    prompt = "Tell me a joke."
    generated_text = handler.generate_text(prompt, max_tokens=100)
    print("Generated Text:")
    print(generated_text)

    # # Generate a chat response using Groq.
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "How does a rocket work?"}
    # ]
    # chat_response = handler.generate_chat(messages, max_tokens=200)
    # print("\nChat Response:")
    # print(chat_response)


def main():
    # Get API key from environment variable or set it directly
    # api_key = os.environ.get("GROQ_API_KEY", "your_groq_api_key")
    api_key = "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"
    
    # Initialize the handler with a Groq model
    handler = LLMHandler(
        # model="groq/llama-3.3-70b-versatile",  # Use the correct model identifier
        model="ollama/smollm",
        api_key=api_key,
        config={"temperature": 0.7, "max_tokens": 150}
    )
    
    # Generate text using Groq
    prompt = "Tell me a joke."
    generated_text = handler.generate_text(prompt, max_tokens=100)
    print("Generated Text:")
    print(generated_text)
    
    # Generate a chat response using Groq
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How does a rocket work?"}
    ]
    chat_response = handler.generate_chat(messages, max_tokens=200)
    print("\nChat Response:")
    print(chat_response)

if __name__ == "__main__":
    main()