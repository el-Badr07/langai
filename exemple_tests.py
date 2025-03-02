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
parser = StrOutputParser()

# Basic parsing
result = parser.parse("Hello world")
print(result)  # Output: "Hello world"
# Parsing dictionary response
result = parser.parse({"output": "Response text"})
print(result)  # Output: "Response text"
# Parsing list response
result = parser.parse(["Response", "text"])
print(result)  # Output: "Response text"
# Parsing list of strings
result = parser.parse(["Response", "text"])
print(result)  # Output: "Response text"
# Parsing nested dictionary
result = parser.parse({"message": {"content": "AI response"}})
print(result)  # Output: "AI response"
# Parsing nested list
result = parser.parse([{"message": {"content": "AI response"}}])
print(result)  # Output: "AI response"
# Parsing nested list of strings
result = parser.parse([["AI", "response"]])
print(result)  # Output: "AI response"
# Parsing empty response
result = parser.parse(None)
print(result)  # Output: ""
# Parsing empty list
result = parser.parse([])
print(result)  # Output: ""
# Parsing empty dictionary
result = parser.parse({})
print(result)  # Output: ""
# Parsing mixed list
result = parser.parse([{"message": {"content": "AI response"}}, "Extra text"])
print(result)  # Output: "AI response"
# Parsing mixed dictionary
result = parser.parse({"choices": [{"message": {"content": "AI response"}}]})
print(result)  # Output: "AI response"

# Parsing OpenAI style response
result = parser.parse({
    "choices": [{"message": {"content": "AI response"}}]
})
print(result)  # Output: "AI response"
# Pipeline example
def uppercase(text: str) -> str:
    return text.upper()

pipeline = parser.pipe(uppercase)
result = pipeline.parse("hello")  # Returns "HELLO"
print(result)  # Output: "HELLO"
# Chaining multiple functions
pipeline = parser.pipe(uppercase).pipe(lambda x: x + "!").pipe(lambda x: x * 2)
result = pipeline.parse("hello")  # Returns "HELLO!HELLO!"
print(result)  # Output: "HELLO!HELLO!"
# Functional usage
result = parser("Hello world")
print(result)  # Output: "Hello world"
# Format instructions
instructions = StrOutputParser.get_format_instructions()
print(instructions)  # Output: "Your response should be a plain text string."
# Configuration example
new_parser = parser.with_config(strip_whitespace=False)
result = new_parser.parse("  Hello world  ")
print(result)  # Output: "  Hello world  "


# Time comparison between custom StrOutputParser and langchain's StrOutputParser
# from langchain.schema.output_parser import StrOutputParser as LangchainParser
import time

# Test data
test_cases = [
    "Hello world",
    {"output": "Response text"},
    ["Response", "text"],
    {"message": {"content": "AI response"}},
    {"choices": [{"message": {"content": "AI response"}}]},
]

# Custom parser
custom_parser = StrOutputParser()
start_time = time.time()
for _ in range(1000):
    for case in test_cases:
        _ = custom_parser.parse(case)
custom_time = time.time() - start_time

# Langchain parser
lc_parser = LangchainParser()
start_time = time.time()
for _ in range(1000):
    for case in test_cases:
        _ = lc_parser.parse(case)
lc_time = time.time() - start_time

print(f"Custom Parser Time: {custom_time:.4f} seconds")
print(f"Langchain Parser Time: {lc_time:.4f} seconds")
print(f"Custom Parser is {lc_time/custom_time:.2f}x faster than Langchain Parser")
import time

