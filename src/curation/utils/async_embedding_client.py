import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from openai import AsyncOpenAI
import nest_asyncio
# from tqdm.asyncio import tqdm
from tqdm.notebook import tqdm
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncEmbeddingClient:
    """Async client for generating embeddings using OpenAI API"""
    def __init__(self, api_key: str, model: str, dimensions: int = 1536, 
                 max_chunk_size: int = 100, max_retries: int = 3,
                 min_seconds: float = 1, max_seconds: float = 60):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.max_chunk_size = max_chunk_size
        self.max_retries = max_retries
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=60))
    async def _get_chunk_embeddings(self, chunk: List[str]):
        """Get embeddings for a single chunk with retries"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=chunk,
            dimensions=self.dimensions
        )
        return response

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts with chunking and retries"""
        all_embeddings = []
        total_chunks = (len(texts) + self.max_chunk_size - 1) // self.max_chunk_size

        for i in tqdm(range(0, len(texts), self.max_chunk_size), total=total_chunks, desc="Fetching embeddings"):
            chunk = texts[i:i + self.max_chunk_size]
            try:
                response = await self._get_chunk_embeddings(chunk)
                
                logger.info(f"Generated embeddings for {len(chunk)} texts. "
                           f"Tokens used: {response.usage.total_tokens}")
                
                all_embeddings.extend([item.embedding for item in sorted(response.data, key=lambda x: x.index)])
            except Exception as e:
                logger.error(f"Failed to get embeddings after {self.max_retries} retries: {str(e)}")
                raise
        return all_embeddings

@dataclass
class ChatRequest:
    content: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class AsyncChatClient:
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-3.5-turbo",
                 max_chunk_size: int = 20,
                 max_retries: int = 3,
                 min_seconds: float = 1,
                 max_seconds: float = 60):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_chunk_size = max_chunk_size
        self.max_retries = max_retries
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=60))
    async def _get_chat_completion(self, request: ChatRequest) -> str:
        """Get chat completion for a single request with retries"""
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.content})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return response.choices[0].message.content

    async def get_chat_completions(self, requests: List[ChatRequest]) -> List[str]:
        """Get chat completions for multiple requests with chunking and retries"""
        all_responses = []
        total_chunks = (len(requests) + self.max_chunk_size - 1) // self.max_chunk_size

        for i in tqdm(range(0, len(requests), self.max_chunk_size), 
                     total=total_chunks, 
                     desc="Processing chat completions"):
            chunk = requests[i:i + self.max_chunk_size]
            chunk_responses = []
            
            # Process chunk in parallel
            tasks = [self._get_chat_completion(req) for req in chunk]
            try:
                chunk_responses = await asyncio.gather(*tasks)
                logger.info(f"Generated responses for {len(chunk)} requests")
                all_responses.extend(chunk_responses)
            except Exception as e:
                logger.error(f"Failed to get chat completions: {str(e)}")
                raise

        return all_responses

def run_async_safely(coro):
    """Helper to run async code synchronously, handling nested event loops"""
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if 'asyncio.run() cannot be called from a running event loop' in str(e):
            # For environments like Jupyter notebooks that may have nested loops
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(coro)
        raise

def embed(prompts, embedding_model="text-embedding-3-small", dimensions=1024, batch_size=500, api_key=None):
    """
    Get embeddings for a list of prompts using AsyncEmbeddingClient.
    Compatible with openai_api_parallelizer's embed function interface.
    
    Args:
        prompts (List[str]): List of text prompts to embed
        embedding_model (str): OpenAI embedding model name
        dimensions (int): Number of dimensions for the embeddings
        batch_size (int): Number of prompts to process in each batch
        api_key (str): OpenAI API key. If None, will try to get from environment
    
    Returns:
        List[Tuple[int, List[float], str]]: List of (id, embedding, prompt) tuples
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")

    client = AsyncEmbeddingClient(
        api_key=api_key,
        model=embedding_model,
        dimensions=dimensions,
        max_chunk_size=batch_size
    )
    
    embeddings = run_async_safely(client.get_embeddings(prompts))
    
    # Format results to match parallelizer output
    results = []
    for idx, (embedding, prompt) in enumerate(zip(embeddings, prompts)):
        results.append((idx, [embedding], prompt))
    
    return results

def chat(
    prompts: List[str],
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    chat_model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    batch_size: int = 20
) -> List[Tuple[int, List[str], str]]:
    """
    Process multiple chat prompts in parallel using OpenAI's chat models.
    
    Args:
        prompts: List of prompts to process
        system_prompt: Optional system prompt to use for all requests
        api_key: OpenAI API key
        chat_model: Model to use (default: gpt-3.5-turbo)
        temperature: Sampling temperature (default: 0.7)
        batch_size: Number of prompts to process in parallel (default: 20)
    
    Returns:
        List of tuples containing (index, [response], original_prompt)
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("No API key provided")

    # Initialize chat client
    client = AsyncChatClient(
        api_key=api_key,
        model=chat_model,
        max_chunk_size=batch_size
    )
    
    # Create chat requests
    requests = [
        ChatRequest(
            content=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        ) for prompt in prompts
    ]
    
    # Get completions
    responses = run_async_safely(client.get_chat_completions(requests))
    
    # Format results to match parallelizer output
    results = []
    for idx, (response, prompt) in enumerate(zip(responses, prompts)):
        results.append((idx, [response], prompt))
    
    return results

# Example usage in a Jupyter notebook
if __name__ == "__main__":
    # Initialize the client with your OpenAI API key
    api_key = "your-openai-api-key"
    client = AsyncEmbeddingClient(api_key=api_key)

    # Example texts to get embeddings for
    texts = ["Hello, how are you?", "The weather is nice today"] * 3000  # Example with more than 2048 texts

    # Fetch embeddings synchronously
    embeddings = run_sync(client.get_embeddings(texts))

    # Print the embeddings
    for i, embedding in enumerate(embeddings[:5]):  # Print first 5 embeddings for brevity
        print(f"Embedding for '{texts[i]}': {embedding[:5]}...")  # Print first 5 dimensions for brevity

    # Example usage with the embed function
    prompts = ["Hello world", "How are you?", "OpenAI embeddings"]
    results = embed(prompts)
    
    for pr_id, embedding, prompt in results[:2]:  # Show first 2 results
        print(f"ID: {pr_id}")
        print(f"Prompt: {prompt}")
        print(f"Embedding (first 5 dims): {embedding[0][:5]}...")
        print("---")

    # Example usage with the AsyncChatClient
    chat_client = AsyncChatClient(api_key=api_key)
    chat_requests = [
        ChatRequest(
            content="Tell me a joke",
            system_prompt="You are a helpful assistant",
            temperature=0.7
        ),
        ChatRequest(
            content="What is 2+2?",
            temperature=0
        )
    ]
    chat_responses = run_chat_sync(chat_client.get_chat_completions(chat_requests))
    
    for response in chat_responses:
        print(f"Chat response: {response}")

    # Example usage with the chat function
    prompts = [
        "Write a haiku about programming",
        "Explain what a closure is",
        "Tell me a short joke"
    ]
    
    results = chat(
        prompts=prompts,
        system_prompt="You are a helpful assistant",
        temperature=0.7
    )
    
    for idx, responses, prompt in results:
        print(f"\nPrompt: {prompt}")
        print(f"Response: {responses[0]}")