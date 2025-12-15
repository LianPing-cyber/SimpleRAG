import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Union, Any, Awaitable
from openai import AsyncOpenAI # 导入异步客户端

OPENROUTER_CLIENT = AsyncOpenAI(
    base_url=os.environ.get("BASE_URL"),
    api_key=os.environ.get("API_KEY") 
)
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL")

RAGEntry = Dict[str, Union[List[float], str]] 

def chunk_text_by_char_limit(
    text: str, 
    max_chars_per_chunk: int, 
    overlap_chars: int = 200
) -> List[str]:
    if max_chars_per_chunk <= 0:
        raise ValueError("max_chars_per_chunk must be greater than 0.")
    if overlap_chars >= max_chars_per_chunk:
        raise ValueError("overlap_chars must be less than max_chars_per_chunk.")

    chunks = []
    text_len = len(text)
    start = 0
    
    while start < text_len:
        end = min(start + max_chars_per_chunk, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start += (max_chars_per_chunk - overlap_chars)
        
    return chunks

async def get_api_embeddings_async(texts: List[str], model_name: str) -> List[List[float]]:
    response = await OPENROUTER_CLIENT.embeddings.create(
            model=model_name,
            input=texts
        )        
    embeddings = [item.embedding for item in response.data]
    return embeddings
async def build_rag_library_api_async(
    long_context: str, 
    max_chars_per_entry: int = 10000
) -> List[RAGEntry]:
    overlap_chars = int(max_chars_per_entry * 0.05) # 5% 的重叠
    chunks = chunk_text_by_char_limit(
        long_context, 
        max_chars_per_entry, 
        overlap_chars
    )
    print(f"Text chunked into {len(chunks)} RAG entries (chunks).")
    if not chunks:
        return []
    all_embeddings = await get_api_embeddings_async(chunks, EMBEDDING_MODEL_NAME)
    if not all_embeddings:
        return []
    rag_library: List[RAGEntry] = []
    if len(chunks) != len(all_embeddings):
        print("Error: Embedding count does not match chunk count.")
        return []
    for i, chunk in enumerate(chunks):
        rag_library.append({
            "embedding": all_embeddings[i],  # List[float]
            "text": chunk
        })
    return rag_library


async def get_query_embedding(query: str, model_name: str) -> List[float]:
    response = await OPENROUTER_CLIENT.embeddings.create(
            model=model_name,
            input=[query]
        )
    embedding_vector = response.data[0].embedding
    return embedding_vector

async def retrieve_context_from_rag_lib(
    query: str, 
    rag_lib: List[RAGEntry], 
    top_k: int = 3
):
    if not rag_lib:
        return []
    query_vector_list = await get_query_embedding(query, EMBEDDING_MODEL_NAME)
    query_vector = np.array(query_vector_list)
    
    library_embeddings_list = [entry['embedding'] for entry in rag_lib]
    if not all(isinstance(v, list) for v in library_embeddings_list):
        print("Warning: the embedding vector in RAG library is not consistent to your selected embedding model.")

    library_matrix = np.array(library_embeddings_list)
    scores = np.dot(library_matrix, query_vector)
    chunk_scores = list(zip([entry['text'] for entry in rag_lib], scores))
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    retrieved_chunks = chunk_scores[:top_k]
    return retrieved_chunks

