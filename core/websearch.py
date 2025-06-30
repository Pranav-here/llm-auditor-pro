# core/websearch.py
import os
import requests
import faiss
import numpy as np
from typing import List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.tavily.com/search"
API_KEY = os.getenv("TAVILY_API_KEY")

def tavily_search(query: str, k: int = 10) -> List[str]:
    """Return k short snippets relevant to the query using Tavily Search API."""
    if not API_KEY:
        raise RuntimeError("TAVILY_API_KEY not set")
    
    payload = {
        "query": query,
        "max_results": k,
        "search_depth": "advanced"  # balances cost vs depth
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = requests.post(BASE_URL, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        # Each result has a 'content' field (pre-scraped text, ~200 chars)
        results = []
        for item in data.get("results", []):
            content = item.get("content", "").strip()
            if content and len(content) > 20:  # Filter out very short snippets
                results.append(content)
        
        return results
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Tavily API request failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Error processing Tavily response: {e}")

def search_and_prepare_chunks(question: str, answer: str = "", k: int = 10) -> List[str]:
    """
    Search for relevant content and prepare chunks for auditing.
    
    Args:
        question: The user's question
        answer: The model's answer (optional, can help with context)
        k: Number of search results to retrieve
    
    Returns:
        List of content snippets ready for embedding
    """
    try:
        # Create search query - prioritize the question
        search_query = question
        
        # Optionally enhance with key terms from the answer
        if answer and len(answer.strip()) > 0:
            # Extract key terms from answer (simple approach)
            answer_words = answer.lower().split()
            # Add important words from answer if they're not already in question
            question_words = question.lower().split()
            for word in answer_words:
                if len(word) > 4 and word not in question_words and word.isalpha():
                    search_query += f" {word}"
                    if len(search_query.split()) >= 10:  # Limit query length
                        break
        
        # Perform search
        snippets = tavily_search(search_query, k)
        
        if not snippets:
            # Fallback: try with just the question
            snippets = tavily_search(question, k)
        
        # Clean and prepare snippets
        cleaned_snippets = []
        for snippet in snippets:
            # Basic cleaning
            cleaned = snippet.strip()
            if len(cleaned) > 30:  # Ensure minimum content length
                cleaned_snippets.append(cleaned)
        
        return cleaned_snippets[:k]  # Return up to k results
    
    except Exception as e:
        print(f"Error in search_and_prepare_chunks: {e}")
        return []

def create_web_search_index(snippets: List[str], embed_model) -> Tuple[Optional[object], Optional[List[str]]]:
    """
    Create a temporary FAISS index from web search snippets.
    
    Args:
        snippets: List of text snippets from web search
        embed_model: Embedding model for vectorization
    
    Returns:
        Tuple of (faiss_index, chunk_texts) or (None, None) if failed
    """
    try:
        if not snippets:
            return None, None
        
        # Generate embeddings for all snippets
        embeddings = embed_model.encode(snippets, show_progress_bar=False, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors = cosine similarity
        index.add(embeddings.astype('float32'))
        
        return index, snippets
    
    except Exception as e:
        print(f"Error creating web search index: {e}")
        return None, None

def format_web_search_results(snippets: List[str], scores: List[float] = None) -> str:
    """
    Format web search results for display.
    
    Args:
        snippets: List of content snippets
        scores: Optional relevance scores
    
    Returns:
        Formatted string for display
    """
    if not snippets:
        return "No relevant content found from web search."
    
    formatted_results = []
    for i, snippet in enumerate(snippets[:5]):  # Show top 5
        score_text = f" (Score: {scores[i]:.2f})" if scores and i < len(scores) else ""
        formatted_results.append(f"**Source {i+1}**{score_text}\n{snippet}\n")
    
    return "\n---\n".join(formatted_results)

def enhance_query_with_context(question: str, answer: str) -> str:
    """
    Enhance search query by combining question and answer context.
    
    Args:
        question: Original question
        answer: Model's answer to check
    
    Returns:
        Enhanced search query
    """
    # Start with the question
    enhanced_query = question
    
    if answer and len(answer.strip()) > 0:
        # Extract key phrases from answer
        answer_words = answer.split()
        
        # Add important terms (simple heuristic)
        important_words = []
        for word in answer_words:
            word_clean = word.strip('.,!?();:"').lower()
            if (len(word_clean) > 4 and 
                word_clean.isalpha() and 
                word_clean not in question.lower() and
                word_clean not in ['that', 'this', 'they', 'them', 'their', 'there', 'where', 'when', 'what', 'which', 'would', 'could', 'should']):
                important_words.append(word_clean)
        
        # Add top important words to query
        if important_words:
            enhanced_query += " " + " ".join(important_words[:3])
    
    return enhanced_query[:200]  # Limit query length