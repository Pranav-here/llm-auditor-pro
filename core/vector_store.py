# core/vector_store.py
import os
import faiss
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

INDEX_PATH = "data/faiss_index/chunks.index"
CHUNKS_PATH = "data/faiss_index/chunks.pkl"

def build_and_save_index(chunks, embed_model):
    """Build and save FAISS index with improved normalization"""
    os.makedirs("data/faiss_index", exist_ok=True)
    
    # Generate embeddings
    embeddings = embed_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors = cosine similarity
    index.add(embeddings.astype('float32'))
    
    # Save index and chunks
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_index_and_chunks():
    """Load existing FAISS index and chunks"""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None
    
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    
    return index, chunks

def hybrid_search(question, answer, embed_model, index, chunks, top_k=10):
    """Enhanced search combining question and answer context"""
    
    # Create combined query embedding
    question_emb = embed_model.encode([question], convert_to_numpy=True)[0]
    answer_emb = embed_model.encode([answer], convert_to_numpy=True)[0]
    
    # Weight question higher for retrieval
    combined_emb = (question_emb * 0.7 + answer_emb * 0.3)
    combined_emb = combined_emb / np.linalg.norm(combined_emb)
    
    # Search FAISS index
    scores, indices = index.search(np.array([combined_emb], dtype='float32'), top_k)
    
    # Return results with metadata
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(chunks):  # Ensure valid index
            results.append({
                'chunk': chunks[idx],
                'score': float(score),
                'rank': i + 1,
                'chunk_id': int(idx)
            })
    
    return results

def query_index(question, answer, embed_model, index, chunks, top_k=1):
    """Backward compatibility wrapper"""
    results = hybrid_search(question, answer, embed_model, index, chunks, top_k)
    if results:
        best_result = results[0]
        return best_result['chunk'], best_result['score'] * 100
    return "", 0