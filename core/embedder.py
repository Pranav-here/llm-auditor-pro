# core/embedder.py
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import spacy

def load_embedder():
    """Load a better embedding model"""
    try:
        # Try to load a more powerful model
        return SentenceTransformer("all-mpnet-base-v2", device="cpu")
    except:
        # Fallback to original
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def load_summarizer():
    """Load summarization model"""
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def load_ner_model():
    """Load Named Entity Recognition model"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        return None