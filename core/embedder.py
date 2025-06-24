# core/embedder.py - Fixed version with better error handling
import os
import time
from typing import Optional, Any
import streamlit as st

def load_embedder_safe() -> Optional[Any]:
    """Load embedding model with comprehensive error handling and fallbacks"""
    models_to_try = [
        # Start with smallest, most reliable models
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2", 
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2"
    ]
    
    for model_name in models_to_try:
        try:
            # Import here to avoid issues if not available
            from sentence_transformers import SentenceTransformer
            
            # Load with timeout and specific settings for Streamlit Cloud
            model = SentenceTransformer(
                model_name, 
                device="cpu",
                cache_folder="./.cache",  # Use local cache
                use_auth_token=False
            )
            
            # Test the model with a simple encoding
            test_embedding = model.encode(["test"], convert_to_tensor=False)
            if test_embedding is not None and len(test_embedding) > 0:
                st.success(f"✅ Loaded embedding model: {model_name}")
                return model
                
        except Exception as e:
            st.warning(f"⚠️ Failed to load {model_name}: {str(e)[:100]}")
            continue
    
    # Ultimate fallback - create a dummy embedder that uses TF-IDF
    st.warning("Using fallback TF-IDF embedder")
    return create_tfidf_embedder()

def create_tfidf_embedder():
    """Create a simple TF-IDF based embedder as fallback"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    class TFIDFEmbedder:
        def __init__(self):
            self.vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
            self.fitted = False
            
        def encode(self, texts, convert_to_tensor=False, convert_to_numpy=True):
            if isinstance(texts, str):
                texts = [texts]
                
            if not self.fitted:
                # Fit on the input texts (not ideal but works for fallback)
                self.vectorizer.fit(texts)
                self.fitted = True
                
            embeddings = self.vectorizer.transform(texts).toarray()
            
            if convert_to_tensor:
                import torch
                return torch.tensor(embeddings)
            
            return embeddings
    
    return TFIDFEmbedder()

def load_summarizer_safe() -> Optional[Any]:
    """Load summarization model with fallbacks"""
    models_to_try = [
        "sshleifer/distilbart-cnn-6-6",  # Smaller, faster model
        "facebook/bart-large-cnn",
        "t5-small"
    ]
    
    for model_name in models_to_try:
        try:
            from transformers import pipeline
            
            summarizer = pipeline(
                "summarization",
                model=model_name,
                device=-1,  # CPU only
                torch_dtype="auto",
                model_kwargs={"cache_dir": "./.cache"}
            )
            
            # Test with simple text
            test_result = summarizer("This is a test sentence for summarization.", 
                                   max_length=20, min_length=5, do_sample=False)
            
            if test_result and len(test_result) > 0:
                st.success(f"✅ Loaded summarizer: {model_name}")
                return summarizer
                
        except Exception as e:
            st.warning(f"⚠️ Failed to load summarizer {model_name}: {str(e)[:100]}")
            continue
    
    # Fallback summarizer
    st.warning("Using fallback text summarizer")
    return create_fallback_summarizer()

def create_fallback_summarizer():
    """Create a simple extractive summarizer as fallback"""
    class SimpleSummarizer:
        def __call__(self, text, max_length=100, min_length=30, **kwargs):
            sentences = text.split('. ')
            if len(sentences) <= 2:
                return [{"summary_text": text}]
            
            # Simple extractive summary - take first and middle sentences
            summary_sentences = [sentences[0]]
            if len(sentences) > 2:
                summary_sentences.append(sentences[len(sentences)//2])
            
            summary = '. '.join(summary_sentences) + '.'
            return [{"summary_text": summary}]
    
    return SimpleSummarizer()

def load_ner_model_safe() -> Optional[Any]:
    """Load NER model with fallbacks"""
    
    # Try spaCy first
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        st.success("✅ Loaded spaCy NER model")
        return nlp
    except Exception as e:
        st.warning(f"⚠️ spaCy model not available: {str(e)[:100]}")
    
    # Try transformers NER pipeline
    try:
        from transformers import pipeline
        ner_pipeline = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english",
            device=-1,
            aggregation_strategy="simple",
            model_kwargs={"cache_dir": "./.cache"}
        )
        
        # Test the pipeline
        test_result = ner_pipeline("John Smith works at Google.")
        if test_result:
            st.success("✅ Loaded transformers NER model")
            return ner_pipeline
            
    except Exception as e:
        st.warning(f"⚠️ Transformers NER failed: {str(e)[:100]}")
    
    # Ultimate fallback - simple regex-based NER
    st.warning("Using fallback regex-based NER")
    return create_fallback_ner()

def create_fallback_ner():
    """Create a simple regex-based NER as fallback"""
    import re
    
    class SimpleNER:
        def __init__(self):
            # Simple patterns for common entities
            self.patterns = {
                'PERSON': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                'ORG': r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|University|College))\b',
                'GPE': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:City|State|Country|University))\b',
                'MONEY': r'\$[\d,]+(?:\.\d{2})?',
                'DATE': r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b'
            }
        
        def __call__(self, text):
            entities = []
            for label, pattern in self.patterns.items():
                matches = re.finditer(pattern, text)
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'label': label,
                        'start': match.start(),
                        'end': match.end()
                    })
            return entities
    
    return SimpleNER()

# Utility functions for embedder compatibility
def encode_texts(embedder, texts, **kwargs):
    """Universal encoding function that works with all embedder types"""
    try:
        if hasattr(embedder, 'encode'):
            return embedder.encode(texts, **kwargs)
        else:
            # Fallback for custom embedders
            return embedder.encode(texts)
    except Exception as e:
        st.error(f"Error encoding texts: {e}")
        # Return zero embeddings as last resort
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        return np.zeros((len(texts), 384))  # Standard embedding size

def extract_entities_safe(text, ner_model):
    """Safe entity extraction that works with different NER model types"""
    try:
        if hasattr(ner_model, 'nlp') or str(type(ner_model)).find('spacy') != -1:
            # spaCy model
            doc = ner_model(text)
            return [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        
        elif hasattr(ner_model, '__call__'):
            # Pipeline or callable model
            entities = ner_model(text)
            if isinstance(entities, list) and len(entities) > 0:
                # Normalize the output format
                normalized = []
                for ent in entities:
                    if isinstance(ent, dict):
                        normalized.append({
                            'text': ent.get('word', ent.get('text', '')),
                            'label': ent.get('entity', ent.get('label', ''))
                        })
                return normalized
        
        return []
        
    except Exception as e:
        st.warning(f"Entity extraction error: {e}")
        return []

# Model loading functions for backward compatibility
def load_embedder():
    """Backward compatibility function"""
    return load_embedder_safe()

def load_summarizer():
    """Backward compatibility function"""
    return load_summarizer_safe()

def load_ner_model():
    """Backward compatibility function"""
    return load_ner_model_safe()