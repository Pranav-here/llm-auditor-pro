# core/loader.py
import fitz
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import re

nltk_downloads = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
for item in nltk_downloads:
    try:
        nltk.data.find(f'tokenizers/{item}')
    except LookupError:
        nltk.download(item, quiet=True)

def extract_text_from_pdf(file):
    """Enhanced PDF text extraction with better formatting"""
    try:
        with fitz.open(stream=file.read(), filetype='pdf') as doc:
            text_blocks = []
            for page in doc:
                # Get text with layout preservation
                text = page.get_text()
                # Clean up common PDF artifacts
                text = re.sub(r'\n+', '\n', text)  # Multiple newlines
                text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces
                text_blocks.append(text)
            
            full_text = '\n\n'.join(text_blocks)
            return full_text if full_text.strip() else None
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None

def smart_chunk_text(text, target_size=500, overlap=100):
    """Intelligent chunking that respects sentence boundaries"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds target size and we have content
        if len(current_chunk) + len(sentence) > target_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Create overlap by keeping last few sentences
            overlap_sentences = current_chunk.split('. ')[-2:]  # Keep last 2 sentences
            current_chunk = '. '.join(overlap_sentences) + '. ' + sentence
        else:
            current_chunk += ' ' + sentence if current_chunk else sentence
    
    # Add final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_entities(text, ner_model):
    """Extract named entities from text"""
    if not ner_model:
        return []
    
    try:
        doc = ner_model(text)
        entities = [{'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char} 
                   for ent in doc.ents]
        return entities
    except Exception as e:
        print(f"Entity extraction error: {e}")
        return []
