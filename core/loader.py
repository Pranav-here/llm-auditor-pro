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

def smart_chunk_text(text, target_size=800, overlap=200):
    """Intelligent chunking that respects sentence boundaries"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        
        # If adding this sentence would exceed target size and we have content
        if len(current_chunk) + len(sentence) > target_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Create proper overlap by going back
            overlap_text = ""
            overlap_length = 0
            j = i - 1
            
            # Build overlap from previous sentences
            while j >= 0 and overlap_length < overlap:
                sent_to_add = sentences[j]
                if overlap_length + len(sent_to_add) <= overlap:
                    overlap_text = sent_to_add + " " + overlap_text
                    overlap_length += len(sent_to_add)
                    j -= 1
                else:
                    break
            
            current_chunk = overlap_text + sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
        
        i += 1
    
    # Add final chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out chunks that are too small (less than 50 characters)
    chunks = [chunk for chunk in chunks if len(chunk) > 50]
    
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
