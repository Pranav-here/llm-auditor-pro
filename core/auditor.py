# core/auditor.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .vector_store import hybrid_search  # Add this import
from .embedder import load_embedder
import re
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher

def enhanced_audit(question: str, model_answer: str, embed_model, faiss_index, 
                  chunk_texts: List[str], ner_model=None, document_entities=None, 
                  mode='standard') -> Dict[str, Any]:
    """
    Enhanced audit function with multi-factor scoring
    """
    # Use hybrid search from vector_store.py
    top_chunks = hybrid_search(question, model_answer, embed_model, faiss_index, chunk_texts, top_k=5)
    
    if not top_chunks:
        return {
            'trust_score': 0,
            'confidence_level': 'Low',
            'best_chunk': 'No relevant content found',
            'similarity_score': 0,
            'explanation': 'No matching content found in the document'
        }
    
    # Get the best chunk
    best_chunk_data = top_chunks[0]
    best_chunk = best_chunk_data['chunk']
    similarity_score = best_chunk_data['score'] * 100
    
    # Multi-factor scoring with improved logic
    scores = calculate_multi_factor_score(
        question, model_answer, best_chunk, embed_model, 
        ner_model, document_entities, mode, top_chunks
    )
    
    # Calculate final trust score
    trust_score = calculate_trust_score(scores, mode)
    
    # Determine confidence level
    confidence_level = determine_confidence_level(trust_score, scores)
    
    # Generate explanation
    explanation = generate_explanation(scores, trust_score, mode)
    
    # Highlight relevant parts
    highlighted_chunk = highlight_relevant_content(best_chunk, model_answer, embed_model)
    
    result = {
        'trust_score': int(trust_score),
        'confidence_level': confidence_level,
        'best_chunk': best_chunk,
        'highlighted_chunk': highlighted_chunk,
        'similarity_score': similarity_score,
        'explanation': explanation,
        'score_breakdown': scores,
        'all_chunks': top_chunks
    }
    
    # Add entity analysis if available
    if ner_model and document_entities:
        entity_analysis = analyze_entities(model_answer, document_entities, ner_model)
        result['entity_analysis'] = entity_analysis
    
    return result

def calculate_multi_factor_score(question: str, answer: str, chunk: str, 
                               embed_model, ner_model=None, document_entities=None, 
                               mode='standard', all_chunks=None) -> Dict[str, float]:
    """Calculate multiple scoring factors with improved logic"""
    
    scores = {}
    
    # 1. Semantic similarity (improved)
    scores['semantic_similarity'] = calculate_semantic_similarity(answer, chunk, embed_model, all_chunks)
    
    # 2. Textual overlap (improved)
    scores['textual_overlap'] = calculate_improved_textual_overlap(answer, chunk)
    
    # 3. Entity matching
    if ner_model and document_entities:
        scores['entity_matching'] = calculate_entity_matching(answer, document_entities, ner_model)
    else:
        scores['entity_matching'] = 50  # Neutral score when not available
    
    # 4. Factual consistency (improved)
    scores['factual_consistency'] = calculate_improved_factual_consistency(answer, chunk, question)
    
    return scores

def calculate_semantic_similarity(answer: str, chunk: str, embed_model, all_chunks=None) -> float:
    """Improved semantic similarity calculation"""
    
    try:
        answer_emb = embed_model.encode([answer], convert_to_numpy=True)
        
        # Check similarity with best chunk
        chunk_emb = embed_model.encode([chunk], convert_to_numpy=True)
        best_sim = cosine_similarity(answer_emb, chunk_emb)[0][0]
        
        # Also check with other top chunks for better coverage
        if all_chunks and len(all_chunks) > 1:
            similarities = []
            for chunk_data in all_chunks[:3]:  # Top 3 chunks
                try:
                    chunk_text = chunk_data['chunk']
                    chunk_emb = embed_model.encode([chunk_text], convert_to_numpy=True)
                    sim = cosine_similarity(answer_emb, chunk_emb)[0][0]
                    similarities.append(sim)
                except:
                    continue
            
            if similarities:
                # Use the best similarity among top chunks
                best_sim = max(similarities)
        
        # Scale the score more favorably for good matches
        scaled_score = best_sim * 100
        
        # Boost scores that are reasonably good
        if scaled_score >= 40:
            scaled_score = min(100, scaled_score * 1.2)
        
        return max(0, scaled_score)
        
    except Exception:
        return 0

def calculate_improved_textual_overlap(answer: str, chunk: str) -> float:
    """Improved textual overlap calculation"""
    
    # Multiple approaches to textual similarity
    
    # 1. Sequence matching for phrase-level similarity
    seq_matcher = SequenceMatcher(None, answer.lower(), chunk.lower())
    sequence_ratio = seq_matcher.ratio() * 100
    
    # 2. Word-level overlap (improved)
    answer_words = set(re.findall(r'\b\w{3,}\b', answer.lower()))  # Only words 3+ chars
    chunk_words = set(re.findall(r'\b\w{3,}\b', chunk.lower()))
    
    if not answer_words:
        word_overlap = 0
    else:
        intersection = len(answer_words.intersection(chunk_words))
        # Use answer words as base (not union) for better scoring
        word_overlap = (intersection / len(answer_words)) * 100
    
    # 3. N-gram overlap for phrase detection
    ngram_overlap = calculate_ngram_overlap(answer, chunk)
    
    # 4. Key phrase matching
    key_phrase_score = calculate_key_phrase_matching(answer, chunk)
    
    # Combine scores with weights
    combined_score = (
        sequence_ratio * 0.2 +
        word_overlap * 0.3 +
        ngram_overlap * 0.3 +
        key_phrase_score * 0.2
    )
    
    return min(100, combined_score)

def calculate_ngram_overlap(answer: str, chunk: str, n=2) -> float:
    """Calculate n-gram overlap between answer and chunk"""
    
    def get_ngrams(text, n):
        words = re.findall(r'\b\w+\b', text.lower())
        return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    
    answer_ngrams = get_ngrams(answer, n)
    chunk_ngrams = get_ngrams(chunk, n)
    
    if not answer_ngrams:
        return 0
    
    intersection = len(answer_ngrams.intersection(chunk_ngrams))
    return (intersection / len(answer_ngrams)) * 100

def calculate_key_phrase_matching(answer: str, chunk: str) -> float:
    """Look for key phrases and important terms"""
    
    # Extract potential key phrases (2-4 words)
    answer_phrases = set()
    chunk_phrases = set()
    
    # Get phrases of 2-4 words
    for n in range(2, 5):
        answer_words = re.findall(r'\b\w+\b', answer.lower())
        chunk_words = re.findall(r'\b\w+\b', chunk.lower())
        
        for i in range(len(answer_words)-n+1):
            phrase = ' '.join(answer_words[i:i+n])
            answer_phrases.add(phrase)
            
        for i in range(len(chunk_words)-n+1):
            phrase = ' '.join(chunk_words[i:i+n])
            chunk_phrases.add(phrase)
    
    if not answer_phrases:
        return 0
    
    matches = len(answer_phrases.intersection(chunk_phrases))
    return (matches / len(answer_phrases)) * 100

def calculate_improved_factual_consistency(answer: str, chunk: str, question: str) -> float:
    """Improved factual consistency checking"""
    
    answer_lower = answer.lower()
    chunk_lower = chunk.lower()
    question_lower = question.lower()
    
    consistency_score = 100  # Start with perfect consistency
    penalty = 0
    
    # Check for direct contradictions
    contradictions = [
        # Yes/No contradictions
        ('yes' in answer_lower and 'not allowed' in chunk_lower),
        ('no' in answer_lower and 'required' in chunk_lower),
        ('allowed' in answer_lower and 'not allowed' in chunk_lower),
        ('required' in answer_lower and 'optional' in chunk_lower),
        ('must' in answer_lower and 'optional' in chunk_lower),
        ('can' in answer_lower and 'cannot' in chunk_lower),
        ('cannot' in answer_lower and 'can' in chunk_lower and 'cannot' not in chunk_lower),
    ]
    
    # Count actual contradictions
    for contradiction in contradictions:
        if contradiction:
            penalty += 20
    
    # Check for numeric contradictions
    answer_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', answer)
    chunk_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', chunk)
    
    if answer_numbers and chunk_numbers:
        # Look for contradictory numbers in similar contexts
        for ans_num in answer_numbers:
            for chunk_num in chunk_numbers:
                if abs(float(ans_num) - float(chunk_num)) > 0.1:
                    # Check if they're in similar context
                    ans_context = get_number_context(answer, ans_num)
                    chunk_context = get_number_context(chunk, chunk_num)
                    
                    if calculate_context_similarity(ans_context, chunk_context) > 0.5:
                        penalty += 15
    
    # Reduce penalty for ambiguous cases
    if 'gpa' in question_lower and 'gpa' in chunk_lower:
        penalty = max(0, penalty - 10)  # GPA questions are often complex
    
    return max(0, consistency_score - penalty)

def get_number_context(text: str, number: str) -> str:
    """Get context around a number"""
    words = text.split()
    for i, word in enumerate(words):
        if number in word:
            start = max(0, i-3)
            end = min(len(words), i+4)
            return ' '.join(words[start:end])
    return ""

def calculate_context_similarity(context1: str, context2: str) -> float:
    """Calculate similarity between two contexts"""
    words1 = set(context1.lower().split())
    words2 = set(context2.lower().split())
    
    if not words1 or not words2:
        return 0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

def calculate_entity_matching(answer: str, document_entities: List[Dict], ner_model) -> float:
    """Calculate how well entities in answer match document entities"""
    
    try:
        # Extract entities from answer
        answer_entities = extract_answer_entities(answer, ner_model)
        
        if not answer_entities or not document_entities:
            return 50  # Neutral score
        
        # Count matches
        doc_entity_texts = [e.get('text', '').lower() for e in document_entities]
        matches = 0
        
        for ans_entity in answer_entities:
            ans_text = ans_entity.get('text', '').lower()
            if ans_text in doc_entity_texts:
                matches += 1
        
        if len(answer_entities) == 0:
            return 50
            
        return (matches / len(answer_entities)) * 100
        
    except Exception:
        return 50

def extract_answer_entities(answer: str, ner_model):
    """Extract entities from answer text"""
    try:
        # This is a simplified version - implement based on your NER model
        # For now, return empty list
        return []
    except Exception:
        return []

def calculate_trust_score(scores: Dict[str, float], mode: str) -> float:
    """Calculate final trust score with improved weighting"""
    
    # Improved weights based on mode
    if mode == 'strict':
        weights = {
            'semantic_similarity': 0.35,
            'textual_overlap': 0.35,
            'entity_matching': 0.15,
            'factual_consistency': 0.15
        }
        base_threshold = 0.85
    elif mode == 'lenient':
        weights = {
            'semantic_similarity': 0.45,
            'textual_overlap': 0.30,
            'entity_matching': 0.10,
            'factual_consistency': 0.15
        }
        base_threshold = 0.75
    else:  # standard
        weights = {
            'semantic_similarity': 0.40,
            'textual_overlap': 0.35,
            'entity_matching': 0.12,
            'factual_consistency': 0.13
        }
        base_threshold = 0.80
    
    # Calculate weighted score
    trust_score = sum(scores.get(factor, 0) * weight 
                     for factor, weight in weights.items())
    
    # Apply intelligent adjustments based on score patterns
    semantic_score = scores.get('semantic_similarity', 0)
    textual_score = scores.get('textual_overlap', 0)
    factual_score = scores.get('factual_consistency', 0)
    
    # Boost score if both semantic and factual consistency are good
    if semantic_score >= 50 and factual_score >= 80:
        trust_score = min(100, trust_score * 1.15)
    
    # Boost score if textual overlap is very high (direct quotes/paraphrases)
    if textual_score >= 60:
        trust_score = min(100, trust_score * 1.10)
    
    # Penalty for factual inconsistencies
    if factual_score < 60:
        trust_score *= 0.85
    
    # Mode-specific final adjustments (more reasonable)
    if mode == 'strict':
        trust_score *= 0.95  # Slightly more conservative
    elif mode == 'lenient':
        trust_score = min(100, trust_score * 1.05)  # Slightly more generous
    
    return max(0, min(100, trust_score))

def determine_confidence_level(trust_score: float, scores: Dict[str, float]) -> str:
    """Determine confidence level with improved thresholds"""
    
    # Check score consistency
    score_values = [v for v in scores.values() if v > 0]
    if len(score_values) > 1:
        score_std = np.std(score_values)
        if score_std > 35:  # Very high variance
            return 'Low'
    
    # Improved thresholds
    if trust_score >= 75:
        return 'High'
    elif trust_score >= 55:
        return 'Medium'
    else:
        return 'Low'

def generate_explanation(scores: Dict[str, float], trust_score: float, mode: str) -> str:
    """Generate explanation for the trust score"""
    
    explanations = []
    
    # Semantic similarity
    sem_score = scores.get('semantic_similarity', 0)
    if sem_score >= 70:
        explanations.append("Strong semantic similarity with document content")
    elif sem_score >= 45:
        explanations.append("Moderate semantic similarity with document content")
    else:
        explanations.append("Low semantic similarity with document content")
    
    # Textual overlap
    text_score = scores.get('textual_overlap', 0)
    if text_score >= 50:
        explanations.append("Good textual overlap with source material")
    elif text_score >= 25:
        explanations.append("Some textual overlap with source material")
    else:
        explanations.append("Limited textual overlap with source material")
    
    # Entity matching
    entity_score = scores.get('entity_matching', 50)
    if entity_score >= 70:
        explanations.append("Mentioned entities align well with document")
    elif entity_score >= 40:
        explanations.append("Some entity alignment with document")
    else:
        explanations.append("Limited entity alignment with document")
    
    # Overall assessment with improved thresholds
    if trust_score >= 75:
        overall = "The answer appears to be well-supported by the document."
    elif trust_score >= 55:
        overall = "The answer has moderate support from the document."
    elif trust_score >= 35:
        overall = "The answer has limited support from the document."
    else:
        overall = "The answer appears to have little support from the document."
    
    return f"{overall} " + " ".join(explanations)

def highlight_relevant_content(chunk: str, answer: str, embed_model) -> str:
    """Highlight relevant parts of the chunk"""
    
    # Simple approach: highlight sentences with high similarity
    sentences = re.split(r'[.!?]+', chunk)
    
    if len(sentences) <= 1:
        return chunk
    
    try:
        answer_emb = embed_model.encode([answer], convert_to_numpy=True)[0]
        
        highlighted_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                highlighted_sentences.append(sentence)
                continue
                
            try:
                sent_emb = embed_model.encode([sentence], convert_to_numpy=True)[0]
                similarity = cosine_similarity([answer_emb], [sent_emb])[0][0]
                
                if similarity >= 0.4:  # Lower threshold for highlighting
                    highlighted_sentences.append(f"**{sentence}**")
                else:
                    highlighted_sentences.append(sentence)
            except Exception:
                highlighted_sentences.append(sentence)
        
        return ". ".join(highlighted_sentences)
        
    except Exception:
        return chunk

def analyze_entities(answer: str, document_entities: List[Dict], ner_model) -> Dict[str, Any]:
    """Analyze entity matching between answer and document"""
    
    try:
        # Extract entities from answer
        answer_entities = extract_answer_entities(answer, ner_model)
        
        # Find matches
        doc_entity_texts = [e.get('text', '').lower() for e in document_entities]
        matches = []
        
        for ans_entity in answer_entities:
            ans_text = ans_entity.get('text', '').lower()
            if ans_text in doc_entity_texts:
                matches.append(ans_text)
        
        return {
            'answer_entities': answer_entities,
            'document_entities': document_entities[:10],  # Limit for display
            'matches': matches,
            'match_ratio': len(matches) / max(1, len(answer_entities))
        }
        
    except Exception:
        return {
            'answer_entities': [],
            'document_entities': [],
            'matches': [],
            'match_ratio': 0
        }

def generate_detailed_report(audit_result: Dict[str, Any]) -> str:
    """Generate a detailed markdown report"""
    
    report = f"""# Audit Report

## Overall Assessment
- **Trust Score:** {audit_result['trust_score']}%
- **Confidence Level:** {audit_result['confidence_level']}
- **Similarity Score:** {audit_result.get('similarity_score', 0):.1f}%

## Score Breakdown
"""
    
    if 'score_breakdown' in audit_result:
        for factor, score in audit_result['score_breakdown'].items():
            report += f"- **{factor.replace('_', ' ').title()}:** {score:.1f}%\n"
    
    report += f"""
## Analysis
{audit_result.get('explanation', 'No explanation available')}

## Most Relevant Content
{audit_result.get('best_chunk', 'No relevant content found')}
"""
    
    if 'entity_analysis' in audit_result:
        entity_data = audit_result['entity_analysis']
        report += f"""
## Entity Analysis
- **Entities in Answer:** {len(entity_data.get('answer_entities', []))}
- **Matching Entities:** {len(entity_data.get('matches', []))}
- **Match Ratio:** {entity_data.get('match_ratio', 0):.2%}
"""
    
    return report