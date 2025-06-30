# core/auditor.py - IMPROVED VERSION
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re
from typing import Dict, List, Any, Optional, Tuple

# Add this near the top after imports
TOP_K = 10  # global tweakable constant for chunk retrieval

def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """Extract key phrases from text using simple heuristics"""
    # Remove common stop words and extract meaningful phrases
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    
    # Simple phrase extraction
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    phrases = [word for word in words if word not in stop_words]
    
    # Return unique phrases
    return list(dict.fromkeys(phrases))[:max_phrases]

def calculate_semantic_similarity(question: str, answer: str, chunk: str, embed_model) -> float:
    """Calculate semantic similarity between question+answer and chunk"""
    try:
        # Create embeddings
        question_emb = embed_model.encode([question], convert_to_numpy=True)[0]
        answer_emb = embed_model.encode([answer], convert_to_numpy=True)[0]
        chunk_emb = embed_model.encode([chunk], convert_to_numpy=True)[0]
        
        # Combined query embedding (weighted towards question)
        combined_emb = (question_emb * 0.7 + answer_emb * 0.3)  # CHANGED: More weight to answer
        
        # Calculate cosine similarity
        similarity = cosine_similarity([combined_emb], [chunk_emb])[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}")
        return 0.0

def calculate_lexical_overlap(answer: str, chunk: str) -> float:
    """Calculate improved lexical overlap with length normalization"""
    try:
        answer_phrases = set(extract_key_phrases(answer, max_phrases=20))
        chunk_phrases = set(extract_key_phrases(chunk, max_phrases=50))
        
        if not answer_phrases:
            return 0.0
        
        # Basic overlap
        overlap = len(answer_phrases & chunk_phrases)
        basic_score = overlap / len(answer_phrases)
        
        # CHANGED: More lenient length penalty
        answer_length = len(answer.split())
        if answer_length < 5:
            length_penalty = 0.9  # Less harsh penalty
        elif answer_length < 10:
            length_penalty = 0.95
        else:
            length_penalty = 1.0
        
        # Weighted overlap based on phrase importance
        weighted_overlap = 0
        for phrase in answer_phrases:
            if phrase in chunk_phrases:
                # CHANGED: More generous weighting
                weight = min(2.5, len(phrase) / 4.0)  # Higher max weight, lower divisor
                weighted_overlap += weight
        
        if answer_phrases:
            weighted_score = weighted_overlap / sum(min(2.5, len(p) / 4.0) for p in answer_phrases)
        else:
            weighted_score = 0
        
        # CHANGED: More balanced combination
        final_score = (basic_score * 0.5 + weighted_score * 0.5) * length_penalty
        return min(1.0, final_score)
        
    except Exception as e:
        print(f"Error calculating lexical overlap: {e}")
        return 0.0

def calculate_answer_specificity(answer: str, chunk: str) -> float:
    """Calculate how specific/detailed the answer is relative to the source"""
    try:
        # CHANGED: More comprehensive specificity indicators
        answer_specifics = len(re.findall(r'\b\d+\.?\d*\b|\b[A-Z][a-z]+ [A-Z][a-z]+\b|\b\d{4}\b|\b[A-Z]{2,}\b', answer))
        chunk_specifics = len(re.findall(r'\b\d+\.?\d*\b|\b[A-Z][a-z]+ [A-Z][a-z]+\b|\b\d{4}\b|\b[A-Z]{2,}\b', chunk))
        
        if chunk_specifics == 0:
            return 0.6  # CHANGED: More generous neutral score
        
        # Ratio of specific elements
        specificity_ratio = min(1.0, answer_specifics / chunk_specifics)
        
        # CHANGED: Bigger bonus for having specifics
        if answer_specifics > 0:
            specificity_ratio += 0.3
        
        return min(1.0, specificity_ratio)
        
    except Exception as e:
        return 0.6  # CHANGED: Higher default

def calculate_answer_completeness(answer: str, question: str) -> float:
    """Estimate how complete the answer is relative to the question"""
    try:
        # Simple heuristics for completeness
        answer_length = len(answer.split())
        question_words = set(extract_key_phrases(question))
        answer_words = set(extract_key_phrases(answer))
        
        # Check if answer addresses question keywords
        keyword_coverage = len(question_words & answer_words) / max(1, len(question_words))
        
        # CHANGED: More generous length-based completeness
        if answer_length < 5:
            length_score = answer_length / 8.0  # Less harsh for short answers
        elif answer_length < 20:
            length_score = 0.6 + (answer_length - 5) / 25.0  # Higher base
        else:
            length_score = min(1.0, 0.85 + (answer_length - 20) / 80.0)  # Higher ceiling
        
        return (keyword_coverage * 0.6 + length_score * 0.4)  # CHANGED: More weight to length
        
    except Exception as e:
        return 0.6  # CHANGED: Higher default

def analyze_entities(answer: str, chunk: str, ner_model=None, document_entities: List = None) -> Dict[str, Any]:
    """Analyze entity matching between answer and chunk"""
    entity_analysis = {
        'answer_entities': [],
        'chunk_entities': [],
        'matches': [],
        'match_ratio': 0.0,
        'precision': 0.0
    }
    
    if not ner_model:
        return entity_analysis
    
    try:
        # Extract entities from answer
        answer_entities = ner_model(answer)
        if hasattr(answer_entities, 'ents'):
            entity_analysis['answer_entities'] = [
                {'text': ent.text, 'label': ent.label_} 
                for ent in answer_entities.ents
            ]
        
        # Extract entities from chunk
        chunk_entities = ner_model(chunk)
        if hasattr(chunk_entities, 'ents'):
            entity_analysis['chunk_entities'] = [
                {'text': ent.text, 'label': ent.label_} 
                for ent in chunk_entities.ents
            ]
        
        # Find matches with more lenient matching
        answer_entity_texts = {ent['text'].lower() for ent in entity_analysis['answer_entities']}
        chunk_entity_texts = {ent['text'].lower() for ent in entity_analysis['chunk_entities']}
        
        matches = set()
        for answer_entity in answer_entity_texts:
            for chunk_entity in chunk_entity_texts:
                # CHANGED: More lenient fuzzy matching
                if (answer_entity == chunk_entity or
                    answer_entity in chunk_entity or chunk_entity in answer_entity or
                    answer_entity.replace(" ", "").lower() == chunk_entity.replace(" ", "").lower() or
                    abs(len(answer_entity) - len(chunk_entity)) <= 2 and 
                    sum(a == b for a, b in zip(answer_entity, chunk_entity)) / max(len(answer_entity), len(chunk_entity)) > 0.8):
                    matches.add(answer_entity)

        matches = list(matches)
        entity_analysis['matches'] = list(matches)
        
        # Calculate match ratio (recall)
        if answer_entity_texts:
            entity_analysis['match_ratio'] = len(matches) / len(answer_entity_texts)
        
        # Calculate precision
        if answer_entity_texts:
            entity_analysis['precision'] = len(matches) / len(answer_entity_texts)
        
    except Exception as e:
        print(f"Error in entity analysis: {e}")
    
    return entity_analysis

def apply_nonlinear_scaling(score: float, curve_type: str = "sigmoid") -> float:
    """Apply non-linear scaling to make scores more discriminating"""
    if curve_type == "sigmoid":
        # CHANGED: Much gentler sigmoid curve
        return 1 / (1 + np.exp(-5 * (score - 0.4)))  # Less steep, lower threshold
    elif curve_type == "power":
        # CHANGED: Less harsh power curve
        return 0.9 * (score ** 1.05) + 0.2  # Reduced from 1.5
    elif curve_type == "threshold":
        # CHANGED: More generous threshold scaling
        if score < 0.25:  # Lowered from 0.3
            length_penalty = 0.7  # Less harsh
        elif score < 0.5:  # Lowered from 0.6
            return 0.2 + (score - 0.25) * 0.8  # More generous
        else:
            return 0.4 + (score - 0.5) * 1.2  # Better rewards
    else:
        return score

def enhanced_audit(
    question: str,
    model_answer: str,
    embed_model,
    faiss_index,
    chunk_texts: List[str],
    ner_model=None,
    document_entities: List = None,
    mode: str = "standard"
) -> Dict[str, Any]:
    """
    Enhanced audit function with improved calibration
    """
    if not question or not model_answer or not chunk_texts:
        return {
            'trust_score': 0,
            'confidence_level': 'Low',
            'explanation': 'Invalid input parameters',
            'best_chunk': '',
            'all_chunks': []
        }
    
    try:
        # Search for relevant chunks
        from core.vector_store import hybrid_search
        search_results = hybrid_search(
        question=question,
        answer=model_answer,
        embed_model=embed_model,
        index=faiss_index,
        chunks=chunk_texts,
        top_k=min(TOP_K, len(chunk_texts))
        )
        
        if not search_results:
            return {
                'trust_score': 0,
                'confidence_level': 'Low',
                'explanation': 'No relevant content found',
                'best_chunk': '',
                'all_chunks': []
            }
        
        # Analyze all top chunks and aggregate scores
        chunk_analyses = []
        total_semantic = 0
        total_lexical = 0
        total_specificity = 0
        total_entity = 0
        best_entity_analysis = None
        
        for i, result in enumerate(search_results):
            chunk = result['chunk']
            chunk_semantic = calculate_semantic_similarity(question, model_answer, chunk, embed_model)
            chunk_lexical = calculate_lexical_overlap(model_answer, chunk)
            chunk_specificity = calculate_answer_specificity(model_answer, chunk)
            chunk_entity_analysis = analyze_entities(model_answer, chunk, ner_model, document_entities)
            chunk_entity_score = (chunk_entity_analysis['match_ratio'] + chunk_entity_analysis['precision']) / 2
            
            # Weight by rank (higher weight for better ranked chunks)
            weight = 1.0 / (i + 1)
            
            total_semantic += chunk_semantic * weight
            total_lexical += chunk_lexical * weight
            total_specificity += chunk_specificity * weight
            total_entity += chunk_entity_score * weight
            
            chunk_analyses.append({
                'chunk': chunk,
                'semantic_score': chunk_semantic,
                'lexical_score': chunk_lexical,
                'specificity_score': chunk_specificity,
                'entity_score': chunk_entity_score,
                'weight': weight
            })
            
            # Keep the best entity analysis
            if i == 0 or chunk_entity_score > total_entity:
                best_entity_analysis = chunk_entity_analysis
        
        # Normalize aggregated scores
        total_weights = sum(1.0 / (i + 1) for i in range(len(search_results)))
        semantic_score = total_semantic / total_weights
        lexical_score = total_lexical / total_weights
        specificity_score = total_specificity / total_weights
        entity_score = total_entity / total_weights
        entity_analysis = best_entity_analysis
        
        # Use best chunk for display purposes
        best_result = search_results[0]
        best_chunk = best_result['chunk']
        retrieval_score = best_result['score']
        completeness_score = calculate_answer_completeness(model_answer, question)
        
        # CHANGED: Rebalanced mode-based weighting with higher entity weight
        if mode == "strict":
            weights = {
                'semantic': 0.25,      # Reduced from 0.30
                'lexical': 0.20,       # Reduced from 0.25
                'entity': 0.30,        # Increased from 0.20
                'specificity': 0.10,   # Same
                'completeness': 0.15   # Same
            }
            scaling_curve = "threshold"
            base_threshold = 0.35  # Lowered from 0.4
        elif mode == "lenient":
            weights = {
                'semantic': 0.25,      # Reduced from 0.40
                'lexical': 0.10,       # Reduced from 0.20
                'entity': 0.35,        # Increased from 0.15
                'specificity': 0.20,   # Increased from 0.15
                'completeness': 0.10   # Same
            }
            scaling_curve = "power"
            base_threshold = 0.20  # Lowered from 0.25
        else:  # standard
            weights = {
                'semantic': 0.25,      # Reduced from 0.35
                'lexical': 0.20,       # Reduced from 0.25
                'entity': 0.25,        # Increased from 0.15
                'specificity': 0.20,   # Increased from 0.15
                'completeness': 0.10   # Same
            }
            scaling_curve = "sigmoid"
            base_threshold = 0.30  # Lowered from 0.35
        
        # Calculate weighted score
        raw_score = (
            semantic_score * weights['semantic'] +
            lexical_score * weights['lexical'] +
            entity_score * weights['entity'] +
            specificity_score * weights['specificity'] +
            completeness_score * weights['completeness']
        )

        # BONUS: Mode-aware boost for strong factual alignment
        if entity_score > 0.5 and semantic_score > 0.5:
            if mode == "strict":
                raw_score += 0.035
            elif mode == "standard":
                raw_score += 0.05
            elif mode == "lenient":
                raw_score += 0.07
        elif entity_score > 0.5 and specificity_score > 0.6:
            if mode == "strict":
                raw_score += 0.025
            elif mode == "standard":
                raw_score += 0.04
            elif mode == "lenient":
                raw_score += 0.055



        
        # Apply non-linear scaling for better discrimination
        scaled_score = apply_nonlinear_scaling(raw_score, scaling_curve)
        
        # CHANGED: More generous retrieval quality adjustment
        if retrieval_score > 0.7:  # Lowered threshold
            retrieval_boost = 0.08  # Increased boost
        elif retrieval_score > 0.5:  # Lowered threshold
            retrieval_boost = 0.05  # Increased boost
        else:
            retrieval_boost = 0.02  # Small boost even for lower scores
        
        # Final trust score with adjusted thresholds
        trust_score = min(100, max(0, (scaled_score + retrieval_boost) * 100))
        
        # CHANGED: More balanced confidence levels
        if trust_score >= 80:  # Lowered from 85
            confidence_level = 'Very High'
        elif trust_score >= 65:  # Lowered from 70
            confidence_level = 'High'
        elif trust_score >= 45:  # Lowered from 50
            confidence_level = 'Medium'
        elif trust_score >= 25:  # Lowered from 30
            confidence_level = 'Low'
        else:
            confidence_level = 'Very Low'
        
        # Generate explanation
        explanation = generate_explanation(
            trust_score=trust_score,
            semantic_score=semantic_score,
            lexical_score=lexical_score,
            entity_score=entity_score,
            specificity_score=specificity_score,
            completeness_score=completeness_score,
            entity_analysis=entity_analysis,
            mode=mode
        )
        
        # Prepare detailed results
        result = {
            'trust_score': round(trust_score, 1),
            'confidence_level': confidence_level,
            'explanation': explanation,
            'best_chunk': best_chunk,
            'highlighted_chunk': highlight_relevant_parts(best_chunk, model_answer),
            'all_chunks': search_results,
            'score_breakdown': {
                'Semantic Similarity': round(semantic_score * 100, 1),
                'Lexical Overlap': round(lexical_score * 100, 1),
                'Entity Matching': round(entity_score * 100, 1),
                'Answer Specificity': round(specificity_score * 100, 1),
                'Answer Completeness': round(completeness_score * 100, 1),
                'Retrieval Quality': round(retrieval_score * 100, 1)
            },
            'entity_analysis': entity_analysis,
            'metadata': {
                'mode': mode,
                'timestamp': datetime.now().isoformat(),
                'chunks_analyzed': len(search_results),
                'weights_used': weights,
                'raw_score': round(raw_score, 3),
                'scaled_score': round(scaled_score, 3)
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Error in enhanced_audit: {e}")
        return {
            'trust_score': 0,
            'confidence_level': 'Error',
            'explanation': f'Error during audit: {str(e)}',
            'best_chunk': '',
            'all_chunks': []
        }

def generate_explanation(
    trust_score: float,
    semantic_score: float,
    lexical_score: float,
    entity_score: float,
    specificity_score: float,
    completeness_score: float,
    entity_analysis: Dict,
    mode: str
) -> str:
    """Generate human-readable explanation of the audit results"""
    
    explanations = []
    
    # CHANGED: More balanced overall assessment
    if trust_score >= 80:  # Lowered thresholds
        explanations.append("✅ **Very High Trust**: The answer is strongly supported by the source content with high confidence.")
    elif trust_score >= 65:
        explanations.append("✅ **High Trust**: The answer appears well-supported by the available content.")
    elif trust_score >= 45:
        explanations.append("⚠️ **Medium Trust**: The answer has reasonable support and appears generally reliable.")
    elif trust_score >= 25:
        explanations.append("⚠️ **Low Trust**: The answer has limited support from the available content.")
    else:
        explanations.append("❌ **Very Low Trust**: The answer appears to have minimal support from the source.")
    
    # CHANGED: More generous score thresholds
    high_scores = []
    low_scores = []
    
    if semantic_score > 0.6:  # Lowered from 0.7
        high_scores.append(f"strong semantic alignment ({semantic_score:.2f})")
    elif semantic_score < 0.35:  # Lowered from 0.4
        low_scores.append(f"weak semantic similarity ({semantic_score:.2f})")
    
    if lexical_score > 0.5:  # Lowered from 0.6
        high_scores.append(f"good lexical overlap ({lexical_score:.1%})")
    elif lexical_score < 0.25:  # Lowered from 0.3
        low_scores.append(f"low lexical overlap ({lexical_score:.1%})")
    
    if specificity_score > 0.6:  # Lowered from 0.7
        high_scores.append("good specificity")
    elif specificity_score < 0.35:  # Lowered from 0.4
        low_scores.append("lacks specificity")
    
    if completeness_score > 0.6:  # Lowered from 0.7
        high_scores.append("comprehensive coverage")
    elif completeness_score < 0.35:  # Lowered from 0.4
        low_scores.append("incomplete answer")
    
    # CHANGED: Give more credit to entity matching
    if entity_score > 0.4:  # New condition
        high_scores.append("strong entity alignment")
    
    if high_scores:
        explanations.append(f"Strengths: {', '.join(high_scores)}.")
    
    if low_scores:
        explanations.append(f"Concerns: {', '.join(low_scores)}.")
    
    # Entity analysis
    if entity_analysis['matches']:
        explanations.append(f"Found {len(entity_analysis['matches'])} matching entities, indicating factual alignment.")
    else:
        explanations.append("No matching entities found, which may indicate divergent content.")
    
    return " ".join(explanations)

def highlight_relevant_parts(chunk: str, answer: str) -> str:
    """Highlight parts of the chunk that are relevant to the answer"""
    try:
        answer_phrases = extract_key_phrases(answer)
        highlighted_chunk = chunk
        
        for phrase in answer_phrases[:5]:  # Limit to top 5 phrases
            if len(phrase) > 3:
                # Case-insensitive replacement with highlighting
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                highlighted_chunk = pattern.sub(f"**{phrase}**", highlighted_chunk)
        
        return highlighted_chunk
    except:
        return chunk

def generate_detailed_report(audit_result: Dict[str, Any]) -> str:
    """Generate a detailed markdown report of the audit results"""
    
    report_lines = [
        "## 📋 Detailed Audit Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "### 📊 Overall Assessment",
        f"- **Trust Score:** {audit_result['trust_score']}%",
        f"- **Confidence Level:** {audit_result['confidence_level']}",
        f"- **Audit Mode:** {audit_result.get('metadata', {}).get('mode', 'standard').title()}",
        ""
    ]
    
    # Score breakdown
    if 'score_breakdown' in audit_result:
        report_lines.extend([
            "### 📈 Score Breakdown",
            ""
        ])
        for metric, score in audit_result['score_breakdown'].items():
            report_lines.append(f"- **{metric}:** {score}%")
        report_lines.append("")
    
    # Technical details
    metadata = audit_result.get('metadata', {})
    if 'raw_score' in metadata:
        report_lines.extend([
            "### 🔧 Calibration Details",
            f"- **Raw Score:** {metadata['raw_score']}",
            f"- **Scaled Score:** {metadata['scaled_score']}",
            f"- **Final Trust Score:** {audit_result['trust_score']}%",
            ""
        ])
    
    # Entity analysis
    if 'entity_analysis' in audit_result:
        entity_data = audit_result['entity_analysis']
        report_lines.extend([
            "### 🏷️ Entity Analysis",
            f"- **Entities in Answer:** {len(entity_data.get('answer_entities', []))}",
            f"- **Matching Entities:** {len(entity_data.get('matches', []))}",
            f"- **Entity Recall:** {entity_data.get('match_ratio', 0):.1%}",
            f"- **Entity Precision:** {entity_data.get('precision', 0):.1%}",
            ""
        ])
        
        if entity_data.get('matches'):
            report_lines.append("**Matched Entities:**")
            for match in entity_data['matches']:
                report_lines.append(f"- {match}")
            report_lines.append("")
    
    # Source content
    report_lines.extend([
        "### 📖 Most Relevant Source Content",
        audit_result.get('best_chunk', 'No content available'),
        "",
        "### 📝 Analysis Summary",
        audit_result.get('explanation', 'No explanation available'),
        ""
    ])
    
    return "\n".join(report_lines)