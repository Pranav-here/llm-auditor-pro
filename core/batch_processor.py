# core/batch_processor.py
import pandas as pd
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_batch_audits(batch_df, embed_model, faiss_index, chunk_texts, ner_model=None):
    """Process multiple Q&A pairs in batch with enhanced error handling"""
    results = []
    
    # Import the audit function here to avoid circular imports
    try:
        from core.auditor import enhanced_audit
    except ImportError as e:
        logger.error(f"Failed to import enhanced_audit: {e}")
        # Fallback to a basic audit function if enhanced_audit is not available
        enhanced_audit = basic_audit_fallback
    
    # Process each row
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Processing audits"):
        try:
            # Validate required columns
            if 'question' not in row or 'answer' not in row:
                raise ValueError("Missing required columns: question and/or answer")
            
            if pd.isna(row['question']) or pd.isna(row['answer']):
                raise ValueError("Question or answer is empty/null")
            
            # Perform audit
            audit_result = enhanced_audit(
                question=str(row['question']),
                model_answer=str(row['answer']),
                embed_model=embed_model,
                faiss_index=faiss_index,
                chunk_texts=chunk_texts,
                ner_model=ner_model,
                mode='standard'
            )
            
            # Validate audit result structure
            if not isinstance(audit_result, dict):
                raise ValueError("Invalid audit result format")
            
            trust_score = audit_result.get('trust_score', 0)
            if not isinstance(trust_score, (int, float)):
                trust_score = 0
            
            # Compile result
            result = {
                'question': str(row['question']),
                'answer': str(row['answer']),
                'model': str(row.get('model', 'Unknown')),
                'trust_score': float(trust_score),
                'confidence_level': audit_result.get('confidence_level', 'Unknown'),
                'status': get_status_from_score(trust_score),
                'semantic_score': audit_result.get('score_breakdown', {}).get('semantic', 0),
                'entity_score': audit_result.get('score_breakdown', {}).get('entity', 0),
                'factual_score': audit_result.get('score_breakdown', {}).get('factual', 0),
                'coverage_score': audit_result.get('score_breakdown', {}).get('coverage', 0),
                'timestamp': datetime.now().isoformat(),
                'error': None
            }
            
            results.append(result)
            logger.info(f"Processed audit {idx + 1}/{len(batch_df)} - Trust Score: {trust_score}%")
            
        except Exception as e:
            # Handle individual failures gracefully
            logger.error(f"Error processing row {idx}: {str(e)}")
            error_result = {
                'question': str(row.get('question', 'N/A')),
                'answer': str(row.get('answer', 'N/A')),
                'model': str(row.get('model', 'Unknown')),
                'trust_score': 0,
                'confidence_level': 'Error',
                'status': '❌ Processing Error',
                'semantic_score': 0,
                'entity_score': 0,
                'factual_score': 0,
                'coverage_score': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(error_result)
    
    return results

def basic_audit_fallback(question, model_answer, embed_model, faiss_index, chunk_texts, ner_model=None, mode='standard'):
    """
    Fallback audit function if enhanced_audit is not available
    """
    try:
        from core.vector_store import hybrid_search
        
        # Simple similarity-based audit
        results = hybrid_search(question, faiss_index, chunk_texts, embed_model, top_k=3)
        
        if results:
            best_chunk = results[0]['text']
            similarity_score = results[0]['score'] * 100  # Convert to percentage
        else:
            best_chunk = "No relevant content found"
            similarity_score = 0
        
        return {
            'trust_score': min(similarity_score, 100),
            'confidence_level': 'Medium' if similarity_score > 50 else 'Low',
            'best_chunk': best_chunk,
            'score_breakdown': {
                'semantic': similarity_score,
                'entity': 0,
                'factual': 0,
                'coverage': 0
            },
            'explanation': f"Basic similarity analysis. Score based on semantic similarity to document content."
        }
    
    except Exception as e:
        logger.error(f"Fallback audit failed: {e}")
        return {
            'trust_score': 0,
            'confidence_level': 'Error',
            'best_chunk': "Error processing",
            'score_breakdown': {'semantic': 0, 'entity': 0, 'factual': 0, 'coverage': 0},
            'explanation': f"Error during audit: {str(e)}"
        }

def get_status_from_score(score):
    """Convert numeric score to status label"""
    try:
        score = float(score)
        if score >= 80:
            return "✅ Trusted"
        elif score >= 50:
            return "⚠️ Caution"
        else:
            return "❌ Suspicious"
    except (ValueError, TypeError):
        return "❌ Error"

def export_results(results_df, format='csv'):
    """Export results in various formats with error handling"""
    try:
        if format == 'csv':
            return results_df.to_csv(index=False)
        
        elif format == 'detailed':
            # Generate detailed markdown report
            total_audits = len(results_df)
            avg_trust = results_df['trust_score'].mean() if total_audits > 0 else 0
            high_trust = (results_df['trust_score'] >= 80).sum()
            medium_trust = ((results_df['trust_score'] >= 50) & (results_df['trust_score'] < 80)).sum()
            low_trust = (results_df['trust_score'] < 50).sum()
            
            report = f"""# AI Knowledge Audit Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics
- **Total Audits**: {total_audits}
- **Average Trust Score**: {avg_trust:.1f}%
- **High Trust (≥80%)**: {high_trust}
- **Medium Trust (50-79%)**: {medium_trust}
- **Low Trust (<50%)**: {low_trust}

## Detailed Results"""
            
            for idx, row in results_df.iterrows():
                status_emoji = "✅" if row['trust_score'] >= 80 else "⚠️" if row['trust_score'] >= 50 else "❌"
                report += f"""

### Audit #{idx + 1} {status_emoji}
**Question**: {row['question']}
**Answer**: {row['answer'][:200]}{'...' if len(str(row['answer'])) > 200 else ''}
**Model**: {row.get('model', 'Unknown')}
**Trust Score**: {row['trust_score']:.1f}%
**Status**: {row['status']}
**Confidence**: {row.get('confidence_level', 'Unknown')}
"""
                
                if row.get('error'):
                    report += f"**Error**: {row['error']}\n"
                
                report += "\n---"
            
            return report
        
        else:
            return results_df.to_json(orient='records', indent=2)
    
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return f"Error generating export: {str(e)}"

def validate_batch_dataframe(df):
    """Validate the structure of the batch dataframe"""
    required_columns = ['question', 'answer']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty rows
    empty_questions = df['question'].isna().sum()
    empty_answers = df['answer'].isna().sum()
    
    if empty_questions > 0 or empty_answers > 0:
        logger.warning(f"Found {empty_questions} empty questions and {empty_answers} empty answers")
    
    return True

# Additional helper functions
def compare_entities(answer_entities, document_entities):
    """Compare entities between answer and document with better error handling"""
    if not answer_entities or not document_entities:
        return []
    
    matches = []
    try:
        doc_texts = [e.get('text', '').lower() for e in document_entities if isinstance(e, dict)]
        
        for ans_ent in answer_entities:
            if not isinstance(ans_ent, dict):
                continue
                
            entity_text = ans_ent.get('text', '').lower()
            entity_label = ans_ent.get('label', 'UNKNOWN')
            
            is_matched = entity_text in doc_texts
            matches.append({
                'entity': ans_ent.get('text', 'N/A'),
                'type': entity_label,
                'matched': is_matched
            })
    
    except Exception as e:
        logger.error(f"Error comparing entities: {e}")
        return []
    
    return matches

def get_batch_statistics(results):
    """Generate statistics from batch results"""
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    stats = {
        'total_processed': len(results),
        'successful_audits': len([r for r in results if r.get('error') is None]),
        'failed_audits': len([r for r in results if r.get('error') is not None]),
        'average_trust_score': df['trust_score'].mean(),
        'median_trust_score': df['trust_score'].median(),
        'high_trust_count': (df['trust_score'] >= 80).sum(),
        'medium_trust_count': ((df['trust_score'] >= 50) & (df['trust_score'] < 80)).sum(),
        'low_trust_count': (df['trust_score'] < 50).sum(),
        'processing_time': datetime.now().isoformat()
    }
    
    return stats