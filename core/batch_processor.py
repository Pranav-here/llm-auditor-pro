# core/batch_processor.py
import pandas as pd
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

def process_batch_audits(batch_df, embed_model, faiss_index, chunk_texts, ner_model=None):
    """Process multiple Q&A pairs in batch"""
    results = []
    
    # Process each row
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Processing audits"):
        try:
            # Perform audit
            audit_result = enhanced_audit(
                question=row['question'],
                model_answer=row['answer'],
                embed_model=embed_model,
                faiss_index=faiss_index,
                chunk_texts=chunk_texts,
                ner_model=ner_model,
                mode='standard'
            )
            
            # Compile result
            result = {
                'question': row['question'],
                'answer': row['answer'],
                'model': row.get('model', 'Unknown'),
                'trust_score': audit_result['trust_score'],
                'confidence_level': audit_result['confidence_level'],
                'status': get_status_from_score(audit_result['trust_score']),
                'semantic_score': audit_result['score_breakdown'].get('semantic', 0),
                'entity_score': audit_result['score_breakdown'].get('entity', 0),
                'factual_score': audit_result['score_breakdown'].get('factual', 0),
                'coverage_score': audit_result['score_breakdown'].get('coverage', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            
        except Exception as e:
            # Handle individual failures gracefully
            error_result = {
                'question': row['question'],
                'answer': row['answer'],
                'model': row.get('model', 'Unknown'),
                'trust_score': 0,
                'confidence_level': 'Error',
                'status': '❌ Processing Error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(error_result)
    
    return results

def get_status_from_score(score):
    """Convert numeric score to status label"""
    if score >= 80:
        return "✅ Trusted"
    elif score >= 50:
        return "⚠️ Caution"
    else:
        return "❌ Suspicious"

def export_results(results_df, format='csv'):
    """Export results in various formats"""
    if format == 'csv':
        return results_df.to_csv(index=False)
    
    elif format == 'detailed':
        # Generate detailed markdown report
        report = f"""# AI Knowledge Audit Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics
- **Total Audits**: {len(results_df)}
- **Average Trust Score**: {results_df['trust_score'].mean():.1f}%
- **High Trust (≥80%)**: {(results_df['trust_score'] >= 80).sum()}
- **Medium Trust (50-79%)**: {((results_df['trust_score'] >= 50) & (results_df['trust_score'] < 80)).sum()}
- **Low Trust (<50%)**: {(results_df['trust_score'] < 50).sum()}

## Detailed Results"""
        
        for idx, row in results_df.iterrows():
            report += f"""
### Audit #{idx + 1}
**Question**: {row['question']}
**Answer**: {row['answer']}
**Model**: {row.get('model', 'Unknown')}
**Trust Score**: {row['trust_score']}%
**Status**: {row['status']}
**Confidence**: {row.get('confidence_level', 'Unknown')}

---"""
        
        return report
    
    else:
        return results_df.to_json(orient='records', indent=2)

# Additional helper functions
def compare_entities(answer_entities, document_entities):
    """Compare entities between answer and document"""
    if not answer_entities or not document_entities:
        return []
    
    matches = []
    doc_texts = [e['text'].lower() for e in document_entities]
    
    for ans_ent in answer_entities:
        if ans_ent['text'].lower() in doc_texts:
            matches.append({
                'entity': ans_ent['text'],
                'type': ans_ent['label'],
                'matched': True
            })
        else:
            matches.append({
                'entity': ans_ent['text'],
                'type': ans_ent['label'],
                'matched': False
            })
    
    return matches