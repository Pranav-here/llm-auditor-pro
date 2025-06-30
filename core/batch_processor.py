# core/batch_processor.py
import pandas as pd
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_batch_audits(
    batch_df: pd.DataFrame, 
    embed_model, 
    faiss_index=None, 
    chunk_texts: List[str] = None, 
    ner_model=None,
    use_web_search: bool = False,
    web_search_enabled: bool = True
) -> List[Dict[str, Any]]:
    """
    Process multiple Q&A pairs in batch with enhanced error handling and web search support
    
    Args:
        batch_df: DataFrame with question and answer columns
        embed_model: Embedding model for similarity calculations
        faiss_index: FAISS index for document search (optional if using web search)
        chunk_texts: Document chunks (optional if using web search)
        ner_model: Named entity recognition model (optional)
        use_web_search: Whether to use web search when no document is available
        web_search_enabled: Global flag for web search availability
    
    Returns:
        List of audit results
    """
    results = []
    
    # Import audit functions
    try:
        from core.auditor import enhanced_audit
        from core.websearch import search_and_prepare_chunks, create_web_search_index
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        enhanced_audit = basic_audit_fallback
        search_and_prepare_chunks = None
        create_web_search_index = None
    
    # Determine processing mode
    has_document = faiss_index is not None and chunk_texts is not None
    will_use_web_search = (not has_document and use_web_search and 
                          web_search_enabled and search_and_prepare_chunks is not None)
    
    if not has_document and not will_use_web_search:
        logger.warning("No document loaded and web search not enabled/available")
    
    logger.info(f"Processing {len(batch_df)} audits - Document: {has_document}, Web Search: {will_use_web_search}")
    
    # Process each row
    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Processing batch audits"):
        try:
            # Validate required columns
            if 'question' not in row or 'answer' not in row:
                raise ValueError("Missing required columns: question and/or answer")
            
            if pd.isna(row['question']) or pd.isna(row['answer']):
                raise ValueError("Question or answer is empty/null")
            
            question = str(row['question']).strip()
            answer = str(row['answer']).strip()
            
            if not question or not answer:
                raise ValueError("Question or answer is empty after cleaning")
            
            # Prepare search context based on available resources
            current_index = faiss_index
            current_chunks = chunk_texts
            search_mode = "document"
            
            # Use web search if no document is available and web search is enabled
            if not has_document and will_use_web_search:
                try:
                    # Get web search results
                    web_snippets = search_and_prepare_chunks(question, answer, k=10)
                    
                    if web_snippets:
                        # Create temporary index from web results
                        current_index, current_chunks = create_web_search_index(web_snippets, embed_model)
                        search_mode = "web_search"
                        logger.info(f"Using web search for audit {idx + 1} - found {len(web_snippets)} snippets")
                    else:
                        logger.warning(f"No web search results for audit {idx + 1}")
                        current_index, current_chunks = None, []
                
                except Exception as web_error:
                    logger.error(f"Web search failed for audit {idx + 1}: {web_error}")
                    current_index, current_chunks = None, []
                    search_mode = "failed"
            
            # Perform audit if we have content to work with
            if current_index is not None and current_chunks:
                audit_result = enhanced_audit(
                    question=question,
                    model_answer=answer,
                    embed_model=embed_model,
                    faiss_index=current_index,
                    chunk_texts=current_chunks,
                    ner_model=ner_model,
                    mode='standard'
                )
            else:
                # No content available for auditing
                audit_result = {
                    'trust_score': 0,
                    'confidence_level': 'Unable to Verify',
                    'explanation': 'No content available for verification (no document loaded and web search unavailable/failed)',
                    'best_chunk': '',
                    'score_breakdown': {
                        'Semantic Similarity': 0,
                        'Lexical Overlap': 0,
                        'Entity Matching': 0,
                        'Retrieval Quality': 0
                    }
                }
            
            # Validate audit result structure
            if not isinstance(audit_result, dict):
                raise ValueError("Invalid audit result format")
            
            trust_score = audit_result.get('trust_score', 0)
            if not isinstance(trust_score, (int, float)):
                trust_score = 0
            
            # Extract score breakdown safely
            score_breakdown = audit_result.get('score_breakdown', {})
            
            # Compile result
            result = {
                'question': question,
                'answer': answer,
                'model': str(row.get('model', 'Unknown')),
                'trust_score': float(trust_score),
                'confidence_level': audit_result.get('confidence_level', 'Unknown'),
                'status': get_status_from_score(trust_score),
                'search_mode': search_mode,
                'semantic_score': score_breakdown.get('Semantic Similarity', 0),
                'lexical_score': score_breakdown.get('Lexical Overlap', 0),
                'entity_score': score_breakdown.get('Entity Matching', 0),
                'retrieval_score': score_breakdown.get('Retrieval Quality', 0),
                'explanation': audit_result.get('explanation', 'No explanation available'),
                'best_chunk': audit_result.get('best_chunk', '')[:500] + ('...' if len(audit_result.get('best_chunk', '')) > 500 else ''),
                'timestamp': datetime.now().isoformat(),
                'error': None
            }
            
            results.append(result)
            logger.info(f"Processed audit {idx + 1}/{len(batch_df)} - Trust Score: {trust_score:.1f}% ({search_mode})")
            
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
                'search_mode': 'error',
                'semantic_score': 0,
                'lexical_score': 0,
                'entity_score': 0,
                'retrieval_score': 0,
                'explanation': f'Processing error: {str(e)}',
                'best_chunk': '',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            results.append(error_result)
    
    # Log final statistics
    successful_audits = len([r for r in results if r.get('error') is None])
    web_search_audits = len([r for r in results if r.get('search_mode') == 'web_search'])
    document_audits = len([r for r in results if r.get('search_mode') == 'document'])
    
    logger.info(f"Batch processing complete: {successful_audits}/{len(results)} successful")
    logger.info(f"Search modes: {document_audits} document, {web_search_audits} web search")
    
    return results

def process_batch_with_threading(
    batch_df: pd.DataFrame, 
    embed_model, 
    faiss_index=None, 
    chunk_texts: List[str] = None, 
    ner_model=None,
    use_web_search: bool = False,
    max_workers: int = 3
) -> List[Dict[str, Any]]:
    """
    Process batch audits with threading for improved performance
    Note: Use with caution as web search APIs may have rate limits
    """
    results = []
    
    def process_single_audit(row_data):
        idx, row = row_data
        try:
            # Process single audit (similar logic to main function but for one row)
            return process_single_row_audit(row, idx, embed_model, faiss_index, chunk_texts, ner_model, use_web_search)
        except Exception as e:
            logger.error(f"Thread error processing row {idx}: {e}")
            return create_error_result(row, str(e))
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_row = {
            executor.submit(process_single_audit, (idx, row)): idx 
            for idx, row in batch_df.iterrows()
        }
        
        # Collect results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_row), 
                          total=len(future_to_row), desc="Processing with threads"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                row_idx = future_to_row[future]
                logger.error(f"Future exception for row {row_idx}: {e}")
                results.append(create_error_result(batch_df.iloc[row_idx], str(e)))
    
    # Sort results by original order
    results.sort(key=lambda x: x.get('original_index', 0))
    
    return results

def process_single_row_audit(row, idx, embed_model, faiss_index, chunk_texts, ner_model, use_web_search):
    """Process a single audit row - helper function for threading"""
    try:
        from core.auditor import enhanced_audit
        from core.websearch import search_and_prepare_chunks, create_web_search_index
    except ImportError:
        from core.auditor import basic_audit_fallback as enhanced_audit
        search_and_prepare_chunks = None
        create_web_search_index = None
    
    question = str(row['question']).strip()
    answer = str(row['answer']).strip()
    
    # Determine search strategy
    has_document = faiss_index is not None and chunk_texts is not None
    current_index, current_chunks = faiss_index, chunk_texts
    search_mode = "document"
    
    if not has_document and use_web_search and search_and_prepare_chunks:
        web_snippets = search_and_prepare_chunks(question, answer, k=8)
        if web_snippets:
            current_index, current_chunks = create_web_search_index(web_snippets, embed_model)
            search_mode = "web_search"
    
    # Perform audit
    if current_index is not None and current_chunks:
        audit_result = enhanced_audit(
            question=question,
            model_answer=answer,
            embed_model=embed_model,
            faiss_index=current_index,
            chunk_texts=current_chunks,
            ner_model=ner_model,
            mode='standard'
        )
    else:
        audit_result = create_no_content_result()
    
    # Format result
    trust_score = float(audit_result.get('trust_score', 0))
    score_breakdown = audit_result.get('score_breakdown', {})
    
    return {
        'original_index': idx,
        'question': question,
        'answer': answer,
        'model': str(row.get('model', 'Unknown')),
        'trust_score': trust_score,
        'confidence_level': audit_result.get('confidence_level', 'Unknown'),
        'status': get_status_from_score(trust_score),
        'search_mode': search_mode,
        'semantic_score': score_breakdown.get('Semantic Similarity', 0),
        'lexical_score': score_breakdown.get('Lexical Overlap', 0),
        'entity_score': score_breakdown.get('Entity Matching', 0),
        'retrieval_score': score_breakdown.get('Retrieval Quality', 0),
        'explanation': audit_result.get('explanation', ''),
        'best_chunk': audit_result.get('best_chunk', '')[:300],
        'timestamp': datetime.now().isoformat(),
        'error': None
    }

def create_error_result(row, error_msg):
    """Create standardized error result"""
    return {
        'question': str(row.get('question', 'N/A')),
        'answer': str(row.get('answer', 'N/A')),
        'model': str(row.get('model', 'Unknown')),
        'trust_score': 0,
        'confidence_level': 'Error',
        'status': '❌ Processing Error',
        'search_mode': 'error',
        'semantic_score': 0,
        'lexical_score': 0,
        'entity_score': 0,
        'retrieval_score': 0,
        'explanation': f'Error: {error_msg}',
        'best_chunk': '',
        'error': error_msg,
        'timestamp': datetime.now().isoformat()
    }

def create_no_content_result():
    """Create result when no content is available for auditing"""
    return {
        'trust_score': 0,
        'confidence_level': 'Unable to Verify',
        'explanation': 'No content available for verification',
        'best_chunk': '',
        'score_breakdown': {
            'Semantic Similarity': 0,
            'Lexical Overlap': 0,
            'Entity Matching': 0,
            'Retrieval Quality': 0
        }
    }

def basic_audit_fallback(question, model_answer, embed_model, faiss_index, chunk_texts, ner_model=None, mode='standard'):
    """
    Fallback audit function if enhanced_audit is not available
    """
    try:
        from core.vector_store import hybrid_search
        
        # Simple similarity-based audit
        if faiss_index is not None and chunk_texts:
            results = hybrid_search(question, model_answer, embed_model, faiss_index, chunk_texts, top_k=3)
            
            if results:
                best_result = results[0]
                best_chunk = best_result.get('chunk', '')
                similarity_score = best_result.get('score', 0) * 100
            else:
                best_chunk = "No relevant content found"
                similarity_score = 0
        else:
            best_chunk = "No content available"
            similarity_score = 0
        
        return {
            'trust_score': min(similarity_score, 100),
            'confidence_level': 'Medium' if similarity_score > 50 else 'Low',
            'best_chunk': best_chunk,
            'score_breakdown': {
                'Semantic Similarity': similarity_score,
                'Lexical Overlap': 0,
                'Entity Matching': 0,
                'Retrieval Quality': similarity_score
            },
            'explanation': f"Basic similarity analysis. Score: {similarity_score:.1f}%"
        }
    
    except Exception as e:
        logger.error(f"Fallback audit failed: {e}")
        return create_no_content_result()

def get_status_from_score(score):
    """Convert numeric score to status label with web search context"""
    try:
        score = float(score)
        if score >= 80:
            return "✅ Trusted"
        elif score >= 60:
            return "⚠️ Likely Accurate"
        elif score >= 40:
            return "⚠️ Needs Verification"
        elif score > 0:
            return "❌ Suspicious"
        else:
            return "❓ Unverifiable"
    except (ValueError, TypeError):
        return "❌ Error"

def export_results(results_df: pd.DataFrame, format: str = 'csv') -> str:
    """Export results in various formats with enhanced web search reporting"""
    try:
        if format == 'csv':
            return results_df.to_csv(index=False)
        
        elif format == 'detailed':
            # Generate detailed markdown report with web search info
            total_audits = len(results_df)
            avg_trust = results_df['trust_score'].mean() if total_audits > 0 else 0
            
            # Status counts
            trusted_count = (results_df['trust_score'] >= 80).sum()
            likely_accurate_count = ((results_df['trust_score'] >= 60) & (results_df['trust_score'] < 80)).sum()
            needs_verification_count = ((results_df['trust_score'] >= 40) & (results_df['trust_score'] < 60)).sum()
            suspicious_count = ((results_df['trust_score'] > 0) & (results_df['trust_score'] < 40)).sum()
            unverifiable_count = (results_df['trust_score'] == 0).sum()
            
            # Search mode counts
            document_searches = (results_df.get('search_mode', '') == 'document').sum()
            web_searches = (results_df.get('search_mode', '') == 'web_search').sum()
            failed_searches = (results_df.get('search_mode', '') == 'failed').sum()
            errors = (results_df.get('search_mode', '') == 'error').sum()
            
            report = f"""# AI Knowledge Audit Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics
- **Total Audits**: {total_audits}
- **Average Trust Score**: {avg_trust:.1f}%

### Trust Level Distribution
- **✅ Trusted (≥80%)**: {trusted_count}
- **⚠️ Likely Accurate (60-79%)**: {likely_accurate_count}
- **⚠️ Needs Verification (40-59%)**: {needs_verification_count}
- **❌ Suspicious (1-39%)**: {suspicious_count}
- **❓ Unverifiable (0%)**: {unverifiable_count}

### Search Method Distribution
- **📄 Document-based**: {document_searches}
- **🌐 Web Search**: {web_searches}
- **⚠️ Search Failed**: {failed_searches}
- **❌ Processing Errors**: {errors}

## Score Breakdown Averages"""
            
            if 'semantic_score' in results_df.columns:
                report += f"""
- **Semantic Similarity**: {results_df['semantic_score'].mean():.1f}%
- **Lexical Overlap**: {results_df['lexical_score'].mean():.1f}%
- **Entity Matching**: {results_df['entity_score'].mean():.1f}%
- **Retrieval Quality**: {results_df['retrieval_score'].mean():.1f}%"""
            
            report += "\n\n## Detailed Results"
            
            for idx, row in results_df.iterrows():
                trust_score = row.get('trust_score', 0)
                search_mode_emoji = {
                    'document': '📄',
                    'web_search': '🌐',
                    'failed': '⚠️',
                    'error': '❌'
                }.get(row.get('search_mode', ''), '❓')
                
                status_emoji = "✅" if trust_score >= 80 else "⚠️" if trust_score >= 40 else "❌" if trust_score > 0 else "❓"
                
                report += f"""

### Audit #{idx + 1} {status_emoji} {search_mode_emoji}
**Question**: {row['question']}
**Answer**: {str(row['answer'])[:200]}{'...' if len(str(row['answer'])) > 200 else ''}
**Model**: {row.get('model', 'Unknown')}
**Trust Score**: {trust_score:.1f}% ({row.get('confidence_level', 'Unknown')})
**Search Mode**: {row.get('search_mode', 'Unknown').replace('_', ' ').title()}"""
                
                if row.get('explanation'):
                    report += f"\n**Analysis**: {row['explanation'][:300]}{'...' if len(str(row['explanation'])) > 300 else ''}"
                
                if row.get('error'):
                    report += f"\n**Error**: {row['error']}"
                
                report += "\n\n---"
            
            return report
        
        elif format == 'summary':
            # Generate executive summary
            stats = get_batch_statistics(results_df.to_dict('records'))
            
            summary = f"""# Executive Summary - AI Audit Results

**Date**: {datetime.now().strftime("%Y-%m-%d")}
**Total Audits**: {stats.get('total_processed', 0)}

## Key Findings
- **Average Trust Score**: {stats.get('average_trust_score', 0):.1f}%
- **High Trust Results**: {stats.get('high_trust_count', 0)} ({stats.get('high_trust_count', 0)/stats.get('total_processed', 1)*100:.1f}%)
- **Results Requiring Verification**: {stats.get('medium_trust_count', 0) + stats.get('low_trust_count', 0)}

## Recommendations
{"✅ Results show high reliability" if stats.get('average_trust_score', 0) >= 70 else "⚠️ Results require additional verification" if stats.get('average_trust_score', 0) >= 50 else "❌ Results show significant reliability concerns"}
"""
            return summary
        
        else:
            return results_df.to_json(orient='records', indent=2)
    
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        return f"Error generating export: {str(e)}"

def validate_batch_dataframe(df: pd.DataFrame) -> bool:
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
    
    # Check data types
    if not df['question'].dtype == 'object':
        logger.warning("Question column should contain text data")
    
    if not df['answer'].dtype == 'object':
        logger.warning("Answer column should contain text data")
    
    return True

def get_batch_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive statistics from batch results"""
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    # Basic statistics
    total_processed = len(results)
    successful_audits = len([r for r in results if r.get('error') is None])
    failed_audits = total_processed - successful_audits
    
    # Trust score statistics
    trust_scores = df['trust_score'].fillna(0)
    avg_trust = trust_scores.mean()
    median_trust = trust_scores.median()
    std_trust = trust_scores.std()
    
    # Trust level counts
    high_trust = (trust_scores >= 80).sum()
    medium_trust = ((trust_scores >= 60) & (trust_scores < 80)).sum()
    low_trust = ((trust_scores >= 40) & (trust_scores < 60)).sum()
    very_low_trust = ((trust_scores > 0) & (trust_scores < 40)).sum()
    unverifiable = (trust_scores == 0).sum()
    
    # Search mode statistics
    search_modes = df.get('search_mode', pd.Series()).value_counts().to_dict()
    
    # Score breakdown averages (if available)
    score_breakdown = {}
    if 'semantic_score' in df.columns:
        score_breakdown = {
            'avg_semantic': df['semantic_score'].fillna(0).mean(),
            'avg_lexical': df['lexical_score'].fillna(0).mean(),
            'avg_entity': df['entity_score'].fillna(0).mean(),
            'avg_retrieval': df['retrieval_score'].fillna(0).mean()
        }
    
    stats = {
        'total_processed': total_processed,
        'successful_audits': successful_audits,
        'failed_audits': failed_audits,
        'success_rate': (successful_audits / total_processed * 100) if total_processed > 0 else 0,
        
        'trust_statistics': {
            'average_trust_score': avg_trust,
            'median_trust_score': median_trust,
            'std_trust_score': std_trust,
            'min_trust_score': trust_scores.min(),
            'max_trust_score': trust_scores.max()
        },
        
        'trust_distribution': {
            'high_trust_count': high_trust,
            'medium_trust_count': medium_trust,
            'low_trust_count': low_trust,
            'very_low_trust_count': very_low_trust,
            'unverifiable_count': unverifiable
        },
        
        'search_modes': search_modes,
        'score_breakdown': score_breakdown,
        'processing_time': datetime.now().isoformat()
    }
    
    return stats

# Utility functions for specific analysis
def analyze_model_performance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze performance by model if model information is available"""
    df = pd.DataFrame(results)
    
    if 'model' not in df.columns:
        return {'error': 'No model information available'}
    
    model_stats = {}
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        model_stats[model] = {
            'count': len(model_data),
            'avg_trust_score': model_data['trust_score'].mean(),
            'high_trust_rate': (model_data['trust_score'] >= 80).sum() / len(model_data) * 100,
            'success_rate': (model_data['error'].isna()).sum() / len(model_data) * 100
        }
    
    return model_stats

def identify_problematic_patterns(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Identify patterns in low-trust or failed audits"""
    df = pd.DataFrame(results)
    
    patterns = {
        'low_trust_questions': [],
        'failed_audits': [],
        'common_issues': {}
    }
    
    # Low trust patterns
    low_trust = df[df['trust_score'] < 40]
    if len(low_trust) > 0:
        patterns['low_trust_questions'] = low_trust[['question', 'trust_score', 'explanation']].to_dict('records')
    
    # Failed audit patterns
    failed = df[df['error'].notna()]
    if len(failed) > 0:
        patterns['failed_audits'] = failed[['question', 'error']].to_dict('records')
        
        # Analyze common error types
        error_types = failed['error'].value_counts().to_dict()
        patterns['common_issues'] = error_types
    
    return patterns