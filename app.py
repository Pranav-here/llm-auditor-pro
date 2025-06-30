# Enhanced AI Knowledge Auditor v5.0
# app.py
import numpy as np
import torch
# --- Monkey-patch torch.Tensor.numpy to catch the RuntimeError ---
orig_tensor_numpy = torch.Tensor.numpy
def safe_tensor_numpy(self, *args, **kwargs):
    try:
        return orig_tensor_numpy(self, *args, **kwargs)
    except RuntimeError:
        # fallback: convert to Python list
        return self.tolist()
torch.Tensor.numpy = safe_tensor_numpy

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import your custom modules - make sure all imports are correct
try:
    from core.loader import extract_text_from_pdf, smart_chunk_text, extract_entities
    from core.embedder import load_embedder, load_summarizer, load_ner_model
    from core.auditor import enhanced_audit, generate_detailed_report
    from core.vector_store import build_and_save_index, load_index_and_chunks, hybrid_search
    from core.batch_processor import process_batch_audits, export_results
    from core.websearch import search_and_prepare_chunks, create_web_search_index
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Please make sure all required modules are in the 'core' directory")
    st.stop()

# Configure page
st.set_page_config(
    page_title="AI Knowledge Auditor Pro", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .trust-high { color: #28a745; font-weight: bold; }
    .trust-medium { color: #ffc107; font-weight: bold; }
    .trust-low { color: #dc3545; font-weight: bold; }
    .highlighted-text { background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    .web-search-info { 
        background-color: #e3f2fd; 
        border-left: 4px solid #2196f3; 
        padding: 1rem; 
        margin: 1rem 0; 
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.status-strip {
    background: var(--secondary-background-color, #333);
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
}
.status-strip span {
    color: var(--text-color, #FFF);
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        "embed_model": None,
        "summarizer": None,
        "ner_model": None,
        "messages": [],
        "faiss_index": None,
        "chunk_texts": None,
        "audit_history": [],
        "current_document": None,
        "batch_results": None,
        "document_entities": [],
        "web_search_enabled": True,
        "using_web_search": False,
        "web_search_results": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Load models with caching
@st.cache_resource
def load_models():
    try:
        embed_model = load_embedder()
        summarizer = load_summarizer()
        ner_model = load_ner_model()
        return embed_model, summarizer, ner_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Check if web search is available
def check_web_search_available():
    try:
        import os
        return bool(os.getenv("TAVILY_API_KEY"))
    except:
        return False

# Sidebar
with st.sidebar:
    st.title("🧠 AI Auditor Pro")
    st.markdown("### 📚 Enhanced Features")
    st.markdown("""
    ✅ **Smart Document Processing**
    - Semantic chunking
    - Named entity recognition
    - Multi-factor scoring
    
    ✅ **Web Search Fallback**
    - No PDF required
    - Real-time web content
    - Same audit pipeline
    
    ✅ **Advanced Analytics**
    - Confidence visualization
    - Entity matching analysis
    - Detailed reporting
    
    ✅ **Batch Processing**
    - Multiple Q&A pairs
    - Export capabilities
    - Comparative analysis
    """)
    
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider("Trust Score Threshold", 0, 100, 70)
    show_advanced = st.checkbox("Show Advanced Metrics", True)
    enable_entity_matching = st.checkbox("Enable Entity Matching", True)
    
    # Web search settings
    if check_web_search_available():
        st.markdown("### 🌐 Web Search")
        st.session_state.web_search_enabled = st.checkbox("Enable Web Search", True)
        if st.session_state.web_search_enabled:
            st.info("Web search available when no PDF is loaded")
    else:
        st.warning("Web search unavailable: TAVILY_API_KEY not set")
        st.session_state.web_search_enabled = False

# Main interface with tabs
st.title("🧠 AI Knowledge Auditor Pro")
st.caption("Advanced AI answer verification with multi-factor analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📄 Single Audit", "📊 Batch Processing", "📈 Analytics", "⚙️ Settings"])

if not st.session_state.embed_model:
    with st.spinner("Loading AI models..."):
        models = load_models()
        if models[0] is not None:
            st.session_state.embed_model, st.session_state.summarizer, st.session_state.ner_model = models
        else:
            st.error("Failed to load models. Please check your setup.")

# Initialize audit inputs persistence
if "audit_inputs" not in st.session_state:
    st.session_state.audit_inputs = {
        "question": "", 
        "model_answer": "", 
        "model_name": "GPT-4", 
        "audit_mode": "Standard"
    }

with tab1:
    if not st.session_state.embed_model:
        st.error("Models not loaded. Please check your setup and refresh the page.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload
            uploaded_pdf = st.file_uploader("📄 Upload PDF Document (Optional)", type=["pdf"])
            
            if uploaded_pdf and uploaded_pdf.size > 200 * 1024 * 1024:
                st.error("🚫 File too large. Please upload a PDF under 200 MB.")
                uploaded_pdf = None
            
            # Process PDF
            if uploaded_pdf and uploaded_pdf != st.session_state.current_document:
                with st.spinner("Processing document..."):
                    try:
                        text = extract_text_from_pdf(uploaded_pdf)
                        if text:
                            chunks = smart_chunk_text(text)
                            entities = extract_entities(text, st.session_state.ner_model)
                            
                            build_and_save_index(chunks, st.session_state.embed_model)
                            st.session_state.faiss_index, st.session_state.chunk_texts = load_index_and_chunks()
                            st.session_state.current_document = uploaded_pdf
                            st.session_state.document_entities = entities
                            st.session_state.using_web_search = False
                            
                            st.success(f"✅ Document processed! Found {len(chunks)} chunks and {len(entities)} entities.")
                        else:
                            st.error("Failed to extract text from PDF. Please try a different file.")
                    except Exception as e:
                        st.error(f"Error processing document: {e}")
            
            # Web search option when no PDF
            if not st.session_state.faiss_index and st.session_state.web_search_enabled:
                st.markdown('<div class="web-search-info">', unsafe_allow_html=True)
                st.markdown("### 🌐 Web Search Mode")
                st.info("No PDF loaded. Auditing will use real-time web search to find relevant content.")
                st.session_state.using_web_search = True
                st.markdown('</div>', unsafe_allow_html=True)
            elif not st.session_state.faiss_index and not st.session_state.web_search_enabled:
                st.warning("Please upload a PDF document or enable web search to proceed.")
        
        with col2:
            if st.session_state.faiss_index:
                st.metric("Document Status", "✅ PDF Ready", "Indexed")
                st.metric("Chunks", len(st.session_state.chunk_texts) if st.session_state.chunk_texts else 0)
                if hasattr(st.session_state, 'document_entities'):
                    st.metric("Entities Found", len(st.session_state.document_entities))
                st.session_state.using_web_search = False
            elif st.session_state.using_web_search:
                st.metric("Search Status", "🌐 Web Ready", "Online")
                st.metric("Mode", "Web Search", "Real-time")
                if st.session_state.web_search_results:
                    st.metric("Sources Found", len(st.session_state.web_search_results))
            else:
                st.metric("Status", "⏳ Waiting", "No source")
        
        # Audit interface
        if st.session_state.faiss_index or st.session_state.using_web_search:
            st.markdown("### 🔍 Audit AI Answer")
            
            with st.form("enhanced_audit_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    question = st.text_area("❓ Your Question", 
                                          value=st.session_state.audit_inputs["question"],
                                          height=100)
                    model_name = st.selectbox("🤖 AI Model Used", 
                        ["GPT-4", "GPT-3.5", "Claude", "Gemini", "Other"], 
                        index=["GPT-4", "GPT-3.5", "Claude", "Gemini", "Other"].index(st.session_state.audit_inputs["model_name"]))
                
                with col2:
                    model_answer = st.text_area("🧠 Model's Answer", 
                                              value=st.session_state.audit_inputs["model_answer"],
                                              height=100)
                    audit_mode = st.selectbox("🎯 Audit Mode", 
                        ["Standard", "Strict", "Lenient"], 
                        index=["Standard", "Strict", "Lenient"].index(st.session_state.audit_inputs["audit_mode"]))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_summary = st.checkbox("📝 Generate Summary", True)
                with col2:
                    show_entities = st.checkbox("🏷️ Show Entity Analysis", enable_entity_matching)
                with col3:
                    show_confidence = st.checkbox("📊 Show Confidence Breakdown", show_advanced)
                
                submitted = st.form_submit_button("🔍 Audit Answer", use_container_width=True)
            
            if submitted and question and model_answer:
                # Update session state with current inputs (but don't clear them)
                st.session_state.audit_inputs.update({
                    "question": question,
                    "model_answer": model_answer,
                    "model_name": model_name,
                    "audit_mode": audit_mode
                })
                
                with st.spinner("Analyzing answer..."):
                    try:
                        # Determine if we need to use web search
                        if st.session_state.using_web_search:
                            # Use web search to get relevant content
                            with st.spinner("Searching web for relevant content..."):
                                web_snippets = search_and_prepare_chunks(question, model_answer, k=10)
                                if web_snippets:
                                    # Create temporary index from web search results
                                    temp_index, temp_chunks = create_web_search_index(web_snippets, st.session_state.embed_model)
                                    st.session_state.web_search_results = web_snippets
                                    
                                    if temp_index and temp_chunks:
                                        # Use web search results for audit
                                        faiss_index = temp_index
                                        chunk_texts = temp_chunks
                                        document_entities = []  # No entities from web search
                                        
                                        st.info(f"🌐 Found {len(web_snippets)} relevant sources from web search")
                                    else:
                                        st.error("Failed to process web search results")
                                        st.stop()
                                else:
                                    st.error("No relevant content found in web search")
                                    st.stop()
                        else:
                            # Use existing PDF-based index
                            faiss_index = st.session_state.faiss_index
                            chunk_texts = st.session_state.chunk_texts
                            document_entities = getattr(st.session_state, 'document_entities', [])
                        
                        # Enhanced audit
                        audit_result = enhanced_audit(
                            question=question,
                            model_answer=model_answer,
                            embed_model=st.session_state.embed_model,
                            faiss_index=faiss_index,
                            chunk_texts=chunk_texts,
                            ner_model=st.session_state.ner_model if enable_entity_matching else None,
                            document_entities=document_entities,
                            mode=audit_mode.lower()
                        )
                        
                        # Display results
                        trust_score = audit_result['trust_score']
                        trust_class = "trust-high" if trust_score >= 80 else "trust-medium" if trust_score >= 50 else "trust-low"
                        
                        st.markdown("### 📊 Audit Results")
                        
                        # Show source type
                        if st.session_state.using_web_search:
                            st.markdown('<div class="web-search-info">🌐 <strong>Source:</strong> Web Search Results</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div style="background-color: #e8f5e8; border-left: 4px solid #4caf50; padding: 1rem; margin: 1rem 0; border-radius: 4px;">📄 <strong>Source:</strong> Uploaded PDF Document</div>', unsafe_allow_html=True)
                        
                        # Trust score with visualization
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f'<div class="metric-card"><h3>Trust Score</h3><h1 class="{trust_class}">{trust_score}%</h1></div>', unsafe_allow_html=True)
                        with col2:
                            status = "✅ Trusted" if trust_score >= 80 else "⚠️ Caution" if trust_score >= 50 else "❌ Suspicious"
                            st.markdown(f'<div class="metric-card"><h3>Status</h3><h2>{status}</h2></div>', unsafe_allow_html=True)
                        with col3:
                            confidence = audit_result.get('confidence_level', 'Medium')
                            st.markdown(f'<div class="metric-card"><h3>Confidence</h3><h2>{confidence}</h2></div>', unsafe_allow_html=True)
                        
                        # Confidence breakdown chart
                        if show_confidence and 'score_breakdown' in audit_result:
                            breakdown = audit_result['score_breakdown']
                            fig = go.Figure(data=[
                                go.Bar(x=list(breakdown.keys()), y=list(breakdown.values()),
                                       marker_color=['#4caf50', '#17a2b8', '#ffc107', '#dc3545', '#6f42c1', '#ff7f0e'])
                            ])
                            fig.update_layout(title="Trust Score Breakdown", yaxis_title="Score")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Relevant content
                        st.markdown("### 📖 Most Relevant Content")
                        highlighted_chunk = audit_result.get('highlighted_chunk', audit_result['best_chunk'])
                        st.markdown(highlighted_chunk)
                        
                        # Show multiple sources for web search
                        if st.session_state.using_web_search and 'all_chunks' in audit_result:
                            with st.expander("🌐 View All Web Sources"):
                                for i, chunk_data in enumerate(audit_result['all_chunks'][:5]):
                                    st.markdown(f"**Source {i+1}** (Score: {chunk_data['score']:.2f})")
                                    st.markdown(chunk_data['chunk'])
                                    st.markdown("---")
                        
                        # Entity analysis
                        if show_entities and 'entity_analysis' in audit_result:
                            st.markdown("### 🏷️ Entity Analysis")
                            entity_data = audit_result['entity_analysis']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Entities in Answer:**")
                                for entity in entity_data.get('answer_entities', []):
                                    st.markdown(f"- {entity.get('text', 'N/A')} ({entity.get('label', 'N/A')})")
                            
                            with col2:
                                st.markdown("**Matching Document Entities:**")
                                for match in entity_data.get('matches', []):
                                    st.markdown(f"- ✅ {match}")
                        
                        # Explanation
                        if audit_result.get('explanation'):
                            st.markdown("### 📝 Analysis")
                            st.info(audit_result['explanation'])
                        
                        # Detailed report
                        if show_advanced:
                            with st.expander("📋 Detailed Analysis Report"):
                                report = generate_detailed_report(audit_result)
                                st.markdown(report)
                        
                        # Save to history
                        audit_record = {
                            'timestamp': datetime.now().isoformat(),
                            'question': question,
                            'answer': model_answer,
                            'model': model_name,
                            'trust_score': trust_score,
                            'status': status,
                            'source_type': 'web_search' if st.session_state.using_web_search else 'pdf',
                            'audit_result': audit_result
                        }
                        st.session_state.audit_history.append(audit_record)
                        
                        # Show success message and option to start new audit
                        st.success("✅ Audit completed! You can now enter new inputs above for another audit.")
                        
                    except Exception as e:
                        st.error(f"Error during audit: {e}")
                        st.error("Please check your inputs and try again.")

with tab2:
    st.markdown("### 📊 Batch Processing")
    st.markdown("Upload multiple Q&A pairs for bulk auditing")
    
    # Check if we have any source (PDF or web search)
    has_source = st.session_state.faiss_index or st.session_state.web_search_enabled
    
    if not has_source:
        st.warning("Please upload a PDF document or enable web search to use batch processing.")
    else:
        # Sample template
        if st.button("📥 Download Template"):
            template_df = pd.DataFrame({
                'question': ['What is the main topic?', 'Who is the author?'],
                'answer': ['Sample answer 1', 'Sample answer 2'],
                'model': ['GPT-4', 'Claude']
            })
            csv = template_df.to_csv(index=False)
            st.download_button("Download CSV Template", csv, "audit_template.csv", "text/csv")
        
        # Batch upload
        batch_file = st.file_uploader("📤 Upload Q&A Batch (CSV)", type=['csv'])
        
        if batch_file:
            try:
                batch_df = pd.read_csv(batch_file)
                st.dataframe(batch_df.head())
                
                if st.button("🚀 Process Batch"):
                    with st.spinner("Processing batch audits..."):
                        try:
                            # Enhanced batch processing with web search support
                            results = []
                            
                            for idx, row in batch_df.iterrows():
                                question = row.get('question', '')
                                answer = row.get('answer', '')
                                model = row.get('model', 'Unknown')
                                
                                if not question or not answer:
                                    continue
                                
                                # Determine source for this audit
                                if st.session_state.using_web_search:
                                    # Use web search for each question
                                    web_snippets = search_and_prepare_chunks(question, answer, k=10)
                                    if web_snippets:
                                        temp_index, temp_chunks = create_web_search_index(web_snippets, st.session_state.embed_model)
                                        if temp_index and temp_chunks:
                                            faiss_index = temp_index
                                            chunk_texts = temp_chunks
                                            document_entities = []
                                        else:
                                            continue
                                    else:
                                        continue
                                else:
                                    # Use PDF-based index
                                    faiss_index = st.session_state.faiss_index
                                    chunk_texts = st.session_state.chunk_texts
                                    document_entities = getattr(st.session_state, 'document_entities', [])
                                
                                # Perform audit
                                audit_result = enhanced_audit(
                                    question=question,
                                    model_answer=answer,
                                    embed_model=st.session_state.embed_model,
                                    faiss_index=faiss_index,
                                    chunk_texts=chunk_texts,
                                    ner_model=st.session_state.ner_model if enable_entity_matching else None,
                                    document_entities=document_entities,
                                    mode='standard'
                                )
                                
                                trust_score = audit_result['trust_score']
                                status = "✅ Trusted" if trust_score >= 80 else "⚠️ Caution" if trust_score >= 50 else "❌ Suspicious"
                                
                                results.append({
                                    'question': question,
                                    'answer': answer,
                                    'model': model,
                                    'trust_score': trust_score,
                                    'status': status,
                                    'source_type': 'web_search' if st.session_state.using_web_search else 'pdf',
                                    'best_chunk': audit_result.get('best_chunk', ''),
                                    'explanation': audit_result.get('explanation', '')
                                })
                            
                            st.session_state.batch_results = results
                            st.success(f"✅ Processed {len(results)} audits!")
                            
                        except Exception as e:
                            st.error(f"Error processing batch: {e}")
                
                # Display batch results
                if st.session_state.batch_results:
                    st.markdown("### 📈 Batch Results")
                    results_df = pd.DataFrame(st.session_state.batch_results)
                    
                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_score = results_df['trust_score'].mean()
                        st.metric("Average Trust Score", f"{avg_score:.1f}%")
                    with col2:
                        high_trust = (results_df['trust_score'] >= 80).sum()
                        st.metric("High Trust Answers", high_trust)
                    with col3:
                        low_trust = (results_df['trust_score'] < 50).sum()
                        st.metric("Suspicious Answers", low_trust)
                    with col4:
                        total_processed = len(results_df)
                        st.metric("Total Processed", total_processed)
                    
                    # Results table
                    display_df = results_df[['question', 'answer', 'model', 'trust_score', 'status']].copy()
                    if 'source_type' in results_df.columns:
                        display_df['source_type'] = results_df['source_type']
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 Export to CSV"):
                            csv = results_df.to_csv(index=False)
                            st.download_button("Download Results", csv, "audit_results.csv", "text/csv")
                    
                    with col2:
                        if st.button("📋 Export Detailed Report"):
                            try:
                                # Generate detailed report
                                report_lines = [
                                    "# Batch Audit Report",
                                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                    f"Total Audits: {len(results_df)}",
                                    f"Average Trust Score: {results_df['trust_score'].mean():.1f}%",
                                    "",
                                    "## Summary Statistics",
                                    f"- High Trust (≥80%): {(results_df['trust_score'] >= 80).sum()}",
                                    f"- Medium Trust (50-79%): {((results_df['trust_score'] >= 50) & (results_df['trust_score'] < 80)).sum()}",
                                    f"- Low Trust (<50%): {(results_df['trust_score'] < 50).sum()}",
                                    "",
                                    "## Detailed Results"
                                ]
                                
                                for idx, row in results_df.iterrows():
                                    report_lines.extend([
                                        f"### Audit {idx + 1}",
                                        f"**Question:** {row['question']}",
                                        f"**Answer:** {row['answer']}",
                                        f"**Model:** {row['model']}",
                                        f"**Trust Score:** {row['trust_score']}%",
                                        f"**Status:** {row['status']}",
                                        f"**Source:** {row.get('source_type', 'pdf')}",
                                        f"**Explanation:** {row.get('explanation', 'N/A')}",
                                        ""
                                    ])
                                
                                report = "\n".join(report_lines)
                                st.download_button("Download Report", report, "detailed_report.md", "text/markdown")
                            except Exception as e:
                                st.error(f"Error generating report: {e}")
            
            except Exception as e:
                st.error(f"Error processing batch file: {str(e)}")

with tab3:
    st.markdown("### 📈 Analytics Dashboard")
    
    if st.session_state.audit_history:
        history_df = pd.DataFrame(st.session_state.audit_history)
        
        # Time series of trust scores
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        fig = px.line(history_df, x='timestamp', y='trust_score', 
                     title='Trust Score Over Time', 
                     color='model')
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by model and source type
        col1, col2 = st.columns(2)
        with col1:
            model_stats = history_df.groupby('model')['trust_score'].agg(['mean', 'count']).reset_index()
            fig = px.bar(model_stats, x='model', y='mean', 
                        title='Average Trust Score by Model')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'source_type' in history_df.columns:
                source_stats = history_df.groupby('source_type')['trust_score'].agg(['mean', 'count']).reset_index()
                fig = px.bar(source_stats, x='source_type', y='mean',
                            title='Average Trust Score by Source Type')
                st.plotly_chart(fig, use_container_width=True)
            else:
                status_counts = history_df['status'].value_counts()
                fig = px.pie(values=status_counts.values, names=status_counts.index,
                            title='Answer Status Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        # Trust score distribution
        fig = px.histogram(history_df, x='trust_score', nbins=20,
                          title='Trust Score Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent audits table
        st.markdown("### 📋 Recent Audits")
        display_cols = ['timestamp', 'question', 'model', 'trust_score', 'status']
        if 'source_type' in history_df.columns:
            display_cols.append('source_type')
        recent_df = history_df.tail(10)[display_cols].copy()
        recent_df['timestamp'] = recent_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(recent_df, use_container_width=True)
        
        # Export analytics
        if st.button("📊 Export Analytics Data"):
            csv = history_df.to_csv(index=False)
            st.download_button("Download Analytics CSV", csv, "audit_analytics.csv", "text/csv")
    
    else:
        st.info("No audit history available. Complete some audits to see analytics!")

with tab4:
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Settings")
        st.info("Current embedding model: all-MiniLM-L6-v2")
        st.info("Current summarizer: distilbart-cnn-12-6")
        
        if st.button("🔄 Reload Models"):
            st.session_state.embed_model = None
            st.session_state.summarizer = None
            st.session_state.ner_model = None
            st.rerun()
    
    with col2:
        st.markdown("#### Data Management")
        
        if st.button("🗑️ Clear Audit History"):
            st.session_state.audit_history = []
            st.success("Audit history cleared!")
        
        if st.button("💾 Export All Data"):
            if st.session_state.audit_history:
                all_data = {
                    'audit_history': st.session_state.audit_history,
                    'export_timestamp': datetime.now().isoformat()
                }
                json_data = json.dumps(all_data, indent=2)
                st.download_button("Download All Data", json_data, "auditor_data.json", "application/json")
    
    # Web Search Configuration
    st.markdown("#### 🌐 Web Search Settings")
    col1, col2 = st.columns(2)
    
    # Your existing code continues from here...
    # Web Search Configuration section in tab4

    with col1:
        api_key_status = "✅ Available" if check_web_search_available() else "❌ Not Set"
        st.metric("API Key Status", api_key_status)
        
        if check_web_search_available():
            st.success("Tavily API key is configured")
        else:
            st.error("TAVILY_API_KEY environment variable not set")
            st.code("export TAVILY_API_KEY=your_key_here")
    
    with col2:
        web_search_results_count = st.slider("Web Search Results", 5, 20, 10)
        st.info(f"Will fetch {web_search_results_count} snippets per search")
        
        if st.button("🧪 Test Web Search"):
            if check_web_search_available():
                try:
                    from core.websearch import tavily_search
                    test_results = tavily_search("artificial intelligence", k=3)
                    if test_results:
                        st.success(f"✅ Web search working! Found {len(test_results)} results")
                        with st.expander("Sample Results"):
                            for i, result in enumerate(test_results[:2]):
                                st.markdown(f"**Result {i+1}:** {result[:200]}...")
                    else:
                        st.warning("Search returned no results")
                except Exception as e:
                    st.error(f"Web search test failed: {e}")
            else:
                st.error("Cannot test: API key not configured")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        st.markdown("#### Chunking Parameters")
        st.session_state.chunk_size = st.slider("Chunk Size", 400, 1500, st.session_state.get('chunk_size', 800))
        st.session_state.overlap_size = st.slider("Chunk Overlap", 50, 400, st.session_state.get('overlap_size', 200))
        st.session_state.max_chunks = st.slider("Max Chunks to Analyze", 3, 20, 
                                            st.session_state.get('max_chunks', 5))
        
        st.markdown("#### Scoring Weights")
        semantic_weight = st.slider("Semantic Similarity Weight", 0.0, 1.0, 0.4)
        entity_weight = st.slider("Entity Matching Weight", 0.0, 1.0, 0.3)
        factual_weight = st.slider("Factual Consistency Weight", 0.0, 1.0, 0.3)
        
        st.markdown("#### Web Search Settings")
        search_timeout = st.slider("Search Timeout (seconds)", 5, 30, 15)
        min_snippet_length = st.slider("Minimum Snippet Length", 20, 100, 30)

# Footer
st.markdown("---")
st.markdown("### 🔗 Quick Actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🎯 New Audit"):
        st.session_state.messages = []
        st.session_state.web_search_results = None

with col2:
    if st.button("📊 View Stats"):
        if st.session_state.audit_history:
            avg_score = np.mean([a['trust_score'] for a in st.session_state.audit_history])
            total_audits = len(st.session_state.audit_history)
            web_audits = len([a for a in st.session_state.audit_history if a.get('source_type') == 'web_search'])
            pdf_audits = total_audits - web_audits
            
            st.success(f"📈 Average Trust Score: {avg_score:.1f}%")
            st.info(f"📄 PDF Audits: {pdf_audits} | 🌐 Web Audits: {web_audits}")

with col3:
    if st.button("💡 Tips"):
        st.info("""
        💡 **Pro Tips:**
        - Use specific questions for better web search results
        - Compare PDF vs web search audits for consistency
        - Web search works best with factual questions
        - Enable entity matching for deeper analysis
        """)

with col4:
    if st.button("❓ Help"):
        st.markdown("""
        ### 🚀 Quick Guide:
        
        **PDF Mode:**
        1. Upload your PDF document
        2. Ask questions about the content
        3. Get trust scores based on document content
        
        **Web Search Mode:**
        1. Enable web search in sidebar
        2. Ask any question (no PDF needed)
        3. System searches web for relevant content
        4. Same audit pipeline with real-time data
        
        **Batch Processing:**
        - Upload CSV with question/answer pairs
        - Works with both PDF and web search modes
        - Export results for further analysis
        
        **Analytics:**
        - Track trust scores over time
        - Compare different AI models
        - Analyze PDF vs web search performance
        """)

# Status bar at bottom
st.markdown("---")
st.markdown("### 🧱 Project Versions")
st.markdown("""
 🔹 [**Initial MVP (v1)** – Live Demo](https://ai-knowledge-auditor.streamlit.app/)
 🔹 [**Initial Codebase (v1)** – GitHub](https://github.com/Pranav-here/ai-knowledge-auditor)
 🔸 [**Current Pro Version (v5.0)** – GitHub](https://github.com/Pranav-here/llm-auditor-pro)
""")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    model_status = "✅ Ready" if st.session_state.embed_model else "❌ Loading"
    st.caption(f"🤖 Models: {model_status}")

with status_col2:
    if st.session_state.faiss_index:
        st.caption("📄 Source: PDF Document")
    elif st.session_state.using_web_search:
        st.caption("🌐 Source: Web Search")
    else:
        st.caption("⏳ Source: None")

with status_col3:
    audit_count = len(st.session_state.audit_history)
    st.caption(f"📊 Audits: {audit_count}")

with status_col4:
    web_search_status = "✅ Available" if check_web_search_available() else "❌ Unavailable"
    st.caption(f"🌐 Web Search: {web_search_status}")

# Debug info (only show in development)
if st.sidebar.checkbox("🔧 Debug Info", False):
    st.sidebar.markdown("### Debug Information")
    
    # Calculate additional stats
    total_web_audits = len([a for a in st.session_state.audit_history if a.get('source_type') == 'web_search'])
    total_pdf_audits = len(st.session_state.audit_history) - total_web_audits
    
    avg_trust_score = None
    if st.session_state.audit_history:
        avg_trust_score = np.mean([a['trust_score'] for a in st.session_state.audit_history])
    
    # Get current document info
    current_doc_size = None
    if st.session_state.current_document:
        current_doc_size = f"{st.session_state.current_document.size / (1024*1024):.2f} MB"
    
    # Get entity count
    entity_count = len(st.session_state.document_entities) if hasattr(st.session_state, 'document_entities') else 0
    
    debug_info = {
        # Core System Status
        "system_status": {
            "models_loaded": st.session_state.embed_model is not None,
            "embed_model_ready": st.session_state.embed_model is not None,
            "summarizer_ready": st.session_state.summarizer is not None,
            "ner_model_ready": st.session_state.ner_model is not None,
        },
        
        # Document Processing
        "document_processing": {
            "faiss_index_loaded": st.session_state.faiss_index is not None,
            "chunk_count": len(st.session_state.chunk_texts) if st.session_state.chunk_texts else 0,
            "current_chunk_size": st.session_state.get('chunk_size', 800),
            "current_overlap_size": st.session_state.get('overlap_size', 200),
            "max_chunks_setting": st.session_state.get('max_chunks', 5),
            "document_entities_found": entity_count,
            "current_document_name": st.session_state.current_document.name if st.session_state.current_document else None,
            "current_document_size": current_doc_size,
        },
        
        # Web Search Status
        "web_search": {
            "web_search_enabled": st.session_state.web_search_enabled,
            "using_web_search": st.session_state.using_web_search,
            "tavily_api_available": check_web_search_available(),
            "web_search_results_cached": st.session_state.web_search_results is not None,
            "web_result_count": len(st.session_state.web_search_results) if st.session_state.web_search_results else 0,
        },
        
        # Audit Statistics
        "audit_statistics": {
            "total_audits": len(st.session_state.audit_history),
            "pdf_audits": total_pdf_audits,
            "web_audits": total_web_audits,
            "average_trust_score": f"{avg_trust_score:.1f}%" if avg_trust_score else "N/A",
            "batch_results_available": st.session_state.batch_results is not None,
            "batch_result_count": len(st.session_state.batch_results) if st.session_state.batch_results else 0,
        },
        
        # Current Settings
        "current_settings": {
            "confidence_threshold": confidence_threshold,
            "show_advanced_metrics": show_advanced,
            "entity_matching_enabled": enable_entity_matching,
            "audit_mode_default": "Standard",
        },
        
        # Session State Keys (useful for debugging)
        "session_state_keys": list(st.session_state.keys()),
        
        # Memory Usage Indicators
        "memory_indicators": {
            "messages_count": len(st.session_state.messages),
            "audit_inputs_stored": bool(st.session_state.get('audit_inputs')),
            "current_inputs_question_length": len(st.session_state.audit_inputs.get('question', '')) if st.session_state.get('audit_inputs') else 0,
            "current_inputs_answer_length": len(st.session_state.audit_inputs.get('answer', '')) if st.session_state.get('audit_inputs') else 0,
        }
    }
    
    # Display in expandable sections for better readability
    with st.sidebar.expander("🖥️ System Status"):
        st.json(debug_info["system_status"])
    
    with st.sidebar.expander("📄 Document Processing"):
        st.json(debug_info["document_processing"])
    
    with st.sidebar.expander("🌐 Web Search"):
        st.json(debug_info["web_search"])
    
    with st.sidebar.expander("📊 Audit Statistics"):
        st.json(debug_info["audit_statistics"])
    
    with st.sidebar.expander("⚙️ Current Settings"):
        st.json(debug_info["current_settings"])
    
    with st.sidebar.expander("💾 Memory & Session"):
        st.json(debug_info["memory_indicators"])
        st.caption(f"Total session keys: {len(debug_info['session_state_keys'])}")
        
    # Quick actions for debugging
    st.sidebar.markdown("#### Debug Actions")
    if st.sidebar.button("🔄 Refresh Debug Info"):
        st.rerun()
        
    if st.sidebar.button("📋 Copy Debug Info"):
        debug_json = json.dumps(debug_info, indent=2)
        st.sidebar.code(debug_json, language="json")
        
    # Performance indicators
    if st.session_state.audit_history:
        recent_audits = st.session_state.audit_history[-5:]
        trust_scores = [a['trust_score'] for a in recent_audits]
        st.sidebar.metric("Recent Avg Score", f"{np.mean(trust_scores):.1f}%", 
                         delta=f"{trust_scores[-1] - np.mean(trust_scores[:-1]):.1f}%" if len(trust_scores) > 1 else None)