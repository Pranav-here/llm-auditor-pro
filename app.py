# Enhanced AI Knowledge Auditor v5.0
# app.py
import numpy as np
import torch

# --- Monkey-patch torch.Tensor.numpy to catch the RuntimeError ---
_orig_tensor_numpy = torch.Tensor.numpy
def _safe_tensor_numpy(self, *args, **kwargs):
    try:
        return _orig_tensor_numpy(self, *args, **kwargs)
    except RuntimeError:
        # fallback: convert to Python list
        return self.tolist()
torch.Tensor.numpy = _safe_tensor_numpy

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
        "document_entities": []
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

# Sidebar
with st.sidebar:
    st.title("🧠 AI Auditor Pro")
    st.markdown("### 📚 Enhanced Features")
    st.markdown("""
    ✅ **Smart Document Processing**
    - Semantic chunking
    - Named entity recognition
    - Multi-factor scoring
    
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

# Main interface with tabs
st.title("🧠 AI Knowledge Auditor Pro")
st.caption("Advanced AI answer verification with multi-factor analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📄 Single Audit", "📊 Batch Processing", "📈 Analytics", "⚙️ Settings"])

# Load models
if not st.session_state.embed_model:
    with st.spinner("Loading AI models..."):
        models = load_models()
        if models[0] is not None:
            st.session_state.embed_model, st.session_state.summarizer, st.session_state.ner_model = models
        else:
            st.error("Failed to load models. Please check your setup.")

with tab1:
    if not st.session_state.embed_model:
        st.error("Models not loaded. Please check your setup and refresh the page.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File upload
            uploaded_pdf = st.file_uploader("📄 Upload PDF Document", type=["pdf"])
            
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
                            
                            st.success(f"✅ Document processed! Found {len(chunks)} chunks and {len(entities)} entities.")
                        else:
                            st.error("Failed to extract text from PDF. Please try a different file.")
                    except Exception as e:
                        st.error(f"Error processing document: {e}")
        
        with col2:
            if st.session_state.faiss_index:
                st.metric("Document Status", "✅ Ready", "Indexed")
                st.metric("Chunks", len(st.session_state.chunk_texts) if st.session_state.chunk_texts else 0)
                if hasattr(st.session_state, 'document_entities'):
                    st.metric("Entities Found", len(st.session_state.document_entities))
        
        # Audit interface
        if st.session_state.faiss_index:
            st.markdown("### 🔍 Audit AI Answer")
            
            with st.form("enhanced_audit_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    question = st.text_area("❓ Your Question", height=100)
                    model_name = st.selectbox("🤖 AI Model Used", 
                        ["GPT-4", "GPT-3.5", "Claude", "Gemini", "Other"], 
                        index=0)
                
                with col2:
                    model_answer = st.text_area("🧠 Model's Answer", height=100)
                    audit_mode = st.selectbox("🎯 Audit Mode", 
                        ["Standard", "Strict", "Lenient"], 
                        index=0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_summary = st.checkbox("📝 Generate Summary", True)
                with col2:
                    show_entities = st.checkbox("🏷️ Show Entity Analysis", enable_entity_matching)
                with col3:
                    show_confidence = st.checkbox("📊 Show Confidence Breakdown", show_advanced)
                
                submitted = st.form_submit_button("🔍 Audit Answer", use_container_width=True)
            
            if submitted and question and model_answer:
                with st.spinner("Analyzing answer..."):
                    try:
                        # Enhanced audit
                        audit_result = enhanced_audit(
                            question=question,
                            model_answer=model_answer,
                            embed_model=st.session_state.embed_model,
                            faiss_index=st.session_state.faiss_index,
                            chunk_texts=st.session_state.chunk_texts,
                            ner_model=st.session_state.ner_model if enable_entity_matching else None,
                            document_entities=getattr(st.session_state, 'document_entities', []),
                            mode=audit_mode.lower()
                        )
                        
                        # Display results
                        trust_score = audit_result['trust_score']
                        trust_class = "trust-high" if trust_score >= 80 else "trust-medium" if trust_score >= 50 else "trust-low"
                        
                        st.markdown("### 📊 Audit Results")
                        
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
                                       marker_color=['#28a745', '#17a2b8', '#ffc107', '#dc3545'])
                            ])
                            fig.update_layout(title="Trust Score Breakdown", yaxis_title="Score")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Relevant content
                        st.markdown("### 📖 Most Relevant Content")
                        highlighted_chunk = audit_result.get('highlighted_chunk', audit_result['best_chunk'])
                        st.markdown(highlighted_chunk)
                        
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
                            'audit_result': audit_result
                        }
                        st.session_state.audit_history.append(audit_record)
                        
                    except Exception as e:
                        st.error(f"Error during audit: {e}")
                        st.error("Please check your inputs and try again.")

with tab2:
    st.markdown("### 📊 Batch Processing")
    st.markdown("Upload multiple Q&A pairs for bulk auditing")
    
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
    
    if batch_file and st.session_state.faiss_index:
        try:
            batch_df = pd.read_csv(batch_file)
            st.dataframe(batch_df.head())
            
            if st.button("🚀 Process Batch"):
                with st.spinner("Processing batch audits..."):
                    try:
                        results = process_batch_audits(
                            batch_df, 
                            st.session_state.embed_model,
                            st.session_state.faiss_index,
                            st.session_state.chunk_texts,
                            st.session_state.ner_model if enable_entity_matching else None
                        )
                        
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
                            report = export_results(results_df, format='detailed')
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
        fig = px.line(history_df, x='timestamp', y='trust_score', 
                     title='Trust Score Over Time', 
                     color='model')
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution by model
        col1, col2 = st.columns(2)
        with col1:
            model_stats = history_df.groupby('model')['trust_score'].agg(['mean', 'count']).reset_index()
            fig = px.bar(model_stats, x='model', y='mean', 
                        title='Average Trust Score by Model')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            status_counts = history_df['status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                        title='Answer Status Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent audits table
        st.markdown("### 📋 Recent Audits")
        recent_df = history_df.tail(10)[['timestamp', 'question', 'model', 'trust_score', 'status']]
        st.dataframe(recent_df, use_container_width=True)
    
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
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 200, 1000, 600)
        overlap_size = st.slider("Chunk Overlap", 50, 300, 150)
        max_chunks = st.slider("Max Chunks to Analyze", 3, 20, 5)
        
        st.markdown("#### Scoring Weights")
        semantic_weight = st.slider("Semantic Similarity Weight", 0.0, 1.0, 0.4)
        entity_weight = st.slider("Entity Matching Weight", 0.0, 1.0, 0.3)
        factual_weight = st.slider("Factual Consistency Weight", 0.0, 1.0, 0.3)

# Footer
st.markdown("---")
st.markdown("### 🔗 Quick Actions")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🎯 New Audit"):
        st.session_state.messages = []

with col2:
    if st.button("📊 View Stats"):
        if st.session_state.audit_history:
            avg_score = np.mean([a['trust_score'] for a in st.session_state.audit_history])
            st.success(f"Average Trust Score: {avg_score:.1f}%")

with col3:
    if st.button("💡 Tips"):
        st.info("💡 Tip: Use specific questions and compare multiple AI models for best results!")

with col4:
    if st.button("❓ Help"):
        st.markdown("""
        ### Quick Guide:
        1. Upload your PDF document
        2. Ask a question and provide the AI's answer
        3. Review the trust score and analysis
        4. Use batch processing for multiple Q&As
        """)