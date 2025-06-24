# Enhanced AI Knowledge Auditor v5.0 - Fixed Version
# app.py
import numpy as np
import torch
import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import time

# --- Monkey-patch torch.Tensor.numpy to catch the RuntimeError ---
_orig_tensor_numpy = torch.Tensor.numpy
def _safe_tensor_numpy(self, *args, **kwargs):
    try:
        return _orig_tensor_numpy(self, *args, **kwargs)
    except RuntimeError:
        return self.tolist()
torch.Tensor.numpy = _safe_tensor_numpy

# Import your custom modules with error handling
try:
    from core.loader import extract_text_from_pdf, smart_chunk_text, extract_entities
    from core.embedder import load_embedder_safe, load_summarizer_safe, load_ner_model_safe
    from core.auditor import enhanced_audit, generate_detailed_report
    from core.vector_store import build_and_save_index, load_index_and_chunks, hybrid_search
    from core.batch_processor import process_batch_audits, export_results
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.error("Some modules may be missing. The app will run with limited functionality.")

# Configure page
st.set_page_config(
    page_title="AI Knowledge Auditor Pro", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .model-status { padding: 10px; border-radius: 5px; margin: 10px 0; }
    .status-loading { background-color: #fff3cd; border: 1px solid #ffeaa7; }
    .status-success { background-color: #d4edda; border: 1px solid #c3e6cb; }
    .status-error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        "embed_model": None,
        "summarizer": None,
        "ner_model": None,
        "models_loaded": False,
        "model_load_attempts": 0,
        "messages": [],
        "faiss_index": None,
        "chunk_texts": None,
        "audit_history": [],
        "current_document": None,
        "batch_results": None,
        "document_entities": [],
        "model_status": "not_loaded"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Safe model loading with retry logic and fallbacks
@st.cache_resource(show_spinner=False)
def load_models_safe():
    """Load models with comprehensive error handling and fallbacks"""
    models_status = {
        "embedder": {"model": None, "status": "loading", "error": None},
        "summarizer": {"model": None, "status": "loading", "error": None},
        "ner": {"model": None, "status": "loading", "error": None}
    }
    
    # Load embedding model with fallbacks
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try different models in order of preference
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2"
        ]
        
        for model_name in embedding_models:
            try:
                with st.spinner(f"Loading embedding model: {model_name}..."):
                    embed_model = SentenceTransformer(model_name, device="cpu")
                    models_status["embedder"]["model"] = embed_model
                    models_status["embedder"]["status"] = "success"
                    break
            except Exception as e:
                models_status["embedder"]["error"] = str(e)
                continue
        
        if not models_status["embedder"]["model"]:
            models_status["embedder"]["status"] = "failed"
            
    except Exception as e:
        models_status["embedder"]["status"] = "failed"
        models_status["embedder"]["error"] = str(e)
    
    # Load summarizer with fallback
    try:
        from transformers import pipeline
        
        summarizer_models = [
            "facebook/bart-large-cnn",
            "sshleifer/distilbart-cnn-12-6",
            "t5-small"
        ]
        
        for model_name in summarizer_models:
            try:
                with st.spinner(f"Loading summarizer: {model_name}..."):
                    summarizer = pipeline(
                        "summarization", 
                        model=model_name, 
                        device=-1,
                        torch_dtype=torch.float32
                    )
                    models_status["summarizer"]["model"] = summarizer
                    models_status["summarizer"]["status"] = "success"
                    break
            except Exception as e:
                models_status["summarizer"]["error"] = str(e)
                continue
                
        if not models_status["summarizer"]["model"]:
            models_status["summarizer"]["status"] = "failed"
            
    except Exception as e:
        models_status["summarizer"]["status"] = "failed"
        models_status["summarizer"]["error"] = str(e)
    
    # Load NER model with fallback
    try:
        import spacy
        from spacy import displacy
        
        # Try to load spaCy model
        try:
            with st.spinner("Loading NER model..."):
                ner_model = spacy.load("en_core_web_sm")
                models_status["ner"]["model"] = ner_model
                models_status["ner"]["status"] = "success"
        except OSError:
            # Fallback to simple NER using transformers
            try:
                from transformers import pipeline
                ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=-1)
                models_status["ner"]["model"] = ner_model
                models_status["ner"]["status"] = "success"
            except Exception as e:
                models_status["ner"]["status"] = "failed"
                models_status["ner"]["error"] = str(e)
                
    except Exception as e:
        models_status["ner"]["status"] = "failed"
        models_status["ner"]["error"] = str(e)
    
    return models_status

# Model status display
def display_model_status(models_status):
    """Display current model loading status"""
    st.markdown("### 🤖 Model Status")
    
    for model_type, status in models_status.items():
        if status["status"] == "success":
            st.markdown(f'<div class="model-status status-success">✅ {model_type.title()}: Ready</div>', 
                       unsafe_allow_html=True)
        elif status["status"] == "loading":
            st.markdown(f'<div class="model-status status-loading">⏳ {model_type.title()}: Loading...</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="model-status status-error">❌ {model_type.title()}: Failed</div>', 
                       unsafe_allow_html=True)
            if status.get("error"):
                st.markdown(f'<small>Error: {status["error"][:100]}...</small>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🧠 AI Auditor Pro")
    
    # Model loading section
    if not st.session_state.models_loaded:
        st.markdown("### 🔄 Loading Models")
        
        if st.button("🚀 Load Models", use_container_width=True):
            with st.spinner("Loading AI models... This may take a few minutes on first run."):
                models_status = load_models_safe()
                
                # Update session state
                if models_status["embedder"]["model"]:
                    st.session_state.embed_model = models_status["embedder"]["model"]
                if models_status["summarizer"]["model"]:
                    st.session_state.summarizer = models_status["summarizer"]["model"]
                if models_status["ner"]["model"]:
                    st.session_state.ner_model = models_status["ner"]["model"]
                
                # Check if at least embedding model loaded (minimum requirement)
                if st.session_state.embed_model:
                    st.session_state.models_loaded = True
                    st.session_state.model_status = "loaded"
                    st.success("✅ Models loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load required models. Please try again.")
                    display_model_status(models_status)
    else:
        st.success("✅ Models Ready")
    
    st.markdown("### 📚 Features")
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

# Main interface
st.title("🧠 AI Knowledge Auditor Pro")
st.caption("Advanced AI answer verification with multi-factor analysis")

# Show model loading status if not loaded
if not st.session_state.models_loaded:
    st.warning("⚠️ Models not loaded yet. Please click 'Load Models' in the sidebar to begin.")
    st.info("💡 First-time loading may take several minutes as models are downloaded.")
    st.stop()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Single Audit", "📊 Batch Processing", "📈 Analytics", "⚙️ Settings"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_pdf = st.file_uploader("📄 Upload PDF Document", type=["pdf"], help="Upload a PDF document to analyze")
        
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
                        
                        # Extract entities if NER model is available
                        if st.session_state.ner_model:
                            entities = extract_entities(text, st.session_state.ner_model)
                        else:
                            entities = []
                        
                        # Build index
                        build_and_save_index(chunks, st.session_state.embed_model)
                        st.session_state.faiss_index, st.session_state.chunk_texts = load_index_and_chunks()
                        st.session_state.current_document = uploaded_pdf
                        st.session_state.document_entities = entities
                        
                        st.success(f"✅ Document processed! Found {len(chunks)} chunks and {len(entities)} entities.")
                    else:
                        st.error("Failed to extract text from PDF. Please try a different file.")
                        
                except Exception as e:
                    st.error(f"Error processing document: {e}")
                    st.info("Try uploading a different PDF file or check if the file is corrupted.")
    
    with col2:
        if st.session_state.faiss_index:
            st.metric("Document Status", "✅ Ready", "Indexed")
            st.metric("Chunks", len(st.session_state.chunk_texts) if st.session_state.chunk_texts else 0)
            if hasattr(st.session_state, 'document_entities'):
                st.metric("Entities Found", len(st.session_state.document_entities))
    
    # Audit interface
    if st.session_state.faiss_index:
        st.markdown("### 🔍 Audit AI Answer")
        
        with st.form("enhanced_audit_form", clear_on_submit=False):
            col1, col2 = st.columns(2)
            
            with col1:
                question = st.text_area("❓ Your Question", height=100, 
                                      placeholder="Enter the question you asked the AI...")
                model_name = st.selectbox("🤖 AI Model Used", 
                    ["GPT-4", "GPT-3.5", "Claude", "Gemini", "Other"], 
                    index=0)
            
            with col2:
                model_answer = st.text_area("🧠 Model's Answer", height=100,
                                          placeholder="Paste the AI model's response here...")
                audit_mode = st.selectbox("🎯 Audit Mode", 
                    ["Standard", "Strict", "Lenient"], 
                    index=0,
                    help="Standard: balanced, Strict: more conservative, Lenient: more forgiving")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                show_summary = st.checkbox("📝 Generate Summary", True)
            with col2:
                show_entities = st.checkbox("🏷️ Show Entity Analysis", 
                                          enable_entity_matching and st.session_state.ner_model is not None)
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
                        st.markdown(f'<div class="metric-card"><h3>Trust Score</h3><h1 class="{trust_class}">{trust_score}%</h1></div>', 
                                  unsafe_allow_html=True)
                    with col2:
                        status = "✅ Trusted" if trust_score >= 80 else "⚠️ Caution" if trust_score >= 50 else "❌ Suspicious"
                        st.markdown(f'<div class="metric-card"><h3>Status</h3><h2>{status}</h2></div>', 
                                  unsafe_allow_html=True)
                    with col3:
                        confidence = audit_result.get('confidence_level', 'Medium')
                        st.markdown(f'<div class="metric-card"><h3>Confidence</h3><h2>{confidence}</h2></div>', 
                                  unsafe_allow_html=True)
                    
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
                    # Show debug info in development
                    if st.checkbox("Show debug info"):
                        st.exception(e)

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
        
        # Display current model status
        if st.session_state.embed_model:
            st.success("✅ Embedding model: Loaded")
        else:
            st.error("❌ Embedding model: Not loaded")
            
        if st.session_state.summarizer:
            st.success("✅ Summarizer: Loaded")
        else:
            st.warning("⚠️ Summarizer: Not loaded")
            
        if st.session_state.ner_model:
            st.success("✅ NER model: Loaded")
        else:
            st.warning("⚠️ NER model: Not loaded")
        
        if st.button("🔄 Reload Models"):
            # Clear cached models
            st.cache_resource.clear()
            st.session_state.embed_model = None
            st.session_state.summarizer = None
            st.session_state.ner_model = None
            st.session_state.models_loaded = False
            st.session_state.model_status = "not_loaded"
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
    if st.button("📊 View Stats") and st.session_state.audit_history:
        avg_score = np.mean([a['trust_score'] for a in st.session_state.audit_history])
        st.success(f"Average Trust Score: {avg_score:.1f}%")

with col3:
    if st.button("💡 Tips"):
        st.info("💡 Tip: Use specific questions and compare multiple AI models for best results!")

with col4:
    if st.button("❓ Help"):
        st.markdown("""
        ### Quick Guide:
        1. **Load Models**: Click 'Load Models' in sidebar (first time only)
        2. **Upload PDF**: Upload your reference document  
        3. **Enter Q&A**: Add your question and AI's answer
        4. **Review Results**: Check trust score and analysis
        5. **Batch Process**: Use CSV for multiple Q&As at once
        """)

# Status indicator at bottom
if st.session_state.models_loaded:
    st.success("🟢 System Ready - All models loaded successfully")
else:
    st.warning("🟡 System Initializing - Please load models to begin")