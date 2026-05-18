import streamlit as st
import pandas as pd
from src.data_profiler import DataProfiler
from src.task_detector import TaskDetector
from src.train_eval import Trainer
from src.explainer import AIExplainer
from src.visualizer import Visualizer

st.set_page_config(page_title="AI Data Science Assistant", layout="wide", page_icon="🤖")

# Custom CSS for a "Pro" Look
st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: #020817;
    color: white;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1f2937;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
}

/* Metric label */
div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
}

/* Metric value */
div[data-testid="metric-container"] div {
    color: white !important;
}

/* Buttons */
.stButton > button {
    background-color: #4f46e5;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
}


h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 Auto Data Science Assistant")
st.markdown("Automated Analysis, Model Recommendation, and Insights")
st.divider()


uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    # --- 1. DATA & PROFILING ---
    df_raw = pd.read_csv(uploaded_file)
    profiler = DataProfiler(df_raw)
    df = profiler.optimize_memory()
    viz = Visualizer(df)
    
    st.sidebar.header("Settings")
    mode = st.sidebar.radio("User Level", ["Beginner", "Advanced"])

    with st.expander("📊 1. Dataset Intelligence Report", expanded=True):
        profile = profiler.generate_profile()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", profile["Total Rows"])
        m2.metric("Total Columns", profile["Total Columns"])
        m3.metric("Numeric Features", profile["Numerical Columns"])
        m4.metric("Missing Values", profile["Missing Values"])
        
        c1, c2 = st.columns(2)
        with c1: 
            st.plotly_chart(viz.plot_correlation_heatmap(), use_container_width=True)
        with c2: 
            st.dataframe(df.head(10))

    # --- 2. TARGET ANALYSIS ---
    st.divider()
    st.subheader("🎯 2. Select Prediction Target")
    target_col = st.selectbox("What do you want to predict?", df.columns.tolist())
    
    detector = TaskDetector(df, target_col)
    task_type = detector.detect()
    
    st.info(AIExplainer.explain_task(target_col, df[target_col].nunique(), str(df[target_col].dtype), task_type))
    st.plotly_chart(viz.plot_target_distribution(target_col, task_type), use_container_width=True)

    # --- 3. AUTO-ML PIPELINE ---
    if st.button("🚀 Run AI Analysis"):
        with st.spinner("🤖 Training models and generating insights..."):
            trainer = Trainer(df, target_col, task_type)
            results = trainer.run_pipeline()
            
            if results:
                # --- CRITICAL FIX START ---
                # 'results' is a LIST. We need the first DICTIONARY in that list.
                best_model = results[0]
                # --- CRITICAL FIX END ---
                
                st.success("### 🏆 Final Recommendation")
                # Now best_model['model_name'] refers to a dictionary key, not a list index
                st.write(AIExplainer.explain_model_selection(
                    best_model['model_name'], 
                    best_model['metric_name'], 
                    best_model['score']
                ))
                
                # Visual Explanations
                st.divider()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Leaderboard")
                    # We display the full list in the table
                    st.table(pd.DataFrame(results).drop(columns=['pipeline']))
                with col_b:
                    # We use the dictionary 'best_model' for the visualization
                    importance_plot = viz.plot_feature_importance(best_model['pipeline'], target_col)
                    if importance_plot:
                        st.plotly_chart(importance_plot, use_container_width=True)
                    else:
                        st.warning("Feature importance not available for this model type.")
            
            st.balloons()
else:
    st.info("Please upload a CSV file to begin the automated analysis.")