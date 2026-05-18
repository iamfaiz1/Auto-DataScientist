import streamlit as st
import pandas as pd
from src.data_profiler import DataProfiler
from src.task_detector import TaskDetector
from src.train_eval import Trainer
from src.explainer import AIExplainer

from src.config import USER_LEVELS
from src.visualizer import Visualizer
from src.ai_commentator import AICommentator
from src.smart_recommender import SmartRecommender

from src.pdf_reporter import ReportBuilder
from src.experiment_tracker import ExperimentTracker
import tempfile
from datetime import datetime


st.set_page_config(page_title="AI Data Science Assistant", layout="wide", page_icon="🤖")
# Initialize experiment tracker
tracker = ExperimentTracker()

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
    
    # --- SIDEBAR NAVIGATION & SETTINGS ---
    st.sidebar.header("⚙️ Settings")
    user_level = st.sidebar.radio("User Level", list(USER_LEVELS.keys()), help="Beginner = simplified, Advanced = full control")
    
    # Load user level config
    config = USER_LEVELS[user_level]
    
    # Color badge for user level
    level_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
    st.sidebar.markdown(f"**Current Level:** {level_colors[user_level]} {user_level}")

    with st.expander("📊 Dataset Intelligence", expanded=True):
        profile = profiler.generate_profile()
        
        # --- HEALTH SCORE DISPLAY ---
        col_score, col_personality = st.columns(2)
        with col_score:
            # Color-coded health score
            health = profile["Health Score"]
            if health >= 80:
                color = "🟢"
            elif health >= 60:
                color = "🟡"
            else:
                color = "🔴"
            st.metric(f"{color} Health Score", f"{health}/100")
        
        with col_personality:
            st.metric("Dataset Personality", profile["Personality"])
        
        # --- ISSUES & RECOMMENDATIONS ---
        if profile["Issues"]:
            with st.container():
                st.warning("**⚠️ Dataset Issues Detected:**")
                for issue in profile["Issues"]:
                    st.write(f"• {issue}")
                
                st.info("**💡 Recommendations:**")
                for rec in profile["Recommendations"]:
                    st.write(f"• {rec}")
        else:
            st.success("✅ No major issues detected!")
        
        # --- METRICS GRID ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Rows", profile["Total Rows"])
        m2.metric("Total Columns", profile["Total Columns"])
        m3.metric("Numeric Features", profile["Numerical Columns"])
        m4.metric("Missing Values", profile["Missing Values"])
        
        # --- VISUALIZATIONS ---
        c1, c2 = st.columns(2)
        with c1: 
            st.plotly_chart(viz.plot_correlation_heatmap(), use_container_width=True)
        with c2: 
            st.dataframe(df.head(10), use_container_width=True)
    

    # --- 2. TARGET ANALYSIS ---
    st.divider()
    st.subheader("🎯 2. Select Prediction Target")
    target_col = st.selectbox("What do you want to predict?", df.columns.tolist())
    
    # --- AI INSIGHTS PANEL ---
    st.divider()
    with st.expander("🤖 AI Analysis & Commentary", expanded=True):
        commentaries = AICommentator.generate_full_commentary(df, profile, "pending", target_col)
        
        for commentary in commentaries:
            st.write(commentary)
    
    
    detector = TaskDetector(df, target_col)
    task_type = detector.detect()
    
    if config["show_explanations"]:
        st.info(AIExplainer.explain_task(target_col, df[target_col].nunique(), str(df[target_col].dtype), task_type))
    else:
        st.info(f"**Task Type:** {task_type}")
    
    st.plotly_chart(viz.plot_target_distribution(target_col, task_type), use_container_width=True)
    
    # --- SMART MODEL RECOMMENDATIONS ---
    st.divider()
    with st.expander("💡 Smart Model Recommendations", expanded=True):
        recommendations = SmartRecommender.recommend_models(df, target_col, task_type, profile)
        rec_text = SmartRecommender.generate_recommendation_text(recommendations, task_type)
        st.markdown(rec_text)
        
        # Show recommendation scores
        rec_df = pd.DataFrame(recommendations)
        st.metric("Top Recommendation Score", f"{rec_df['score'].iloc[0]}/100")


# --- 3. AUTO-ML PIPELINE ---
    if st.button("🚀 Run AI Analysis"):
        with st.spinner("🤖 Training models and generating insights..."):
            trainer = Trainer(df, target_col, task_type)
            results = trainer.run_pipeline()
            
            if results:
                best_model = results[0]
                
                st.success("### 🏆 Final Recommendation")
                
                if config["show_explanations"]:
                    st.write(AIExplainer.explain_model_selection(
                        best_model['model_name'], 
                        best_model['metric_name'], 
                        best_model['score']
                    ))
                    st.write(AIExplainer.generate_business_insights(
                        best_model['model_name'],
                        best_model['score'],
                        task_type
                    ))
                else:
                    st.write(f"**Best Model:** {best_model['model_name']}")
                    st.write(f"**Score ({best_model['metric_name']}):** {best_model['score']:.4f}")
                
                st.divider()
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("📊 Model Leaderboard")
                    results_df = pd.DataFrame(results).drop(columns=['pipeline'])
                    st.dataframe(results_df, use_container_width=True)
                
                with col_b:
                    st.subheader("🎯 Feature Importance")
                    importance_plot = viz.plot_feature_importance(best_model['pipeline'], target_col)
                    if importance_plot:
                        st.plotly_chart(importance_plot, use_container_width=True)
                    else:
                        st.warning("Feature importance unavailable for this model.")
                
                st.session_state["analysis_results"] = results
                st.session_state["analysis_profile"] = profile
                st.session_state["analysis_task_type"] = task_type
                st.session_state["analysis_target_col"] = target_col
                st.session_state["analysis_dataset_name"] = uploaded_file.name.replace(".csv", "")
                st.session_state["analysis_timestamp"] = datetime.now().isoformat()
            
            # st.balloons()
            st.success("✅ Analysis complete!")

    analysis_results = st.session_state.get("analysis_results")
    if analysis_results:
        st.divider()
        col_report, col_export = st.columns(2)
        
        with col_report:
            if st.button("📄 Generate PDF Report"):
                try:
                    with st.spinner("Generating PDF report..."):
                        report_builder = ReportBuilder()
                        commentaries = AICommentator.generate_full_commentary(
                            df, profile, task_type, target_col
                        )
                        reporter = report_builder.build_analysis_report(
                            df, profile, task_type, target_col, 
                            analysis_results, commentaries, profile["Recommendations"]
                        )
                        pdf_bytes = reporter.generate()
                        
                        st.download_button(
                            label="📥 Download Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("✅ PDF report generated!")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
        
        with col_export:
            if st.button("💾 Save to History"):
                dataset_name = st.session_state.get("analysis_dataset_name", uploaded_file.name.replace(".csv", ""))
                exp_id = tracker.log_experiment(
                    dataset_name, task_type, target_col, 
                    df, profile, analysis_results,
                    notes=f"Health: {profile['Health Score']}/100"
                )
                st.success(f"✅ Experiment saved! (ID: {exp_id})")

    # --- EXPERIMENT HISTORY SECTION ---
    st.divider()
    history = tracker.get_history(limit=15)
    
    with st.expander("📚 Experiment History", expanded=False):
        if history:
            # Display history as interactive table
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df, use_container_width=True)
            
            st.divider()
            st.subheader("Experiment Statistics")
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Total Experiments", len(history))
            
            with col_stat2:
                avg_health = hist_df['health_score'].mean()
                st.metric("Avg Health Score", f"{avg_health:.0f}/100")
            
            with col_stat3:
                st.metric("Most Used Model", hist_df['best_model'].mode()[0] if len(hist_df) > 0 else "N/A")
            
            # Model performance over time
            st.subheader("Best Models Used")
            model_stats = tracker.get_model_stats()
            if model_stats:
                model_df = pd.DataFrame(model_stats)
                st.bar_chart(model_df.set_index('best_model')['avg_score'])
        else:
            st.info("No experiment history yet. Run an analysis and save it!")


else:
    st.info("Please upload a CSV file to begin the automated analysis.")