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
from src.difficulty_estimator import DifficultyEstimator
from src.workflow_manager import WorkflowManager
from src.business_insights import BusinessInsights
from src.dashboard_layout import DashboardLayout

st.set_page_config(
    page_title="Auto DataScientist", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

tracker = ExperimentTracker()

# Enhanced Custom CSS
st.markdown("""
<style>

/* Main app background */
.stApp {
    background: linear-gradient(135deg, #020817 0%, #0f172a 100%);
    color: white;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border: 1px solid rgba(79, 70, 229, 0.2);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

div[data-testid="metric-container"] div {
    color: white !important;
    font-size: 28px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    transform: translateY(-2px);
}

/* Headers */
h1, h2, h3 {
    color: white;
    font-weight: 700;
}

h1 {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(79, 70, 229, 0.1);
    border-radius: 8px;
    padding: 12px;
}

.streamlit-expanderHeader:hover {
    background: rgba(79, 70, 229, 0.2);
}

/* Dividers */
hr {
    background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.3), rgba(255,255,255,0));
    border: none;
    height: 1px;
}

/* Dataframe styling */
.stDataFrame {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
}

/* Info/Warning boxes */
.stAlert {
    border-radius: 8px;
    padding: 16px;
}

/* Selectbox styling */
.stSelectbox, .stRadio {
    margin: 10px 0;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 Auto Data Science Assistant")
st.markdown("Automated Analysis, Model Recommendation, and Insights")
st.divider()

# --- WORKFLOW NAVIGATION SIDEBAR ---
with st.sidebar:
    st.markdown("### 📋 Workflow Progress")
    
    # Create session state for tracking
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'target_col_selected' not in st.session_state:
        st.session_state.target_col_selected = None
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    
    has_file = st.session_state.uploaded_file_name is not None
    has_target = st.session_state.target_col_selected is not None
    has_results = st.session_state.analysis_completed
    
    current_stage = WorkflowManager.get_current_stage(has_file, has_target, has_results)
    available_stages = WorkflowManager.get_available_stages(has_file, has_target, has_results)
    progress_pct = WorkflowManager.get_progress_percentage(has_file, has_target, has_results)
    
    # Progress bar
    st.progress(progress_pct / 100, text=f"{progress_pct}% Complete")
    
    st.divider()
    
    # Workflow stages
    for stage_id, stage_info in WorkflowManager.STAGES.items():
        if stage_id == current_stage:
            st.markdown(f"**{stage_info['icon']} {stage_info['name']}**", help=stage_info['description'])
            st.markdown(f"<span style='color: #4f46e5;'>→ Current Stage</span>", unsafe_allow_html=True)
        elif stage_id in available_stages:
            st.markdown(f"{stage_info['icon']} {stage_info['name']}", help=stage_info['description'])
        else:
            st.markdown(f"⭕ {stage_info['name']}", help="Complete previous stages to unlock")
    
    st.divider()
    
    # User level selection
    st.markdown("### ⚙️ Settings")
    user_level = st.radio(
        "User Level", 
        list(USER_LEVELS.keys()),
        help="Beginner = simplified, Advanced = full control"
    )
    config = USER_LEVELS[user_level]
    
    level_colors = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
    st.markdown(f"**Current Level:** {level_colors[user_level]} {user_level}")


# --- STAGE 1: UPLOAD ---
st.subheader("📤 Stage 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    st.session_state.uploaded_file_name = uploaded_file.name
else:
    st.session_state.uploaded_file_name = None

if uploaded_file:
    # --- 1. DATA & PROFILING ---
    df_raw = pd.read_csv(uploaded_file)
    profiler = DataProfiler(df_raw)
    df = profiler.optimize_memory()
    viz = Visualizer(df)
    profile = profiler.generate_profile()

    # --- STAGE 2: EXPLORE & ANALYZE ---
    st.subheader("🔍 Stage 2: Explore & Analyze")
    st.markdown("**Dataset Overview & Quality Assessment**")

    with st.expander("📊 Dataset Intelligence", expanded=True):
        # Key Metrics Grid
        DashboardLayout.render_section_header("Key Metrics", "📈")

        col_score, col_personality, col_rows, col_cols = st.columns(4)

        with col_score:
            health = profile["Health Score"]
            if health >= 80:
                color = "🟢"
            elif health >= 60:
                color = "🟡"
            else:
                color = "🔴"
            st.metric(f"{color} Health", f"{health}/100")

        with col_personality:
            st.metric("Personality", profile["Personality"])

        with col_rows:
            st.metric("Rows", f"{profile['Total Rows']:,}")

        with col_cols:
            st.metric("Columns", profile["Total Columns"])

        # Issues and recommendations
        if profile["Issues"]:
            DashboardLayout.render_section_header("Issues & Recommendations", "⚠️")

            col_issues, col_recs = st.columns(2)

            with col_issues:
                st.markdown("**Detected Issues:**")
                for issue in profile["Issues"]:
                    st.write(f"• {issue}")

            with col_recs:
                st.markdown("**Recommended Actions:**")
                for rec in profile["Recommendations"]:
                    st.write(f"• {rec}")
        else:
            st.success("✅ No major issues detected!")

        DashboardLayout.render_divider()

        # Visualizations
        DashboardLayout.render_section_header("Data Patterns", "📊")

        col_corr, col_data = st.columns(2)

        with col_corr:
            st.plotly_chart(viz.plot_correlation_heatmap(), use_container_width=True)

        with col_data:
            st.markdown("**Sample Data (First 10 Rows)**")
            st.dataframe(df.head(10), use_container_width=True)

    # --- AI INSIGHTS PANEL ---
    st.divider()
    with st.expander("🤖 AI Analysis & Commentary", expanded=True):
        commentaries = AICommentator.analyze_dataset_quality(df, profile)
        for commentary in commentaries:
            st.write(commentary)

    # --- STAGE 3: TARGET ANALYSIS ---
    DashboardLayout.render_section_header("Select Prediction Target", "🎯")

    target_col = st.selectbox(
        "What do you want to predict?",
        df.columns.tolist(),
        help="Choose the column you want to build a model to predict"
    )

    if target_col:
        st.session_state.target_col_selected = target_col

    detector = TaskDetector(df, target_col)
    task_type = detector.detect()

    # Task explanation
    col_task, col_dist = st.columns([1, 1])

    with col_task:
        st.markdown(f"**Task Type:** {task_type}")
        if config["show_explanations"]:
            st.info(AIExplainer.explain_task(target_col, df[target_col].nunique(), str(df[target_col].dtype), task_type))
        else:
            st.write(f"Target has {df[target_col].nunique()} unique values ({df[target_col].dtype})")

    with col_dist:
        st.markdown("**Target Distribution**")
        st.plotly_chart(viz.plot_target_distribution(target_col, task_type), use_container_width=True)

    DashboardLayout.render_divider()

    # --- DIFFICULTY ASSESSMENT ---
    DashboardLayout.render_section_header("Problem Difficulty Analysis", "📊")

    difficulty_data = DifficultyEstimator.estimate_difficulty(df, profile, task_type, target_col)

    with st.expander("View Difficulty Assessment", expanded=True):
        col_diff, col_score, col_desc = st.columns([1, 1, 2])

        with col_diff:
            st.markdown(f"**Level:** {difficulty_data['difficulty']}")

        with col_score:
            DashboardLayout.render_progress_indicator(
                difficulty_data['score'], 
                100, 
                "Difficulty Score"
            )

        with col_desc:
            st.markdown(f"*{difficulty_data['description']}*")

        DashboardLayout.render_divider()

        # Factors breakdown
        st.markdown("**Contributing Factors:**")
        factor_cols = st.columns(3)

        for idx, (factor_name, (factor_status, factor_score)) in enumerate(difficulty_data['factors'].items()):
            col = factor_cols[idx % 3]
            with col:
                score_text = f"+{factor_score}" if factor_score >= 0 else f"{factor_score}"
                st.write(f"**{factor_name}**  \n{factor_status}  \n`{score_text}`")

        DashboardLayout.render_divider()

        # Mitigation strategies
        strategies = DifficultyEstimator.get_mitigation_strategies(difficulty_data, task_type)
        if strategies:
            st.markdown("**Recommended Strategies:**")
            for strategy in strategies:
                st.write(strategy)

    # --- SMART MODEL RECOMMENDATIONS ---
    DashboardLayout.render_section_header("Smart Model Recommendations", "💡")

    recommendations = SmartRecommender.recommend_models(df, target_col, task_type, profile)

    with st.expander("View Model Recommendations", expanded=True):
        rec_df = pd.DataFrame(recommendations)

        # Top recommendation
        top = recommendations[0]
        col_top_title, col_top_score = st.columns([3, 1])
        with col_top_title:
            st.markdown(f"### 🏆 Top Pick: {top['model']}")
            st.write(f"*{top['reason']}*")
        with col_top_score:
            st.metric("Score", f"{top['score']}/100")

        DashboardLayout.render_divider()

        # Alternative recommendations
        st.markdown("**Alternative Options:**")
        alt_cols = st.columns(len(recommendations)-1 if len(recommendations) > 1 else 1)

        for idx, rec in enumerate(recommendations[1:]):
            with alt_cols[idx]:
                st.markdown(f"**{rec['model']}**")
                st.write(rec['reason'])
                st.metric("Score", f"{rec['score']}/100")

    # --- STAGE 4: TRAIN ---
    st.divider()
    st.subheader("🚀 Stage 4: Train & Evaluate Models")
    
    if st.button("🚀 Run AI Analysis"):
        with st.spinner("🤖 Training models and generating insights..."):
            trainer = Trainer(df, target_col, task_type)
            results = trainer.run_pipeline()
            
            st.session_state.analysis_completed = True
            
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
                
                # --- STAGE 5 & 6: INSIGHTS & REPORTS ---
                st.divider()
                st.subheader("💡 Stage 5: AI Insights & Recommendations")
                
                with st.expander("🤖 Detailed AI Commentary", expanded=True):
                    commentaries = AICommentator.generate_full_commentary(df, profile, task_type, target_col)
                    
                    for commentary in commentaries:
                        st.write(commentary)
                
                st.divider()
                st.subheader("📄 Stage 6: Generate Reports")
                
                col_report, col_export = st.columns(2)
                
                with col_report:
                    if st.button("📄 Generate PDF Report"):
                        with st.spinner("Generating PDF report..."):
                            report_builder = ReportBuilder()
                            commentaries = AICommentator.generate_full_commentary(
                                df, profile, task_type, target_col
                            )
                            reporter = report_builder.build_analysis_report(
                                df, profile, task_type, target_col, 
                                results, commentaries, profile["Recommendations"]
                            )
                            
                            pdf_bytes = reporter.generate()
                            st.download_button(
                                label="📥 Download Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"ai_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                
                with col_export:
                    if st.button("💾 Save to History"):
                        dataset_name = uploaded_file.name.replace(".csv", "")
                        exp_id = tracker.log_experiment(
                            dataset_name, task_type, target_col, 
                            df, profile, results,
                            notes=f"Difficulty: {difficulty_data['difficulty']}"
                        )
                        st.success(f"✅ Experiment saved! (ID: {exp_id})")
                
                st.balloons()
    
    # --- EXPERIMENT HISTORY ---
    st.divider()
    with st.expander("📚 Experiment History", expanded=False):
        history = tracker.get_history(limit=15)
        
        if history:
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
            
            st.subheader("Best Models Used")
            model_stats = tracker.get_model_stats()
            if model_stats:
                model_df = pd.DataFrame(model_stats)
                st.bar_chart(model_df.set_index('best_model')['avg_score'])
        else:
            st.info("No experiment history yet. Run an analysis and save it!")

else:
    st.info("👆 Please upload a CSV file to begin the automated analysis.")