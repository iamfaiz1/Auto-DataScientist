"""
Dashboard layout utilities for improved UI organization.
Provides reusable components for consistent styling.
"""

import streamlit as st

class DashboardLayout:
    """Provides dashboard layout utilities and components."""
    
    @staticmethod
    def render_metric_card(title, value, subtitle="", icon=""):
        """Render a styled metric card."""
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
        ">
            <div style="color: rgba(255,255,255,0.8); font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">
                {icon} {title}
            </div>
            <div style="color: white; font-size: 28px; font-weight: bold; margin: 10px 0;">
                {value}
            </div>
            <div style="color: rgba(255,255,255,0.7); font-size: 12px;">
                {subtitle}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_insight_card(insight):
        """Render an insight as a styled card."""
        color_map = {
            "risk_alert": "#ef4444",
            "warning": "#f97316",
            "success": "#10b981",
            "info": "#3b82f6"
        }
        
        color = color_map.get(insight["type"], "#3b82f6")
        
        st.markdown(f"""
        <div style="
            border-left: 4px solid {color};
            background: rgba({color}, 0.05);
            padding: 16px;
            border-radius: 8px;
            margin: 12px 0;
        ">
            <div style="font-weight: bold; color: white; margin-bottom: 8px;">
                {insight['title']}
            </div>
            <div style="color: rgba(255,255,255,0.9); font-size: 14px; margin-bottom: 8px;">
                {insight['description']}
            </div>
            <div style="color: rgba(255,255,255,0.7); font-size: 12px;">
                <strong>Action:</strong> {insight['action']}<br/>
                <strong>Impact:</strong> {insight['impact']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_section_header(title, icon="", subtitle=""):
        """Render a styled section header."""
        st.markdown(f"""
        <div style="margin: 30px 0 20px 0;">
            <h2 style="color: white; margin: 0 0 8px 0;">
                {icon} {title}
            </h2>
            {f'<p style="color: rgba(255,255,255,0.7); margin: 0; font-size: 14px;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_key_metrics(metrics_dict):
        """Render a grid of key metrics."""
        n_metrics = len(metrics_dict)
        cols = st.columns(min(4, n_metrics))
        
        for idx, (label, value) in enumerate(metrics_dict.items()):
            col = cols[idx % len(cols)]
            with col:
                st.metric(label, value)
    
    @staticmethod
    def render_divider():
        """Render a styled divider."""
        st.markdown("""
        <div style="height: 1px; background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.3), rgba(255,255,255,0)); margin: 20px 0;"></div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_collapsible_section(title, icon, content_func):
        """Render a collapsible section that calls a function."""
        with st.expander(f"{icon} {title}", expanded=False):
            content_func()
    
    @staticmethod
    def render_model_card(model_name, score, rank, metric_name):
        """Render a model leaderboard card."""
        rank_colors = ["🥇", "🥈", "🥉"]
        rank_icon = rank_colors[rank-1] if rank <= 3 else f"#{rank}"
        
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            border: 1px solid rgba(79, 70, 229, 0.3);
            padding: 16px;
            border-radius: 8px;
            margin: 10px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 18px; font-weight: bold; color: white;">
                        {rank_icon} {model_name}
                    </div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 12px; margin-top: 4px;">
                        {metric_name}
                    </div>
                </div>
                <div style="font-size: 24px; font-weight: bold; color: #4f46e5;">
                    {score:.4f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_progress_indicator(value, max_value=100, label=""):
        """Render a custom progress indicator."""
        pct = (value / max_value) * 100
        
        st.markdown(f"""
        <div style="margin: 10px 0;">
            {f'<div style="color: rgba(255,255,255,0.7); font-size: 12px; margin-bottom: 4px;">{label}</div>' if label else ''}
            <div style="
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                overflow: hidden;
                height: 8px;
            ">
                <div style="
                    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
                    width: {pct}%;
                    height: 100%;
                    transition: width 0.3s ease;
                "></div>
            </div>
            <div style="color: rgba(255,255,255,0.6); font-size: 12px; margin-top: 4px;">
                {pct:.0f}%
            </div>
        </div>
        """, unsafe_allow_html=True)