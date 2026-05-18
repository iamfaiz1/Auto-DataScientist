"""
Manages workflow state and navigation.
Controls which sections are visible based on user progress.
"""

class WorkflowManager:
    """Tracks workflow progress and visibility of sections."""
    
    STAGES = {
        "1_upload": {
            "name": "Upload",
            "description": "Load your CSV dataset",
            "icon": "📤",
        },
        "2_explore": {
            "name": "Explore",
            "description": "Analyze dataset quality & properties",
            "icon": "🔍",
        },
        "3_target": {
            "name": "Target",
            "description": "Select prediction target",
            "icon": "🎯",
        },
        "4_train": {
            "name": "Train",
            "description": "Train and evaluate models",
            "icon": "🚀",
        },
        "5_insights": {
            "name": "Insights",
            "description": "View AI commentary & recommendations",
            "icon": "💡",
        },
        "6_report": {
            "name": "Report",
            "description": "Generate and download reports",
            "icon": "📄",
        },
    }
    
    @staticmethod
    def get_current_stage(has_file, has_target, has_results):
        """Determine current workflow stage."""
        if not has_file:
            return "1_upload"
        elif not has_target:
            return "2_explore"
        elif has_target and not has_results:
            return "3_target"
        elif has_results:
            return "6_report"
        else:
            return "4_train"
    
    @staticmethod
    def get_available_stages(has_file, has_target, has_results):
        """Get list of available stages based on progress."""
        available = ["1_upload"]
        
        if has_file:
            available.append("2_explore")
        
        if has_file and has_target:
            available.append("3_target")
        
        if has_file and has_target:
            available.append("4_train")
        
        if has_results:
            available.extend(["5_insights", "6_report"])
        
        return available
    
    @staticmethod
    def render_workflow_tabs(selected_stage, available_stages):
        """Return formatted workflow progress display."""
        progress_items = []
        
        for stage_id in WorkflowManager.STAGES.keys():
            stage = WorkflowManager.STAGES[stage_id]
            
            if stage_id == selected_stage:
                progress_items.append(f"**{stage['icon']} {stage['name']}** (Current)")
            elif stage_id in available_stages:
                progress_items.append(f"{stage['icon']} {stage['name']}")
            else:
                progress_items.append(f"⭕ {stage['name']} (Locked)")
        
        return "\n".join(progress_items)
    
    @staticmethod
    def get_progress_percentage(has_file, has_target, has_results):
        """Calculate workflow completion percentage."""
        progress = 0
        
        if has_file:
            progress += 20
        if has_target:
            progress += 20
        if has_results:
            progress += 60
        
        return progress