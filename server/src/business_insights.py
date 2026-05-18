"""
Generates business-focused insights from ML analysis.
Translates technical metrics into actionable business recommendations.
"""

import pandas as pd
import numpy as np

class BusinessInsights:
    """Generates business-style insights from data and model performance."""
    
    @staticmethod
    def generate_churn_insights(df, target_col, task_type, best_score):
        """Generate churn/retention focused insights."""
        if task_type != "Classification":
            return None
        
        target = df[target_col]
        positive_rate = (target.value_counts().iloc[0] / len(target)) * 100
        
        insights = []
        
        # Churn rate analysis
        if positive_rate < 30:
            insights.append({
                "type": "risk_alert",
                "title": "⚠️ High Churn Risk Detected",
                "description": f"Your churn rate is {100-positive_rate:.1f}%. This is above industry average.",
                "action": "Prioritize retention campaigns targeting at-risk customers.",
                "impact": "High"
            })
        elif positive_rate < 50:
            insights.append({
                "type": "warning",
                "title": "📊 Moderate Churn Observed",
                "description": f"Churn rate of {100-positive_rate:.1f}% suggests retention opportunities.",
                "action": "Develop targeted retention strategies for key customer segments.",
                "impact": "Medium"
            })
        else:
            insights.append({
                "type": "success",
                "title": "✅ Strong Retention",
                "description": f"Churn rate of {100-positive_rate:.1f}% indicates good customer loyalty.",
                "action": "Maintain current engagement strategies and monitor for changes.",
                "impact": "Low"
            })
        
        # Model confidence
        if best_score > 0.85:
            insights.append({
                "type": "success",
                "title": "🎯 High Prediction Accuracy",
                "description": f"Model accuracy of {best_score*100:.1f}% enables confident targeting.",
                "action": "Use model predictions to identify at-risk customers proactively.",
                "impact": "High"
            })
        elif best_score > 0.70:
            insights.append({
                "type": "info",
                "title": "📈 Moderate Prediction Reliability",
                "description": f"Model accuracy of {best_score*100:.1f}% suitable for business use.",
                "action": "Combine model insights with domain expertise for decisions.",
                "impact": "Medium"
            })
        
        return insights

    @staticmethod
    def generate_pricing_insights(df, target_col, task_type, best_score):
        """Generate pricing/revenue focused insights."""
        if task_type != "Regression":
            return None
        
        target = df[target_col]
        variance_explained = best_score * 100
        
        insights = []
        
        if variance_explained > 0.80:
            insights.append({
                "type": "success",
                "title": "💰 Strong Revenue Predictability",
                "description": f"Model explains {variance_explained:.1f}% of revenue variance.",
                "action": "Use predictions for dynamic pricing and revenue optimization.",
                "impact": "High"
            })
        elif variance_explained > 0.60:
            insights.append({
                "type": "info",
                "title": "📊 Moderate Revenue Trends Detected",
                "description": f"Model captures {variance_explained:.1f}% of price variance.",
                "action": "Identify key pricing drivers and test interventions.",
                "impact": "Medium"
            })
        else:
            insights.append({
                "type": "warning",
                "title": "🔍 Limited Price Predictability",
                "description": f"Model explains only {variance_explained:.1f}% of variance.",
                "action": "Investigate missing features or non-linear relationships.",
                "impact": "Low"
            })
        
        # Price range insights
        price_range = target.max() - target.min()
        price_std = target.std()
        
        if price_std > (price_range * 0.2):
            insights.append({
                "type": "info",
                "title": "📈 High Price Volatility",
                "description": f"Price variance suggests multiple customer segments.",
                "action": "Develop segment-specific pricing strategies.",
                "impact": "Medium"
            })
        
        return insights

    @staticmethod
    def generate_customer_insights(df, task_type, best_score):
        """Generate customer behavior focused insights."""
        insights = []
        
        # Feature count insight
        n_features = len(df.columns) - 1
        if n_features > 20:
            insights.append({
                "type": "info",
                "title": "👥 Complex Customer Profile",
                "description": f"Dataset includes {n_features} customer attributes.",
                "action": "Focus on top 5-10 features for campaign targeting.",
                "impact": "Medium"
            })
        
        # Data completeness
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        if missing_pct < 5:
            insights.append({
                "type": "success",
                "title": "✅ Clean Customer Data",
                "description": f"Only {missing_pct:.1f}% missing data—excellent data quality.",
                "action": "Proceed confidently with direct mailing/campaigns.",
                "impact": "High"
            })
        elif missing_pct > 20:
            insights.append({
                "type": "warning",
                "title": "⚠️ Data Quality Issues",
                "description": f"Missing data ({missing_pct:.1f}%) may affect targeting accuracy.",
                "action": "Invest in data enrichment or alternative data sources.",
                "impact": "Medium"
            })
        
        return insights

    @staticmethod
    def generate_sales_insights(df, target_col, task_type, best_score):
        """Generate sales/forecasting focused insights."""
        if task_type != "Regression":
            return None
        
        target = df[target_col]
        insights = []
        
        # Sales volume
        avg_sales = target.mean()
        sales_median = target.median()
        
        if avg_sales > sales_median * 1.5:
            insights.append({
                "type": "info",
                "title": "📊 Skewed Sales Distribution",
                "description": f"Few high-value deals inflate average. Median: {sales_median:.0f}",
                "action": "Segment high-value customers for specialized account management.",
                "impact": "High"
            })
        
        # Trend detection
        if len(df) > 100:
            recent = target.tail(len(target)//4).mean()
            earlier = target.head(len(target)//4).mean()
            trend = ((recent - earlier) / earlier) * 100 if earlier != 0 else 0
            
            if trend > 10:
                insights.append({
                    "type": "success",
                    "title": "📈 Positive Sales Trend",
                    "description": f"Recent sales {trend:.1f}% higher than earlier period.",
                    "action": "Replicate successful strategies from high-performing period.",
                    "impact": "High"
                })
            elif trend < -10:
                insights.append({
                    "type": "warning",
                    "title": "📉 Declining Sales Trend",
                    "description": f"Recent sales {abs(trend):.1f}% lower—investigation needed.",
                    "action": "Review market conditions, competition, and customer feedback.",
                    "impact": "High"
                })
        
        # Prediction confidence for forecasting
        if best_score > 0.80:
            insights.append({
                "type": "success",
                "title": "🎯 Reliable Sales Forecasting",
                "description": f"High model accuracy ({best_score*100:.1f}%) enables confident budgeting.",
                "action": "Use predictions for quarterly revenue forecasts.",
                "impact": "High"
            })
        
        return insights

    @staticmethod
    def detect_business_context(df, target_col, task_type):
        """Auto-detect business context from data characteristics."""
        target = df[target_col]
        target_lower = target_col.lower()
        
        # Keyword-based detection
        churn_keywords = ['churn', 'churn_flag', 'attrition', 'left', 'cancelled']
        price_keywords = ['price', 'revenue', 'sales', 'amount', 'cost', 'fee']
        
        if any(keyword in target_lower for keyword in churn_keywords):
            return "churn"
        elif task_type == "Regression" and any(keyword in target_lower for keyword in price_keywords):
            return "pricing"
        elif task_type == "Regression":
            return "sales"
        else:
            return "general"

    @staticmethod
    def generate_insights(df, target_col, task_type, best_score, best_model_name):
        """Main function to generate all relevant business insights."""
        context = BusinessInsights.detect_business_context(df, target_col, task_type)
        
        all_insights = []
        
        # Generate context-specific insights
        if context == "churn":
            all_insights.extend(BusinessInsights.generate_churn_insights(df, target_col, task_type, best_score) or [])
        elif context == "pricing":
            all_insights.extend(BusinessInsights.generate_pricing_insights(df, target_col, task_type, best_score) or [])
        elif context == "sales":
            all_insights.extend(BusinessInsights.generate_sales_insights(df, target_col, task_type, best_score) or [])
        
        # Always include customer insights
        all_insights.extend(BusinessInsights.generate_customer_insights(df, task_type, best_score) or [])
        
        return all_insights, context

    @staticmethod
    def format_insight(insight):
        """Format a single insight for display."""
        if insight["type"] == "risk_alert":
            icon = "🚨"
        elif insight["type"] == "warning":
            icon = "⚠️"
        elif insight["type"] == "success":
            icon = "✅"
        else:
            icon = "ℹ️"
        
        return f"""
**{insight['title']}**  
{insight['description']}  
💡 *Action:* {insight['action']}  
📊 *Impact:* {insight['impact']}
"""