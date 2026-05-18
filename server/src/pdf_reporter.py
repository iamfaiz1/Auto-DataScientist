"""
Lightweight PDF report generation using ReportLab.
Generates professional reports without heavy dependencies.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import io

class PDFReporter:
    def __init__(self, filename="report.pdf"):
        self.filename = filename
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4f46e5'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#1f2937'),
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            alignment=TA_LEFT,
            spaceAfter=6
        ))

    def add_title(self, title, subtitle=""):
        """Add title section."""
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        if subtitle:
            self.story.append(Paragraph(f"<i>{subtitle}</i>", self.styles['CustomBody']))
        self.story.append(Spacer(1, 0.3*inch))

    def add_section(self, heading, content=""):
        """Add a section with heading and optional content."""
        self.story.append(Paragraph(heading, self.styles['CustomHeading']))
        if content:
            self.story.append(Paragraph(content, self.styles['CustomBody']))
        self.story.append(Spacer(1, 0.15*inch))

    def add_metrics_table(self, metrics_dict):
        """Add a formatted metrics table."""
        data = [["Metric", "Value"]]
        for key, value in metrics_dict.items():
            data.append([str(key), str(value)])
        
        table = Table(data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))

    def add_bullet_points(self, points):
        """Add bullet point list."""
        for point in points:
            text = f"• {point}"
            self.story.append(Paragraph(text, self.styles['CustomBody']))
        self.story.append(Spacer(1, 0.15*inch))

    def add_paragraph(self, text):
        """Add a paragraph."""
        self.story.append(Paragraph(text, self.styles['CustomBody']))
        self.story.append(Spacer(1, 0.1*inch))

    def add_page_break(self):
        """Add page break."""
        self.story.append(PageBreak())

    def generate(self):
        """Generate PDF and return bytes."""
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        doc.build(self.story)
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()

    def save(self):
        """Save PDF to file."""
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        doc.build(self.story)
        
        with open(self.filename, 'wb') as f:
            f.write(pdf_buffer.getvalue())
        
        return self.filename


class ReportBuilder:
    """High-level interface to build complete analysis reports."""
    
    @staticmethod
    def build_analysis_report(df, profile, task_type, target_col, results, commentaries, recommendations):
        """Builds a complete analysis report."""
        reporter = PDFReporter()
        
        # --- TITLE PAGE ---
        reporter.add_title(
            "🤖 AI Data Science Analysis Report",
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        # --- EXECUTIVE SUMMARY ---
        reporter.add_section("Executive Summary")
        health = profile["Health Score"]
        health_label = "Excellent" if health >= 80 else "Good" if health >= 60 else "Fair"
        reporter.add_paragraph(
            f"This report presents an automated analysis of your dataset. "
            f"The dataset quality score is <b>{health}/100 ({health_label})</b>. "
            f"The identified task is <b>{task_type}</b> with target variable <b>{target_col}</b>."
        )
        
        # --- DATASET OVERVIEW ---
        reporter.add_page_break()
        reporter.add_section("1. Dataset Overview")
        
        dataset_metrics = {
            "Total Rows": profile["Total Rows"],
            "Total Columns": profile["Total Columns"],
            "Numeric Features": profile["Numerical Columns"],
            "Missing Values": profile["Missing Values"],
            "Missing %": f"{profile['Missing Pct']:.2f}%",
            "Dataset Personality": profile["Personality"],
        }
        reporter.add_metrics_table(dataset_metrics)
        
        # --- DATA QUALITY ---
        reporter.add_section("2. Data Quality Assessment")
        reporter.add_metrics_table({"Health Score": health, "Status": health_label})
        
        if profile["Issues"]:
            reporter.add_section("Issues Detected")
            reporter.add_bullet_points(profile["Issues"])
        else:
            reporter.add_paragraph("✅ <b>No major data quality issues detected.</b>")
        
        if profile["Recommendations"]:
            reporter.add_section("Recommendations")
            reporter.add_bullet_points(profile["Recommendations"])
        
        # --- AI COMMENTARY ---
        reporter.add_page_break()
        reporter.add_section("3. AI Analysis & Insights")
        for commentary in commentaries[:8]:  # Limit to 8 commentaries per page
            reporter.add_paragraph(commentary)
        
        # --- MODEL RESULTS ---
        reporter.add_page_break()
        reporter.add_section("4. Model Performance")
        
        if results:
            best_model = results[0]
            reporter.add_section("Best Model Selected")
            reporter.add_metrics_table({
                "Model Name": best_model['model_name'],
                "Metric": best_model['metric_name'],
                "Score": f"{best_model['score']:.4f}",
            })
            
            # Model leaderboard
            reporter.add_section("Model Leaderboard")
            leaderboard_data = [["Rank", "Model", f"{best_model['metric_name']}"]]
            for idx, result in enumerate(results, 1):
                leaderboard_data.append([
                    str(idx),
                    result['model_name'],
                    f"{result['score']:.4f}"
                ])
            
            table = Table(leaderboard_data, colWidths=[1*inch, 3*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            reporter.story.append(table)
            reporter.story.append(Spacer(1, 0.2*inch))
        
        # --- RECOMMENDATIONS & NEXT STEPS ---
        reporter.add_page_break()
        reporter.add_section("5. Recommendations & Next Steps")
        next_steps = [
            "Review the data quality assessment and address any flagged issues.",
            "Consider feature engineering to improve model performance.",
            "Validate the best model on a separate test set before deployment.",
            "Monitor model performance in production for concept drift.",
            "Iterate and refine based on business feedback and results.",
        ]
        reporter.add_bullet_points(next_steps)
        
        # --- FOOTER ---
        reporter.add_page_break()
        reporter.add_section("About This Report")
        reporter.add_paragraph(
            "This report was generated using <b>AI Data Science Assistant</b>, "
            "an automated machine learning and analytics platform. "
            "All analyses are based on heuristic-driven algorithms and statistical methods. "
            "Human validation is recommended before critical business decisions."
        )
        
        return reporter