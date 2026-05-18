"""
SQLite-based experiment history tracking.
Stores model results for future reference and comparison.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

class ExperimentTracker:
    """Manages experiment history in SQLite database."""
    
    DB_PATH = "experiment_history.db"
    
    def __init__(self):
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with schema."""
        self.conn = sqlite3.connect(self.DB_PATH)
        cursor = self.conn.cursor()
        
        # Create experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                task_type TEXT NOT NULL,
                target_column TEXT NOT NULL,
                dataset_rows INTEGER,
                dataset_cols INTEGER,
                health_score INTEGER,
                best_model TEXT NOT NULL,
                best_metric TEXT NOT NULL,
                best_score REAL NOT NULL,
                all_results TEXT NOT NULL,
                notes TEXT
            )
        """)
        
        self.conn.commit()
    
    def log_experiment(self, dataset_name, task_type, target_col, 
                       df, profile, results, notes=""):
        """Log an experiment to the database."""
        cursor = self.conn.cursor()
        
        best_model = results[0] if results else None
        
        # Serialize all results as JSON
        results_json = json.dumps([
            {
                "model": r['model_name'],
                "metric": r['metric_name'],
                "score": float(r['score'])
            }
            for r in results
        ])
        
        cursor.execute("""
            INSERT INTO experiments 
            (timestamp, dataset_name, task_type, target_column, 
             dataset_rows, dataset_cols, health_score, 
             best_model, best_metric, best_score, all_results, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            dataset_name,
            task_type,
            target_col,
            len(df),
            len(df.columns),
            profile["Health Score"],
            best_model['model_name'] if best_model else "N/A",
            best_model['metric_name'] if best_model else "N/A",
            best_model['score'] if best_model else 0.0,
            results_json,
            notes
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_history(self, limit=10):
        """Retrieve recent experiments."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, dataset_name, task_type, best_model, 
                   best_metric, best_score, health_score
            FROM experiments
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_experiment_detail(self, exp_id):
        """Retrieve full details of a specific experiment."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM experiments WHERE id = ?
        """, (exp_id,))
        
        columns = [desc[0] for desc in cursor.description]
        row = cursor.fetchone()
        
        if row:
            exp = dict(zip(columns, row))
            exp['all_results'] = json.loads(exp['all_results'])
            return exp
        return None
    
    def get_dataset_experiments(self, dataset_name):
        """Get all experiments for a specific dataset."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT timestamp, task_type, target_column, best_model, 
                   best_score, health_score
            FROM experiments
            WHERE dataset_name = ?
            ORDER BY timestamp DESC
        """, (dataset_name,))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def get_model_stats(self):
        """Get statistics about models used."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT best_model, COUNT(*) as count, AVG(best_score) as avg_score
            FROM experiments
            GROUP BY best_model
            ORDER BY count DESC
        """)
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        return [dict(zip(columns, row)) for row in rows]
    
    def delete_experiment(self, exp_id):
        """Delete an experiment from history."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
        self.conn.commit()
    
    def clear_history(self):
        """Clear all experiments (use with caution)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM experiments")
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()