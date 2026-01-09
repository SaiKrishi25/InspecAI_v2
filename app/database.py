#!/usr/bin/env python3
"""
Database models and operations for InspecAI
SQLite database for storing detection history and analytics
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager
import os

DATABASE_PATH = os.environ.get("DATABASE_PATH", "app/data/inspecai.db")

def init_database():
    """Initialize database with schema"""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Detections table - stores each detection session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_id TEXT,
            total_defects INTEGER DEFAULT 0,
            detection_mode TEXT,
            report_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Defects table - stores individual defects from each detection
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS defects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id TEXT NOT NULL,
            defect_id TEXT NOT NULL,
            defect_type TEXT,
            confidence_score REAL,
            bbox_x1 INTEGER,
            bbox_y1 INTEGER,
            bbox_x2 INTEGER,
            bbox_y2 INTEGER,
            location TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (detection_id) REFERENCES detections(detection_id)
        )
    """)
    
    # Reports table - stores PDF report metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE NOT NULL,
            detection_id TEXT NOT NULL,
            pdf_path TEXT,
            json_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (detection_id) REFERENCES detections(detection_id)
        )
    """)
    
    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_defects_detection_id ON defects(detection_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_defects_type ON defects(defect_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reports_detection_id ON reports(detection_id)")
    
    conn.commit()
    conn.close()
    
    print(f"[Database] Initialized at {DATABASE_PATH}")

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

class DetectionStore:
    """Database operations for detections"""
    
    @staticmethod
    def insert_detection(detection_data: Dict) -> str:
        """Insert a new detection record"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (detection_id, timestamp, image_id, total_defects, detection_mode, report_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                detection_data['detection_id'],
                detection_data.get('timestamp', datetime.now().isoformat()),
                detection_data.get('image_id'),
                detection_data.get('total_defects', 0),
                detection_data.get('detection_mode', 'fasterrcnn_only'),
                detection_data.get('report_id')
            ))
            
            # Insert individual defects
            if 'defects' in detection_data:
                for defect in detection_data['defects']:
                    bbox = defect.get('bbox', [0, 0, 0, 0])
                    cursor.execute("""
                        INSERT INTO defects (detection_id, defect_id, defect_type, confidence_score, 
                                           bbox_x1, bbox_y1, bbox_x2, bbox_y2, location)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        detection_data['detection_id'],
                        defect.get('id', ''),
                        defect.get('class', 'Unknown'),
                        defect.get('score', 0.0),
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        defect.get('location', 'unknown')
                    ))
            
            return detection_data['detection_id']
    
    @staticmethod
    def get_recent_detections(limit: int = 10) -> List[Dict]:
        """Get recent detections"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM detections 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    @staticmethod
    def get_detections_by_date_range(start_date: str, end_date: str) -> List[Dict]:
        """Get detections within date range"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM detections 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (start_date, end_date))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    @staticmethod
    def get_detection_by_id(detection_id: str) -> Optional[Dict]:
        """Get a single detection by ID"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM detections WHERE detection_id = ?", (detection_id,))
            row = cursor.fetchone()
            
            if row:
                detection = dict(row)
                # Get associated defects
                cursor.execute("SELECT * FROM defects WHERE detection_id = ?", (detection_id,))
                defect_rows = cursor.fetchall()
                detection['defects'] = [dict(d) for d in defect_rows]
                return detection
            return None
    
    @staticmethod
    def get_analytics_overview(days: int = 30) -> Dict:
        """Get analytics overview for dashboard"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total detections
            cursor.execute("SELECT COUNT(*) as total FROM detections")
            total_detections = cursor.fetchone()['total']
            
            # Detections today
            cursor.execute("""
                SELECT COUNT(*) as count FROM detections 
                WHERE DATE(timestamp) = DATE('now')
            """)
            today_detections = cursor.fetchone()['count']
            
            # Detections this week
            cursor.execute("""
                SELECT COUNT(*) as count FROM detections 
                WHERE timestamp >= DATE('now', '-7 days')
            """)
            week_detections = cursor.fetchone()['count']
            
            # Detections this month
            cursor.execute("""
                SELECT COUNT(*) as count FROM detections 
                WHERE timestamp >= DATE('now', '-30 days')
            """)
            month_detections = cursor.fetchone()['count']
            
            # Total defects found
            cursor.execute("SELECT COUNT(*) as total FROM defects")
            total_defects = cursor.fetchone()['total']
            
            # Defects by type
            cursor.execute("""
                SELECT defect_type, COUNT(*) as count 
                FROM defects 
                GROUP BY defect_type
            """)
            defects_by_type = {row['defect_type']: row['count'] for row in cursor.fetchall()}
            
            # Average confidence score
            cursor.execute("SELECT AVG(confidence_score) as avg_score FROM defects")
            avg_confidence = cursor.fetchone()['avg_score'] or 0
            
            return {
                'total_detections': total_detections,
                'today_detections': today_detections,
                'week_detections': week_detections,
                'month_detections': month_detections,
                'total_defects': total_defects,
                'defects_by_type': defects_by_type,
                'avg_confidence': round(avg_confidence, 2)
            }
    
    @staticmethod
    def get_time_series_data(days: int = 30) -> List[Dict]:
        """Get time series data for charts"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as detection_count,
                    SUM(total_defects) as defect_count
                FROM detections
                WHERE timestamp >= DATE('now', ?)
                GROUP BY DATE(timestamp)
                ORDER BY date ASC
            """, (f'-{days} days',))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    @staticmethod
    def get_defect_distribution() -> List[Dict]:
        """Get defect type distribution for pie/bar charts"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    defect_type,
                    COUNT(*) as count,
                    AVG(confidence_score) as avg_confidence
                FROM defects
                GROUP BY defect_type
                ORDER BY count DESC
            """)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

class ReportStore:
    """Database operations for reports"""
    
    @staticmethod
    def insert_report(report_data: Dict) -> str:
        """Insert a new report record"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO reports (report_id, detection_id, pdf_path, json_data)
                VALUES (?, ?, ?, ?)
            """, (
                report_data['report_id'],
                report_data['detection_id'],
                report_data.get('pdf_path'),
                json.dumps(report_data.get('json_data', {}))
            ))
            
            return report_data['report_id']
    
    @staticmethod
    def get_all_reports(limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all reports with pagination"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.*, d.timestamp, d.total_defects, d.image_id
                FROM reports r
                LEFT JOIN detections d ON r.detection_id = d.detection_id
                ORDER BY r.created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    @staticmethod
    def get_report_by_id(report_id: str) -> Optional[Dict]:
        """Get a single report by ID"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.*, d.timestamp, d.total_defects, d.image_id
                FROM reports r
                LEFT JOIN detections d ON r.detection_id = d.detection_id
                WHERE r.report_id = ?
            """, (report_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    @staticmethod
    def search_reports(search_term: str = '', start_date: str = '', end_date: str = '') -> List[Dict]:
        """Search reports with filters"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT r.*, d.timestamp, d.total_defects, d.image_id
                FROM reports r
                LEFT JOIN detections d ON r.detection_id = d.detection_id
                WHERE 1=1
            """
            params = []
            
            if search_term:
                query += " AND (r.report_id LIKE ? OR d.image_id LIKE ?)"
                params.extend([f'%{search_term}%', f'%{search_term}%'])
            
            if start_date and end_date:
                query += " AND d.timestamp BETWEEN ? AND ?"
                params.extend([start_date, end_date])
            
            query += " ORDER BY r.created_at DESC LIMIT 100"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

# Initialize database on import
init_database()

