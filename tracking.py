import sqlite3
import os
import argparse
from datetime import datetime

# Path to the SQLite database
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'usage_tracking.db')

def get_db_connection():
    # Set timeout just in case of concurrent writes
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # Create api_requests table
    c.execute('''
        CREATE TABLE IF NOT EXISTS api_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT,
            method TEXT,
            ip_address TEXT,
            user_agent TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            duration_ms REAL
        )
    ''')
    # Create jobs_tracking table
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs_tracking (
            job_id TEXT PRIMARY KEY,
            job_type TEXT,
            status TEXT,
            file_size_bytes INTEGER,
            subject_size INTEGER,
            object_size INTEGER,
            environment_size INTEGER,
            permit_rules_count INTEGER,
            deny_rules_count INTEGER,
            created_at DATETIME,
            updated_at DATETIME,
            completed_at DATETIME,
            error_message TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_api_request(endpoint, method, ip_address, user_agent, duration_ms):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO api_requests (endpoint, method, ip_address, user_agent, duration_ms)
            VALUES (?, ?, ?, ?, ?)
        ''', (endpoint, method, ip_address, user_agent, duration_ms))
        conn.commit()
    except Exception as e:
        print(f"Failed to log API request: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def log_job_creation(job_id, job_type="unknown"):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        now = datetime.now().isoformat()
        c.execute('''
            INSERT INTO jobs_tracking (job_id, job_type, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (job_id, job_type, 'PENDING', now, now))
        conn.commit()
    except Exception as e:
        print(f"Failed to log job creation: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def update_job_metrics(job_id, file_size_bytes=None, data=None):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        now = datetime.now().isoformat()
        
        subject_size = None
        object_size = None
        environment_size = None
        permit_rules_count = None
        deny_rules_count = None
        
        if data:
            subject_size = data.get('subject_size')
            object_size = data.get('object_size')
            environment_size = data.get('environment_size')
            permit_rules_count = data.get('permit_rules_count')
            deny_rules_count = data.get('deny_rules_count')
            
        c.execute('''
            UPDATE jobs_tracking 
            SET updated_at = ?,
                file_size_bytes = COALESCE(?, file_size_bytes),
                subject_size = COALESCE(?, subject_size),
                object_size = COALESCE(?, object_size),
                environment_size = COALESCE(?, environment_size),
                permit_rules_count = COALESCE(?, permit_rules_count),
                deny_rules_count = COALESCE(?, deny_rules_count)
            WHERE job_id = ?
        ''', (now, file_size_bytes, subject_size, object_size, environment_size, permit_rules_count, deny_rules_count, job_id))
        conn.commit()
    except Exception as e:
        print(f"Failed to update job metrics: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def update_job_status(job_id, status, error_message=None):
    try:
        conn = get_db_connection()
        c = conn.cursor()
        now = datetime.now().isoformat()
        
        completed_at = now if status in ('COMPLETED', 'FAILED') else None
        
        c.execute('''
            UPDATE jobs_tracking 
            SET status = ?,
                updated_at = ?,
                completed_at = COALESCE(?, completed_at),
                error_message = COALESCE(?, error_message)
            WHERE job_id = ?
        ''', (status, now, completed_at, error_message, job_id))
        conn.commit()
    except Exception as e:
        print(f"Failed to update job status: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def get_stats():
    conn = get_db_connection()
    c = conn.cursor()
    stats = {}
    
    # API Stats
    c.execute("SELECT COUNT(*) as count FROM api_requests")
    stats['total_api_requests'] = c.fetchone()['count']
    
    c.execute("SELECT endpoint, COUNT(*) as count FROM api_requests GROUP BY endpoint ORDER BY count DESC LIMIT 10")
    stats['top_endpoints'] = [dict(row) for row in c.fetchall()]
    
    # Job Stats
    c.execute("SELECT COUNT(*) as count FROM jobs_tracking")
    stats['total_jobs'] = c.fetchone()['count']
    
    c.execute("SELECT status, COUNT(*) as count FROM jobs_tracking GROUP BY status")
    stats['jobs_by_status'] = [dict(row) for row in c.fetchall()]
    
    c.execute("SELECT AVG(file_size_bytes) as avg_size_bytes, SUM(file_size_bytes) as total_size_bytes FROM jobs_tracking")
    size_stats = c.fetchone()
    stats['avg_job_file_size_bytes'] = size_stats['avg_size_bytes']
    stats['total_job_file_bytes'] = size_stats['total_size_bytes']
    
    # Rule metrics
    c.execute("SELECT AVG(permit_rules_count + deny_rules_count) as avg_total_rules FROM jobs_tracking WHERE permit_rules_count IS NOT NULL")
    rule_stats = c.fetchone()
    stats['avg_rules_per_job'] = rule_stats['avg_total_rules']
    
    conn.close()
    return stats

def cleanup_old_records(days=30):
    """Clean up tracking records older than specified days."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Calculate cutoff date
        c.execute("DELETE FROM api_requests WHERE timestamp < datetime('now', '-{} days')".format(days))
        c.execute("DELETE FROM jobs_tracking WHERE created_at < datetime('now', '-{} days')".format(days))
        
        conn.commit()
    except Exception as e:
        print(f"Failed to cleanup old tracking records: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="View ABAC Usage Tracking Statistics")
    parser.add_argument("command", choices=["stats", "init"], help="Command to run: 'stats' to view usage, 'init' to create DB")
    args = parser.parse_args()
    
    if args.command == "init":
        init_db()
        print(f"Initialized database at {DB_PATH}")
    elif args.command == "stats":
        if not os.path.exists(DB_PATH):
            print("Database does not exist yet. Run 'python tracking.py init' or start the app.")
            return
        
        stats = get_stats()
        print("=== ABAC System Usage Statistics ===")
        print(f"Total API Requests: {stats['total_api_requests']}")
        
        print(r"\\nTop Endpoints:")
        for ep in stats['top_endpoints']:
            print(f"  - {ep['endpoint']}: {ep['count']} requests")
            
        print(f"\\nTotal Jobs Processed: {stats['total_jobs']}")
        
        print("\\nJobs by Status:")
        for st in stats['jobs_by_status']:
            print(f"  - {st['status']}: {st['count']}")
            
        print("\\nStorage & Processing:")
        avg_size = stats['avg_job_file_size_bytes']
        total_size = stats['total_job_file_bytes']
        print(f"  - Average Input File Size: {int(avg_size) if avg_size else 0} bytes")
        print(f"  - Total Data Processed: {int(total_size) if total_size else 0} bytes")
        
        avg_rules = stats['avg_rules_per_job']
        if avg_rules is not None:
            print(f"  - Average Rules Generated Per Job: {int(avg_rules)}")

if __name__ == "__main__":
    main()
