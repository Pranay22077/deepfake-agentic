#!/usr/bin/env python3
"""
Setup Supabase database tables for E-Raksha
Run this once to create the required tables
"""

from backend.db.supabase_client import supabase_client

def setup_database():
    """Setup database tables"""
    print("[SETUP] Setting up E-Raksha Database")
    print("=" * 50)
    
    # Create tables (you need to run these SQL commands in Supabase dashboard)
    sql_commands = [
        """
        -- Inference logs table
        CREATE TABLE IF NOT EXISTS inference_logs (
            id SERIAL PRIMARY KEY,
            video_path TEXT NOT NULL,
            result JSONB NOT NULL,
            confidence REAL NOT NULL,
            model_version TEXT NOT NULL DEFAULT 'kaggle-v1',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,
        """
        -- User feedback table
        CREATE TABLE IF NOT EXISTS feedback_buffer (
            id SERIAL PRIMARY KEY,
            video_path TEXT NOT NULL,
            user_label TEXT CHECK (user_label IN ('real', 'fake', 'unknown')),
            user_confidence REAL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,
        """
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at ON inference_logs(created_at);
        CREATE INDEX IF NOT EXISTS idx_inference_logs_model_version ON inference_logs(model_version);
        CREATE INDEX IF NOT EXISTS idx_feedback_buffer_created_at ON feedback_buffer(created_at);
        CREATE INDEX IF NOT EXISTS idx_feedback_buffer_user_label ON feedback_buffer(user_label);
        """
    ]
    
    print("[INFO] SQL Commands to run in Supabase SQL Editor:")
    print("=" * 50)
    
    for i, sql in enumerate(sql_commands, 1):
        print(f"\n-- Command {i}:")
        print(sql.strip())
    
    print("\n" + "=" * 50)
    print("[LINK] Supabase Dashboard: https://rzgplzaytxronhcakemi.supabase.co")
    print("1. Go to SQL Editor")
    print("2. Copy and paste each SQL command above")
    print("3. Run them one by one")
    print("=" * 50)
    
    # Test database connection
    print("\n[TEST] Testing database connection...")
    stats = supabase_client.get_inference_stats()
    
    if 'error' not in stats:
        print("[OK] Database connection successful!")
        print(f"Current stats: {stats}")
    else:
        print(f"[ERROR] Database connection failed: {stats}")
    
    return True

if __name__ == "__main__":
    setup_database()
