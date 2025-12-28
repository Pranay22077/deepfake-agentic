#!/usr/bin/env python3
"""
Supabase Database Client
Handles database operations for E-Raksha
"""

import os
from supabase import create_client, Client
from typing import Dict, List, Optional
import json
from datetime import datetime

class SupabaseClient:
    """Supabase database client for E-Raksha"""
    
    def __init__(self):
        # Supabase credentials from environment or defaults
        self.url = os.getenv('SUPABASE_URL', 'https://rzgplzaytxronhcakemi.supabase.co')
        self.key = os.getenv('SUPABASE_ANON_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ6Z3BsemF5dHhyb25oY2FrZW1pIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjY4NzMxODAsImV4cCI6MjA4MjQ0OTE4MH0.6wxeRoi8ZnUmX4Vu86HKx4wWaiOJm_rPhXGrFKyEvyg')
        
        try:
            self.client: Client = create_client(self.url, self.key)
            print("✅ Supabase client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Supabase client: {e}")
            self.client = None
    
    def log_inference(self, video_filename: str, result: Dict, confidence: float, model_version: str = "kaggle-v1") -> bool:
        """Log inference result to database"""
        if not self.client:
            print("⚠️ Supabase client not available, skipping database log")
            return False
        
        try:
            data = {
                'video_path': video_filename,
                'result': json.dumps(result),
                'confidence': confidence,
                'model_version': model_version,
                'created_at': datetime.now().isoformat()
            }
            
            response = self.client.table('inference_logs').insert(data).execute()
            print(f"✅ Logged inference for {video_filename}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to log inference: {e}")
            return False
    
    def save_feedback(self, video_filename: str, user_label: str, user_confidence: float = None) -> bool:
        """Save user feedback to database"""
        if not self.client:
            print("⚠️ Supabase client not available, skipping feedback save")
            return False
        
        try:
            data = {
                'video_path': video_filename,
                'user_label': user_label.lower(),
                'user_confidence': user_confidence,
                'created_at': datetime.now().isoformat()
            }
            
            response = self.client.table('feedback_buffer').insert(data).execute()
            print(f"✅ Saved feedback for {video_filename}: {user_label}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save feedback: {e}")
            return False
    
    def get_inference_stats(self) -> Dict:
        """Get inference statistics"""
        if not self.client:
            return {"error": "Database not available"}
        
        try:
            # Get total inferences
            total_response = self.client.table('inference_logs').select('id', count='exact').execute()
            total_inferences = len(total_response.data) if total_response.data else 0
            
            # Get recent inferences (last 24 hours)
            recent_response = self.client.table('inference_logs').select('*').gte('created_at', 
                (datetime.now().replace(hour=0, minute=0, second=0)).isoformat()).execute()
            recent_inferences = len(recent_response.data) if recent_response.data else 0
            
            # Get feedback count
            feedback_response = self.client.table('feedback_buffer').select('id', count='exact').execute()
            total_feedback = len(feedback_response.data) if feedback_response.data else 0
            
            return {
                'total_inferences': total_inferences,
                'recent_inferences': recent_inferences,
                'total_feedback': total_feedback,
                'database_status': 'connected'
            }
            
        except Exception as e:
            print(f"❌ Failed to get stats: {e}")
            return {"error": str(e), "database_status": "error"}
    
    def create_tables(self) -> bool:
        """Create required tables (run this once)"""
        if not self.client:
            return False
        
        # Note: In production, create tables via Supabase dashboard
        # This is just for reference
        inference_logs_sql = """
        CREATE TABLE IF NOT EXISTS inference_logs (
            id SERIAL PRIMARY KEY,
            video_path TEXT NOT NULL,
            result JSONB NOT NULL,
            confidence REAL NOT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        feedback_buffer_sql = """
        CREATE TABLE IF NOT EXISTS feedback_buffer (
            id SERIAL PRIMARY KEY,
            video_path TEXT NOT NULL,
            user_label TEXT CHECK (user_label IN ('real', 'fake', 'unknown')),
            user_confidence REAL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        
        print("📋 SQL for creating tables:")
        print(inference_logs_sql)
        print(feedback_buffer_sql)
        print("Run these in your Supabase SQL editor")
        
        return True

# Global instance
supabase_client = SupabaseClient()