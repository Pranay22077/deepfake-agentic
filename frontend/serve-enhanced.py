#!/usr/bin/env python3
"""
Enhanced Frontend Server
Serves the enhanced E-Raksha frontend with proper MIME types
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class EnhancedHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Enhanced HTTP handler with proper MIME types"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def guess_type(self, path):
        """Guess MIME type with enhanced mappings"""
        try:
            mimetype, encoding = super().guess_type(path)
        except (ValueError, TypeError):
            mimetype, encoding = None, None
        
        # Enhanced MIME type mappings
        if path.endswith('.js'):
            return 'application/javascript'
        elif path.endswith('.css'):
            return 'text/css'
        elif path.endswith('.html'):
            return 'text/html'
        elif path.endswith('.json'):
            return 'application/json'
        
        return mimetype or 'text/plain'
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.path = '/enhanced-index.html'
        return super().do_GET()

def main():
    """Start the enhanced frontend server"""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3001
    
    print(f"🌐 Starting E-Raksha Enhanced Frontend Server")
    print(f"📍 Port: {port}")
    print(f"📂 Directory: {Path(__file__).parent}")
    print(f"🔗 URL: http://localhost:{port}")
    print(f"📱 Main App: http://localhost:{port}/enhanced-index.html")
    print("=" * 50)
    
    try:
        with socketserver.TCPServer(("", port), EnhancedHTTPRequestHandler) as httpd:
            print(f"✅ Server running on port {port}")
            print("🛑 Press Ctrl+C to stop")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()