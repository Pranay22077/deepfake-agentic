#!/usr/bin/env python3
"""
Simple HTTP server for the agentic frontend
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    PORT = 3001
    
    print(f"🌐 Starting E-Raksha Agentic Frontend Server")
    print(f"📁 Serving from: {Path(__file__).parent}")
    print(f"🔗 URL: http://localhost:{PORT}")
    print(f"📄 Main page: http://localhost:{PORT}/agentic-index.html")
    print("=" * 60)
    
    try:
        with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
            print(f"✅ Server running on port {PORT}")
            print("🚀 Opening browser...")
            
            # Open browser
            webbrowser.open(f'http://localhost:{PORT}/agentic-index.html')
            
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()