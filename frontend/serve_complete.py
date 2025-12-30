#!/usr/bin/env python3
"""
E-Raksha Complete Website Server
Serves the comprehensive E-Raksha website with all features
"""

import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

class ErakshHandler(SimpleHTTPRequestHandler):
    """Custom handler for E-Raksha website"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Serve the main page for root requests
        if self.path == '/':
            self.path = '/eraksha-complete.html'
        return super().do_GET()

def main():
    """Start the E-Raksha website server"""
    port = 3001
    
    print("🚀 E-Raksha Complete Website")
    print("=" * 50)
    print(f"✅ Server starting on port {port}")
    print(f"🌐 URL: http://localhost:{port}")
    print(f"📁 Serving from: {Path(__file__).parent}")
    print("🎯 Features:")
    print("   • Complete UI matching the design")
    print("   • Home, Analysis, Dashboard, Analytics, Contact pages")
    print("   • Dark/Light theme toggle")
    print("   • Interactive file upload with drag & drop")
    print("   • Simulated analysis with progress tracking")
    print("   • Detailed results with model predictions")
    print("   • Analysis history and statistics")
    print("   • Charts and performance metrics")
    print("   • Responsive design with animations")
    print()
    
    try:
        # Create server
        server = HTTPServer(('localhost', port), ErakshHandler)
        
        # Open browser
        print("🚀 Opening browser...")
        webbrowser.open(f'http://localhost:{port}')
        
        print("Press Ctrl+C to stop the server")
        print()
        
        # Start server
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.server_close()
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()