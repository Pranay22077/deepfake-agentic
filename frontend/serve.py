#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend
"""

import http.server
import socketserver
import os
import webbrowser

PORT = 3000

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

def main():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Frontend server running at http://localhost:{PORT}")
        print("Opening browser...")
        
        # Open browser
        webbrowser.open(f"http://localhost:{PORT}")
        
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")

if __name__ == "__main__":
    main()