#!/usr/bin/env python3
import http.server
import socket
import socketserver
import threading
import webbrowser
import os


def find_free_port(preferred: int = 8000) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", preferred))
            return preferred
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def main() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    os.chdir(here)

    port = find_free_port(8000)
    handler = http.server.SimpleHTTPRequestHandler

    # Threaded server to avoid blocking
    class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True
        allow_reuse_address = True

    with ThreadingTCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/"
        print(f"Serving HIL GUI at {url}")
        # Open the browser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    main()


