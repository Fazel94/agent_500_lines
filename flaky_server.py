import http.server, json

N = [0]
SSE = [
    'data: {"choices":[{"delta":{"content":"hello "}}]}',
    'data: {"choices":[{"delta":{"content":"from the flaky server"}}]}',
    "data: [DONE]",
]


class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        N[0] += 1
        if N[0] == 1:
            self.send_response(429)
            self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(b'{"error":"slow down"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for line in SSE:
            self.wfile.write(line.encode() + b"\n\n")
            self.wfile.flush()

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", 8931), H).serve_forever()
