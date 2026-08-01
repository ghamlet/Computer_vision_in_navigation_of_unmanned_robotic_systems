#!/usr/bin/env python3
"""MJPEG stream sender. Run on the robot (Eyecar).

Streams video from CAMERA_ID over HTTP so that any browser or the
receiver script (video_receiver.py) can fetch frames.

Usage on robot:  python3 video_sender.py [port]
Open in browser: http://<robot-ip>:8080
"""

import sys
import threading

import cv2

CAMERA_ID = '/dev/video0'
HOST = '0.0.0.0'
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
JPEG_QUALITY = 70

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    sys.exit(1)

jpeg_lock = threading.Lock()
current_jpeg = None


def grab_loop():
    global current_jpeg
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with jpeg_lock:
                current_jpeg = buf.tobytes()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/':
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        while True:
            with jpeg_lock:
                frame = current_jpeg
            if frame is None:
                continue
            try:
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError, OSError):
                break

    def log_message(self, *args):
        pass


def main():
    t = threading.Thread(target=grab_loop, daemon=True)
    t.start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f'[INFO] Streaming {CAMERA_ID} on http://{HOST}:{PORT}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[INFO] Stopping...')
    finally:
        server.server_close()
        cap.release()


if __name__ == '__main__':
    main()
