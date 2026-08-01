#!/usr/bin/env python3
"""
Lane detection from StadionRunner.py (Bazovyi_774_kod) without yolo detection and Arduino.
Uses local camera by index, processes frames, streams binarized result with lines.
"""

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

from road_utils import *

# ========== CONFIG ==========
CAMERA_INDEX = 0  # камера по индексу (0, 1, 2...)

# Local streaming config (processed frames)
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8088
JPEG_QUALITY = 70

# Lane detection config (from original StadionRunner.py)
THRESHOLD = 200

jpeg_lock = threading.Lock()
current_jpeg = None


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != '/':
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
    global current_jpeg

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    if not cap.isOpened():
        print(f'[ERROR] Cannot open camera index {CAMERA_INDEX}')
        return 1

    print(f'[INFO] Camera {CAMERA_INDEX} opened: 1280x960')

    # Start local streaming server
    server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
    stream_thread = threading.Thread(target=server.serve_forever, daemon=True)
    stream_thread.start()
    print(f'[INFO] Streaming processed frames on http://{STREAM_HOST}:{STREAM_PORT}')

    find_lines = centre_mass2
    last_err = 0

    # Warmup
    for i in range(30):
        cap.read()

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # --- Lane detection (exactly like original StadionRunner.py) ---
            frame = frame[-720:, :]  # для поиска разметки весь кадр не нужен
            orig_frame = frame.copy()
            frame = cv2.resize(frame, SIZE)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bin = cv2.inRange(gray, THRESHOLD, 255)

            wrapped = trans_perspective(bin, TRAP, RECT, SIZE)
            left, right = find_lines(wrapped)

            # ПИД-регулятор
            err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)
            err = -err
            angle = int(90 + KP * err + KD * (err - last_err))
            last_err = err

            angle = min(max(45, angle), 135)

            # Visualize for streaming
            vis = cv2.cvtColor(bin, cv2.COLOR_GRAY2BGR)
            cv2.line(vis, (left, 0), (left, vis.shape[0]), (0, 255, 0), 2)
            cv2.line(vis, (right, 0), (right, vis.shape[0]), (0, 255, 0), 2)
            center_x = (left + right) // 2
            cv2.line(vis, (center_x, 0), (center_x, vis.shape[0]), (255, 0, 0), 2)
            cv2.putText(vis, f'angle: {angle}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f'left: {left}, right: {right}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis, f'err: {err}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Encode for streaming
            ok, jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with jpeg_lock:
                    current_jpeg = jpg.tobytes()

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            if fps < 10:
                print(f'[WARNING] FPS too low: {fps:.1f}')

    except KeyboardInterrupt:
        print('\n[INFO] Stopping...')
    finally:
        cap.release()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            CAMERA_INDEX = int(sys.argv[1])
        except ValueError:
            CAMERA_INDEX = sys.argv[1]  # allow device path like /dev/video1
    sys.exit(main())