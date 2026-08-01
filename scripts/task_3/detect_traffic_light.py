#!/usr/bin/env python3
"""
Видео анализ с детекцией светофора через yolopy модель.
Стримит результат в браузер по MJPEG.
"""

import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

try:
    import yolopy
    YOLOPY_AVAILABLE = True
except ImportError:
    YOLOPY_AVAILABLE = False
    print("[WARNING] yolopy not available, running without model detection")

# ========== КОНФИГУРАЦИЯ (измени здесь) ==========
# Источник видео: путь к файлу или индекс камеры (0, 1, 2...)
# SOURCE = "/home/avt_user/PROGRAMMS/task_3/received_20260731_172046.avi"
SOURCE = 0  # веб-камера

# Зеленые зоны (как в analyze_video.py)
TOP_FILL_H = 100
BOTTOM_FILL_Y = 640
FILL_X = 870
FILL_COLOR = (0, 255, 0)

# Класс светофора в classes.txt (индекс 9)
TRAFFIC_LIGHT_CLASS = 9

# Resize frame to this resolution (width, height) - None to disable
TARGET_SIZE = (1280, 960)

# Треккинг центра bbox
TRACK_MAX_MISSED = 5      # сколько кадров без детекции терпеть
TRACK_MAX_DIST = 100      # макс. смещение центра между кадрами (px)
TRACK_CONFIRM_FRAMES = 3  # кадров для подтверждения трека

# HTTP streaming config
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8089
JPEG_QUALITY = 70

MODEL_DIR = Path("/home/arrma/Computer_vision_in_navigation_of_unmanned_robotic_systems/scripts/models")

jpeg_lock = threading.Lock()
current_jpeg = None


def fill_zones(frame):
    """Закрашивает зеленым: верх 100px, низ от 640px, лево до 870px"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, min(TOP_FILL_H, h)), FILL_COLOR, -1)
    cv2.rectangle(frame, (0, min(BOTTOM_FILL_Y, h)), (w, h), FILL_COLOR, -1)
    cv2.rectangle(frame, (0, 0), (min(FILL_X, w), h), FILL_COLOR, -1)


def find_model_file(name):
    for d in [Path.cwd(), MODEL_DIR, Path('/home/avt_user/Base_Code'), Path('/home/avt_user/PROGRAMMS')]:
        p = d / name
        if p.exists():
            return p
    return None


class TrafficLightDetector:
    def __init__(self, detect_class=TRAFFIC_LIGHT_CLASS):
        self.detect_class = detect_class
        self.model = None
        self.class_names = []

        if not YOLOPY_AVAILABLE:
            print("[INFO] yolopy not available, detector disabled")
            return

        classes_file = find_model_file('classes.txt')
        model_file = find_model_file('yolo_uint8.tmfile')

        if classes_file is None or model_file is None:
            print("[WARNING] Model files not found, detector disabled")
            return

        with open(classes_file) as f:
            self.class_names = f.read().splitlines()

        try:
            self.model = yolopy.Model(str(model_file), use_uint8=True, use_timvx=True, cls_num=10)
            self.model.set_anchors([18, 33, 33, 48, 25, 71, 58, 76, 40, 113, 87, 140])
            print(f"[INFO] Model loaded: {model_file}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self.model = None

    def detect(self, frame):
        if self.model is None:
            return []

        try:
            classes, scores, boxes = self.model.detect(frame)
            results = []
            for classid, score, box in zip(classes, scores, boxes):
                if classid == self.detect_class and score > 0.3:
                    results.append({
                        'class': classid,
                        'score': score,
                        'box': box,
                        'label': self.class_names[classid] if classid < len(self.class_names) else f'class_{classid}'
                    })
            return results
        except Exception as e:
            print(f"[ERROR] Detection failed: {e}")
            return []

    def draw_detections(self, frame, detections):
        for det in detections:
            box = det['box']
            label = f"{det['label']} [{det['score']*100:.1f}%]"
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


class StreamHandler(BaseHTTPRequestHandler):
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
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video source: {SOURCE}')
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] Source: {SOURCE}')
    print(f'[INFO] Resolution: {width}x{height}')
    print(f'[INFO] Green zones: top {TOP_FILL_H}px, bottom from {BOTTOM_FILL_Y}px, left {FILL_X}px')
    print(f'[INFO] Stream: http://{STREAM_HOST}:{STREAM_PORT}/')

    detector = TrafficLightDetector()

    # Tracking state
    track_state = {
        'x': None, 'y': None,
        'missed': 0,
        'confirmed': 0,
        'lost_printed': False
    }

    server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
    stream_thread = threading.Thread(target=server.serve_forever, daemon=True)
    stream_thread.start()
    print("[INFO] Streaming server started")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(SOURCE, str) and SOURCE.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Resize to target size
            if TARGET_SIZE:
                frame = cv2.resize(frame, TARGET_SIZE)

            # Наложение зеленых зон (НЕ обрезаем кадр)
            fill_zones(frame)

            detections = detector.detect(frame)
            frame = detector.draw_detections(frame, detections)

            # --- Tracking logic ---
            centers = []
            for det in detections:
                box = det['box']
                cx = box[0] + box[2] // 2
                cy = box[1] + box[3] // 2
                centers.append((cx, cy))

            if centers:
                if track_state['x'] is None:
                    # No track yet - start with first detection
                    track_state['x'], track_state['y'] = centers[0]
                    track_state['missed'] = 0
                    track_state['confirmed'] = 1
                    track_state['lost_printed'] = False
                else:
                    # Find closest detection to current track
                    distances = [((cx - track_state['x'])**2 + (cy - track_state['y'])**2)**0.5 for cx, cy in centers]
                    idx = int(np.argmin(distances))
                    if distances[idx] <= TRACK_MAX_DIST:
                        # Detection matches track
                        track_state['x'], track_state['y'] = centers[idx]
                        track_state['missed'] = 0
                        track_state['confirmed'] = min(track_state['confirmed'] + 1, TRACK_CONFIRM_FRAMES + 10)
                    else:
                        # Detection too far - treat as missed
                        track_state['missed'] += 1
            else:
                # No detections this frame
                track_state['missed'] += 1

            # Draw tracked center if confirmed
            if track_state['confirmed'] >= TRACK_CONFIRM_FRAMES and track_state['x'] is not None:
                cv2.drawMarker(frame, (int(track_state['x']), int(track_state['y'])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(frame, (int(track_state['x']), int(track_state['y'])), 6, (0, 0, 255), -1)
                cv2.putText(frame, 'TRACKED', (int(track_state['x'])+10, int(track_state['y'])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Check if track is completely lost
            if track_state['missed'] > TRACK_MAX_MISSED and track_state['confirmed'] >= TRACK_CONFIRM_FRAMES and not track_state['lost_printed']:
                print('СТОП')
                track_state['lost_printed'] = True

            # Reset track if lost for too long
            if track_state['missed'] > TRACK_MAX_MISSED * 2:
                track_state.update(x=None, y=None, missed=0, confirmed=0, lost_printed=False)

            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with jpeg_lock:
                    global current_jpeg
                    current_jpeg = jpg.tobytes()

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        cap.release()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())