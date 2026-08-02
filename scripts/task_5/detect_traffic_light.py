#!/usr/bin/env python3
"""
Видео анализ с детекцией объектов через yolopy модель.
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

# ========== КОНФИГУРАЦИЯ ==========
SOURCE = 0  # веб-камера

# Resize frame to this resolution (width, height) - None to disable
TARGET_SIZE = (1280, 960)

# HTTP streaming config
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8089
JPEG_QUALITY = 70

MODEL_DIR = Path("/home/arrma/Computer_vision_in_navigation_of_unmanned_robotic_systems/scripts/models")

jpeg_lock = threading.Lock()
current_jpeg = None


def find_model_file(name):
    for d in [Path.cwd(), MODEL_DIR, Path('/home/avt_user/Base_Code'), Path('/home/avt_user/PROGRAMMS')]:
        p = d / name
        if p.exists():
            return p
    return None


class ObjectDetector:
    def __init__(self):
        self.model = None
        self.class_names = []

        if not YOLOPY_AVAILABLE:
            print("[INFO] yolopy not available, detector disabled")
            return

        classes_file = "/home/avt_user/PROGRAMMS/task_5/classes.txt"
        model_file = "/home/avt_user/PROGRAMMS/task_5/yolov4-tiny-original.tmfile"

        if not os.path.exists(classes_file) or not os.path.exists(model_file):
            print("[WARNING] Model files not found, detector disabled")
            return

        with open(classes_file) as f:
            self.class_names = f.read().splitlines()

        try:
            self.model = yolopy.Model(str(model_file), use_uint8=True, use_timvx=True, cls_num=2)
            self.model.set_anchors([18, 33, 33, 48, 25, 71, 58, 76, 40, 113, 87, 140])
            print(f"[INFO] Model loaded: {model_file}")
            print(f"[INFO] Classes: {self.class_names}")
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
                if score > 0.3:  # Детектируем все классы
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
            
            # Разные цвета для разных классов
            color = (0, 255, 0)  # зеленый по умолчанию
            if det['class'] == 0:
                color = (255, 0, 0)  # синий для класса 0
            elif det['class'] == 1:
                color = (0, 255, 0)  # зеленый для класса 1
            elif det['class'] == 2:
                color = (0, 0, 255)  # красный для класса 2
            
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            
            # Фон для текста
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (box[0], box[1] - text_h - 5), (box[0] + text_w, box[1]), color, -1)
            cv2.putText(frame, label, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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
    print(f'[INFO] Stream: http://{STREAM_HOST}:{STREAM_PORT}/')

    detector = ObjectDetector()

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

            # Детекция на всём кадре (без зелёных зон)
            detections = detector.detect(frame)
            frame = detector.draw_detections(frame, detections)

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