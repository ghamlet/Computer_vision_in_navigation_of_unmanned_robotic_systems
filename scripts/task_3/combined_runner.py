#!/usr/bin/env python3
"""
Объединённый скрипт: движение по разметке + детекция светофора.
При потере трека светофора -> STATE = STOP.
Стримит результат в браузер по MJPEG.
"""

import atexit
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from Arduino_A26 import Arduino
from road_utils import *

try:
    import yolopy
    YOLOPY_AVAILABLE = True
except ImportError:
    YOLOPY_AVAILABLE = False
    print("[WARNING] yolopy not available, running without model detection")

# ========== КОНФИГУРАЦИЯ ==========
# Движение
CAR_SPEED = 1600
THRESHOLD = 230
CAMERA_ID = '/dev/video0'
ARDUINO_PORT = '/dev/ttyUSB0'

# Детекция светофора
TRAFFIC_LIGHT_CLASS = 9
TARGET_SIZE = (1280, 960)

TRACK_MAX_MISSED = 15
TRACK_MAX_DIST = 200
TRACK_CONFIRM_FRAMES = 3

# Зеленые зоны
TOP_FILL_H = 100
BOTTOM_FILL_Y = 640
FILL_X = 870
FILL_COLOR = (0, 255, 0)

# Состояния
GO = 'GO'
STOP = 'STOP'
STATE = GO
PREV_STATE = None

# HTTP streaming
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8088
JPEG_QUALITY = 70

MODEL_DIR = Path("/home/arrma/Computer_vision_in_navigation_of_unmanned_robotic_systems/scripts/models")

jpeg_lock = threading.Lock()
current_jpeg = None

arduino = None


def fill_zones(frame):
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


@atexit.register
def exit_func(*args):
    global arduino
    if arduino is not None:
        try:
            arduino.close()
        finally:
            arduino = None


def main():
    global STATE, PREV_STATE, arduino

    # Arduino
    arduino = Arduino(ARDUINO_PORT)
    print("Arduino connected")

    # Camera
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    if not cap.isOpened():
        print('[ERROR] Cannot open camera ID:', CAMERA_ID)
        return 1

    # Streaming server
    server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
    stream_thread = threading.Thread(target=server.serve_forever, daemon=True)
    stream_thread.start()
    print(f'[INFO] Streaming on http://{STREAM_HOST}:{STREAM_PORT}')

    # Detector
    detector = TrafficLightDetector()

    # Tracking state
    track_state = {
        'x': None, 'y': None,
        'missed': 0,
        'confirmed': 0,
        'lost_printed': False
    }

    find_lines = centre_mass2
    last_err = 0

    # Warmup frames
    for i in range(30):
        cap.read()

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # --- Подготовка кадра для детекции светофора (полный кадр) ---
            det_frame = frame.copy()
            if TARGET_SIZE:
                det_frame = cv2.resize(det_frame, TARGET_SIZE)
            fill_zones(det_frame)

            # --- Детекция светофора ---
            detections = detector.detect(det_frame)
            det_frame = detector.draw_detections(det_frame, detections)

            # --- Треккинг центра bbox ---
            centers = []
            for det in detections:
                box = det['box']
                cx = box[0] + box[2] // 2
                cy = box[1] + box[3] // 2
                centers.append((cx, cy))

            if centers:
                if track_state['x'] is None:
                    track_state['x'], track_state['y'] = centers[0]
                    track_state['missed'] = 0
                    track_state['confirmed'] = 1
                    track_state['lost_printed'] = False
                else:
                    distances = [((cx - track_state['x'])**2 + (cy - track_state['y'])**2)**0.5 for cx, cy in centers]
                    idx = int(np.argmin(distances))
                    if distances[idx] <= TRACK_MAX_DIST:
                        track_state['x'], track_state['y'] = centers[idx]
                        track_state['missed'] = 0
                        track_state['confirmed'] = min(track_state['confirmed'] + 1, TRACK_CONFIRM_FRAMES + 10)
                    else:
                        track_state['missed'] += 1
            else:
                track_state['missed'] += 1

            # Отрисовка трека на кадре для стрима
            if track_state['confirmed'] >= TRACK_CONFIRM_FRAMES and track_state['x'] is not None:
                cv2.drawMarker(det_frame, (int(track_state['x']), int(track_state['y'])), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(det_frame, (int(track_state['x']), int(track_state['y'])), 6, (0, 0, 255), -1)
                cv2.putText(det_frame, 'TRACKED', (int(track_state['x'])+10, int(track_state['y'])-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Проверка потери трека -> STOP
            if track_state['missed'] > TRACK_MAX_MISSED and track_state['confirmed'] >= TRACK_CONFIRM_FRAMES and not track_state['lost_printed']:
                print('СТОП')
                STATE = STOP
                track_state['lost_printed'] = True

            # Сброс трека при длительной потере
            if track_state['missed'] > TRACK_MAX_MISSED * 2:
                track_state.update(x=None, y=None, missed=0, confirmed=0, lost_printed=False)

            # --- Движение по разметке (нижняя часть кадра) ---
            if STATE != STOP:
                # Обрезка для поиска линий (как в StadionRunner)
                line_frame = frame[-720:, :]
                line_frame = cv2.resize(line_frame, SIZE)
                gray = cv2.cvtColor(line_frame, cv2.COLOR_BGR2GRAY)
                bin = cv2.inRange(gray, THRESHOLD, 255)

                wrapped = trans_perspective(bin, TRAP, RECT, SIZE)
                left, right = find_lines(wrapped)

                err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)
                err = -err
                angle = int(90 + KP * err + KD * (err - last_err))
                last_err = err
                angle = min(max(45, angle), 135)

                arduino.set_speed(CAR_SPEED)
                arduino.set_angle(angle)
            else:
                arduino.set_speed(1500)  # Стоп

            # Лог смены состояния
            if PREV_STATE != STATE:
                print(f'STATE: {STATE}')
                PREV_STATE = STATE

            # Стрим кадра с детекцией
            ok, jpg = cv2.imencode('.jpg', det_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with jpeg_lock:
                    global current_jpeg
                    current_jpeg = jpg.tobytes()

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            if fps < 10:
                print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')

    except KeyboardInterrupt:
        print('\n[INFO] KeyboardInterrupt received, stopping...')
    finally:
        if arduino is not None:
            arduino.close()
            arduino = None
        server.shutdown()
        server.server_close()
        cap.release()
    return 0


if __name__ == '__main__':
    sys.exit(main())