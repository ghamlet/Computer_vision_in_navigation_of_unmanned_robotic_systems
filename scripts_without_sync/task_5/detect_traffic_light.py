#!/usr/bin/env python3
"""Детекция светофора и его состояния (working / notworking) через OpenCV DNN.

Двухэтапная логика:
  1. Сначала ищем светофор: объединяем ВСЕ детекции (working + notworking)
     в одну общую рамку (union bbox) - это и есть "светофор".
  2. Затем внутри рамки светофора детектим состояние (working / notworking):
       - центр bbox каждого объекта должен лежать внутри bbox светофора;
       - класс подтверждается только после НЕСКОЛЬКИХ кадров-детекций
         (CONFIRM_FRAMES подряд).
  Итоговый класс выводится в консоль и рисуется на кадре.

Стриминг в браузер по MJPEG, ОДИН топик: http://<host>:<port>/stream
"""

import argparse
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')

CFG = os.path.join(MODEL_DIR, 'yolov4-tiny-svetofor.cfg')
WEIGHTS = os.path.join(MODEL_DIR, 'yolov4-tiny-svetofor_best_weights.weights')
NAMES = os.path.join(MODEL_DIR, 'svetofor.names')

INPUT_SIZE = 416

# ========== КОНФИГУРАЦИЯ ==========
SOURCE = 'auto'                 # путь к видео, индекс камеры или 'auto' (первое видео из ../records)
CONF_THRESHOLD = 0.3            # порог уверенности для детекций
NMS_THRESHOLD = 0.4             # порог NMS
CONFIRM_FRAMES = 5              # сколько кадров-детекций нужно для подтверждения класса
MAX_MISSED = 10                 # сколько кадров без детекций сбрасывают голосование
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8089
STREAM_TOPIC = '/stream'        # единственный топик для стриминга в браузер
JPEG_QUALITY = 70

COLORS = {
    'working': (0, 255, 0),
    'notworking': (0, 0, 255),
}
TRAFFIC_LIGHT_COLOR = (0, 255, 255)   # жёлтый - рамка светофора
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ========== HTTP-стриминг (один топик) ==========
jpeg_lock = threading.Lock()
current_jpeg = None


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != STREAM_TOPIC:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
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


# ========== Детектор светофора (OpenCV DNN) ==========
class SvetoforDetector:
    def __init__(self, conf_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        if not os.path.exists(CFG) or not os.path.exists(WEIGHTS):
            print(f'[ERROR] Model files not found in {MODEL_DIR}')
            sys.exit(1)
        with open(NAMES) as f:
            self.class_names = f.read().splitlines()
        self.net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
        self.out_layers = self.net.getUnconnectedOutLayersNames()
        print(f'[INFO] Model loaded, classes={self.class_names}')

    def detect(self, frame):
        """Возвращает список детекций состояния: {class, score, box, label}."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)

        boxes, confidences, class_ids = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < self.conf_threshold:
                    continue
                cx, cy, bw, bh = detection[:4] * np.array([w, h, w, h])
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_id)

        results = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
            indices = np.array(indices).flatten() if len(indices) else []
            for i in indices:
                results.append({
                    'class': self.class_names[class_ids[i]],
                    'score': confidences[i],
                    'box': boxes[i],
                    'label': f"{self.class_names[class_ids[i]]} {confidences[i] * 100:.0f}%",
                })
        return results


# ========== Вспомогательные функции ==========
def union_box(boxes):
    """Общая рамка (union bbox) всех детекций = найденный светофор."""
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2 = [b[0] + b[2] for b in boxes]
    y2 = [b[1] + b[3] for b in boxes]
    return [min(xs), min(ys), max(x2) - min(xs), max(y2) - min(ys)]


def center_inside(box, region):
    """Проверка, что центр bbox находится внутри рамки светофора."""
    cx = box[0] + box[2] / 2
    cy = box[1] + box[3] / 2
    x, y, w, h = region
    return x <= cx <= x + w and y <= cy <= y + h


def resolve_source(source):
    if source != 'auto':
        return source
    records = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'records'))
    videos = sorted(f for f in os.listdir(records)
                    if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))) if os.path.isdir(records) else []
    if not videos:
        print(f'[ERROR] No videos in {records}, pass --source explicitly')
        sys.exit(1)
    return os.path.join(records, videos[0])


def main():
    parser = argparse.ArgumentParser(description='Traffic light + state detection with MJPEG stream')
    parser.add_argument('--source', '-s', default=SOURCE, help='Video source (file, camera index, auto)')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=NMS_THRESHOLD, help='NMS threshold')
    parser.add_argument('--confirm', type=int, default=CONFIRM_FRAMES,
                        help='Frames needed to confirm final class')
    parser.add_argument('--port', type=int, default=STREAM_PORT, help='HTTP stream port')
    args = parser.parse_args()

    detector = SvetoforDetector(args.conf, args.nms)

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video source: {source}')
        return 1
    print(f'[INFO] Source: {source}')

    server = ThreadingHTTPServer((STREAM_HOST, args.port), StreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'[INFO] Stream topic: http://{STREAM_HOST}:{args.port}{STREAM_TOPIC}')

    # --- состояние голосования ---
    vote = {
        'class': None,        # текущий кандидат
        'frames': 0,          # сколько кадров подряд его видим
        'missed': 0,          # сколько кадров без детекций
        'final': None,        # подтверждённый класс
        'printed': None,
    }

    def confirm_vote(detections):
        """Голосование: подтверждаем класс после CONFIRM_FRAMES кадров подряд."""
        if detections:
            # кандидат - класс с максимальной уверенностью в кадре
            best = max(detections, key=lambda d: d['score'])
            if best['class'] == vote['class']:
                vote['frames'] += 1
            else:
                vote['class'] = best['class']
                vote['frames'] = 1
            vote['missed'] = 0
        else:
            vote['missed'] += 1
            if vote['missed'] > MAX_MISSED:
                vote['class'] = None
                vote['frames'] = 0
            if vote['missed'] > MAX_MISSED * 2:
                vote['final'] = None

        if vote['class'] is not None and vote['frames'] >= args.confirm:
            if vote['final'] != vote['class']:
                vote['final'] = vote['class']
                print(f'[RESULT] Светофор: {vote["final"]}')

    global current_jpeg
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str) and source.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # зацикливаем видео
                    vote.update({'class': None, 'frames': 0, 'missed': 0, 'final': None})
                    continue
                break

            # 1) детектируем состояние (working/notworking)
            detections = detector.detect(frame)

            # 2) если нашли хоть что-то - это светофор (union bbox)
            if detections:
                tl_box = union_box([d['box'] for d in detections])
                # 3) оставляем только объекты, чей центр внутри рамки светофора
                inside = [d for d in detections if center_inside(d['box'], tl_box)]
            else:
                tl_box = None
                inside = []

            # 4) голосование для подтверждения класса (наличие нескольких детекций)
            confirm_vote(inside)

            # --- отрисовка ---
            if tl_box is not None:
                x, y, w, h = tl_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), TRAFFIC_LIGHT_COLOR, 2)
                cv2.putText(frame, 'svetofor', (x, y - 10), FONT, 0.6, TRAFFIC_LIGHT_COLOR, 2)
            for det in inside:
                box = det['box']
                color = COLORS.get(det['class'], (255, 255, 255))
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
                cv2.putText(frame, det['label'], (box[0], box[1] - 10), FONT, 0.5, color, 2)
                cx, cy = box[0] + box[2] // 2, box[1] + box[3] // 2
                cv2.drawMarker(frame, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)

            # итоговый класс
            if vote['final']:
                text = f'CLASS: {vote["final"]}'
                color = COLORS.get(vote['final'], (255, 255, 255))
                cv2.putText(frame, text, (30, 50), FONT, 1.0, color, 3)

            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with jpeg_lock:
                    current_jpeg = jpg.tobytes()

    except KeyboardInterrupt:
        print('\n[INFO] Stopping...')
    finally:
        cap.release()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
