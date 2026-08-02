#!/usr/bin/env python3
"""Тест модели YOLOv4-tiny (working / notworking) с веб-камеры и стримингом в браузер.

Модель: model/yolov4-tiny-svetofor_best_weights.weights (OpenCV DNN)
Источник: камера 0 по умолчанию, либо файл через --source
Стриминг: MJPEG в браузер на http://0.0.0.0:8089/

Примеры:
  python3 test_on_video.py
  python3 test_on_video.py --source video.avi
  python3 test_on_video.py --conf 0.25 --input-size 320
"""

import argparse
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')

CFG = os.path.join(MODEL_DIR, 'yolov4-tiny-svetofor.cfg')
WEIGHTS = os.path.join(MODEL_DIR, 'yolov4-tiny-svetofor_best_weights.weights')
NAMES = os.path.join(MODEL_DIR, 'svetofor.names')

DEFAULT_SOURCE = 0           # веб-камера по умолчанию
DEFAULT_INPUT_SIZE = 416     # размер для сети (можно уменьшить для скорости)

# Настройки стриминга
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8089
JPEG_QUALITY = 70

COLORS = {
    'notworking': (0, 0, 255),
    'working': (0, 255, 0),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Глобальная переменная для передачи кадров в HTTP-поток
jpeg_lock = threading.Lock()
current_jpeg = None


def check_model_files():
    missing = []
    for label, path in [('cfg', CFG), ('weights', WEIGHTS), ('names', NAMES)]:
        if not os.path.exists(path):
            missing.append(f'{label}: {path}')
    if missing:
        print('[ERROR] Model files not found:')
        for m in missing:
            print(f'  - {m}')
        sys.exit(1)

    print('[INFO] Model files OK:')
    for label, path in [('cfg', CFG), ('weights', WEIGHTS), ('names', NAMES)]:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f'  - {label}: {os.path.basename(path)} ({size_mb:.1f} MB)')


def load_detector(conf_threshold, nms_threshold):
    check_model_files()
    with open(NAMES) as f:
        class_names = [line.strip() for line in f if line.strip()]

    net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
    out_layers = net.getUnconnectedOutLayersNames()
    print(f'[INFO] Model loaded, classes={class_names}, outputs={out_layers}')
    return net, out_layers, class_names, conf_threshold, nms_threshold


def detect(net, out_layers, class_names, frame, conf_threshold, nms_threshold, input_size):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (input_size, input_size),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < conf_threshold:
                continue
            cx, cy, bw, bh = detection[:4] * np.array([w, h, w, h])
            boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
            confidences.append(confidence)
            class_ids.append(class_id)

    results = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        indices = np.array(indices).flatten() if len(indices) else []
        for i in indices:
            label = class_names[class_ids[i]]
            results.append({
                'class': label,
                'score': confidences[i],
                'box': boxes[i],
                'label': f'{label} {confidences[i] * 100:.0f}%',
            })
    return results


def draw_detections(frame, detections):
    for det in detections:
        x, y, bw, bh = det['box']
        color = COLORS.get(det['class'], (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(frame, det['label'], (x, max(y - 8, 16)),
                    FONT, 0.6, color, 2)


class StreamHandler(BaseHTTPRequestHandler):
    """HTTP хендлер для MJPEG стриминга."""
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
        pass  # отключаем логи HTTP-запросов в консоль


def main():
    parser = argparse.ArgumentParser(description='Test svetofor model with webcam and browser streaming')
    parser.add_argument('--source', '-s', default=DEFAULT_SOURCE,
                        help=f'Camera index or video path (default: {DEFAULT_SOURCE})')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='NMS threshold')
    parser.add_argument('--skip', type=int, default=0,
                        help='Skip N frames between detections (0 = every frame)')
    parser.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE,
                        help=f'Input size for the network (default: {DEFAULT_INPUT_SIZE}). '
                             'Decrease to speed up (e.g. 320, 288).')
    parser.add_argument('--stream-port', type=int, default=STREAM_PORT,
                        help=f'HTTP streaming port (default: {STREAM_PORT})')
    args = parser.parse_args()

    # Открываем источник видео
    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video source: {args.source}')
        return 1

    # Загружаем модель
    net, out_layers, class_names, conf_threshold, nms_threshold = load_detector(
        args.conf, args.nms)

    # Свойства источника
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or fps_in > 120:
        fps_in = 25.0  # fallback для веб-камер
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_video_file = isinstance(args.source, str) and os.path.exists(args.source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video_file else 0

    print(f'[INFO] Source: {args.source}')
    print(f'[INFO] {width}x{height} @ {fps_in:.1f} fps' + (f', frames={total}' if total else ''))
    print(f'[INFO] Using input size: {args.input_size}x{args.input_size}')

    # Запускаем HTTP-сервер для стриминга
    server = ThreadingHTTPServer((STREAM_HOST, args.stream_port), StreamHandler)
    stream_thread = threading.Thread(target=server.serve_forever, daemon=True)
    stream_thread.start()
    print(f'[INFO] Stream: http://{STREAM_HOST}:{args.stream_port}/')

    frame_idx = 0
    det_frames = 0
    total_dets = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    # Зацикливание видеофайла
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    continue
                else:
                    # Камера отключилась или закончился файл
                    print('[WARNING] No more frames, waiting...')
                    time.sleep(0.1)
                    continue

            frame_idx += 1
            detections = []
            if args.skip == 0 or (frame_idx - 1) % (args.skip + 1) == 0:
                detections = detect(net, out_layers, class_names, frame,
                                    conf_threshold, nms_threshold,
                                    args.input_size)
                if detections:
                    det_frames += 1
                    total_dets += len(detections)
                    if total > 0:
                        print(f'[frame {frame_idx}/{total}] {len(detections)} det(s):')
                    else:
                        print(f'[frame {frame_idx}] {len(detections)} det(s):')
                    for det in detections:
                        print(f'  - {det["class"]}: {det["score"] * 100:.1f}% '
                              f'box={det["box"]}')

            draw_detections(frame, detections)

            # Наложение счётчика кадров
            if total > 0:
                info = f'frame {frame_idx}/{total}'
            else:
                info = f'frame {frame_idx}'
            cv2.putText(frame, info, (10, 24), FONT, 0.6, (255, 255, 255), 2)

            # Сжатие и отправка в стрим
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with jpeg_lock:
                    global current_jpeg
                    current_jpeg = jpg.tobytes()

            # Периодический вывод FPS
            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                print(f'[progress] {frame_idx} frames processed, '
                      f'{frame_idx / elapsed:.1f} fps')

    except KeyboardInterrupt:
        print('\n[INFO] Interrupted')
    finally:
        cap.release()
        server.shutdown()
        server.server_close()

    elapsed = time.time() - t0
    print(f'\n[INFO] Done: {frame_idx} frames in {elapsed:.1f}s')
    print(f'[INFO] Frames with detections: {det_frames}/{frame_idx}')
    print(f'[INFO] Total detections: {total_dets}')
    return 0


if __name__ == '__main__':
    sys.exit(main())