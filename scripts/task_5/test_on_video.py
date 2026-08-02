#!/usr/bin/env python3
"""Тест модели YOLOv4-tiny (working / notworking) на Raspberry Pi с камерой.

Оптимизировано для Raspberry Pi:
- нет GUI окон
- стриминг в браузер
- адаптивные пути
- проверка доступности камеры
"""

import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

# ========== НАСТРОЙКИ ДЛЯ RASPBERRY PI ==========
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / 'model'

# Поиск модели в разных местах
POSSIBLE_MODEL_DIRS = [
    MODEL_DIR,
    Path.home() / 'Base_Code',
    Path.home() / 'PROGRAMMS',
    Path.home() / 'models',
    Path('/home/arrma/Computer_vision_in_navigation_of_unmanned_robotic_systems/scripts/models'),
]

CFG_NAME = 'yolov4-tiny-svetofor.cfg'
WEIGHTS_NAME = 'yolov4-tiny-svetofor_best_weights.weights'  
NAMES_NAME = 'svetofor.names'

# Камера
CAMERA_INDEX = 0
FRAME_WIDTH = 640   # уменьшаем для производительности
FRAME_HEIGHT = 480

# DNN настройки
INPUT_SIZE = 320    # меньше = быстрее для Pi
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.4

# Стриминг
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8089
JPEG_QUALITY = 70

COLORS = {
    'notworking': (0, 0, 255),
    'working': (0, 255, 0),
}

jpeg_lock = threading.Lock()
current_jpeg = None


def find_model_files():
    """Ищет файлы модели в возможных директориях."""
    for dir_path in POSSIBLE_MODEL_DIRS:
        cfg = dir_path / CFG_NAME
        weights = dir_path / WEIGHTS_NAME
        names = dir_path / NAMES_NAME
        
        if cfg.exists() and weights.exists() and names.exists():
            return str(cfg), str(weights), str(names)
    
    print("[ERROR] Model files not found! Searched in:")
    for d in POSSIBLE_MODEL_DIRS:
        print(f"  - {d}")
    return None, None, None


def load_detector(conf_threshold, nms_threshold):
    """Загружает DNN модель."""
    cfg, weights, names = find_model_files()
    if cfg is None:
        sys.exit(1)
    
    # Проверка файлов
    for path, desc in [(cfg, 'cfg'), (weights, 'weights'), (names, 'names')]:
        if not os.path.exists(path):
            print(f'[ERROR] Missing: {desc} at {path}')
            sys.exit(1)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f'[INFO] {desc}: {os.path.basename(path)} ({size_mb:.1f} MB)')
    
    # Загрузка классов
    with open(names) as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    # Загрузка сети
    try:
        # Для Raspberry Pi может потребоваться другая загрузка
        print("[INFO] Loading DNN model (this may take a while on Pi)...")
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
        
        # Попытка использовать оптимизацию (если доступна)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("[INFO] Using CUDA backend")
        else:
            # Для Pi используем оптимизацию CPU
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("[INFO] Using CPU backend")
        
        out_layers = net.getUnconnectedOutLayersNames()
        print(f'[INFO] Model loaded, classes={class_names}')
        return net, out_layers, class_names
        
    except Exception as e:
        print(f'[ERROR] Failed to load model: {e}')
        print("[TIP] Try installing opencv with DNN support:")
        print("  pip install opencv-python-headless==4.5.5.64")
        sys.exit(1)


def detect(net, out_layers, class_names, frame):
    """Детекция объектов на кадре."""
    h, w = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(out_layers)
    
    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < CONF_THRESHOLD:
                continue
            cx, cy, bw, bh = detection[:4] * np.array([w, h, w, h])
            boxes.append([int(cx - bw/2), int(cy - bh/2), int(bw), int(bh)])
            confidences.append(confidence)
            class_ids.append(class_id)
    
    results = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        indices = np.array(indices).flatten() if len(indices) else []
        for i in indices:
            label = class_names[class_ids[i]]
            results.append({
                'class': label,
                'score': confidences[i],
                'box': boxes[i],
                'label': f'{label} {confidences[i]*100:.0f}%',
            })
    return results


def draw_detections(frame, detections):
    """Отрисовка результатов детекции."""
    for det in detections:
        x, y, bw, bh = det['box']
        color = COLORS.get(det['class'], (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
        cv2.putText(frame, det['label'], (x, max(y-8, 16)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class StreamHandler(BaseHTTPRequestHandler):
    """HTTP handler для MJPEG стриминга."""
    def do_GET(self):
        if self.path == '/':
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
        else:
            self.send_error(404)
    
    def log_message(self, *args):
        pass


def init_camera():
    """Инициализация камеры для Raspberry Pi."""
    # Пробуем разные методы открытия камеры для Pi
    for api in [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_DSHOW]:
        cap = cv2.VideoCapture(CAMERA_INDEX, api)
        if cap.isOpened():
            print(f"[INFO] Camera opened with API: {api}")
            break
    else:
        print("[ERROR] Cannot open camera")
        return None
    
    # Установка разрешения
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Умеренный FPS для Pi
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'[INFO] Camera resolution: {actual_w}x{actual_h}')
    
    return cap


def main():
    print("[INFO] Raspberry Pi Traffic Light Detector")
    print("[INFO] ====================================")
    
    # Загрузка модели
    net, out_layers, class_names = load_detector(CONF_THRESHOLD, NMS_THRESHOLD)
    
    # Инициализация камеры
    cap = init_camera()
    if cap is None:
        return 1
    
    # Запуск стриминг-сервера
    server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
    stream_thread = threading.Thread(target=server.serve_forever, daemon=True)
    stream_thread.start()
    print(f'[INFO] Stream: http://{STREAM_HOST}:{STREAM_PORT}/')
    
    frame_idx = 0
    det_frames = 0
    total_dets = 0
    t0 = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame capture failed")
                time.sleep(0.1)
                continue
            
            frame_idx += 1
            
            # Детекция (каждый кадр для теста)
            detections = detect(net, out_layers, class_names, frame)
            
            if detections:
                det_frames += 1
                total_dets += len(detections)
                print(f'[frame {frame_idx}] {len(detections)} det(s):')
                for det in detections:
                    print(f'  - {det["class"]}: {det["score"]*100:.1f}%')
            
            # Отрисовка
            draw_detections(frame, detections)
            
            # Статистика
            elapsed = time.time() - t0
            fps = frame_idx / elapsed if elapsed > 0 else 0
            info = f'FPS: {fps:.1f} | Dets: {total_dets}'
            cv2.putText(frame, info, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Сжатие и отправка
            ok, jpg = cv2.imencode('.jpg', frame, 
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with jpeg_lock:
                    global current_jpeg
                    current_jpeg = jpg.tobytes()
            
            # Периодический вывод
            if frame_idx % 30 == 0:
                print(f'[INFO] FPS: {fps:.1f}, Dets: {total_dets}')
    
    except KeyboardInterrupt:
        print('\n[INFO] Interrupted')
    finally:
        cap.release()
        server.shutdown()
        server.server_close()
    
    elapsed = time.time() - t0
    print(f'\n[INFO] Done: {frame_idx} frames in {elapsed:.1f}s')
    print(f'[INFO] Detection frames: {det_frames}/{frame_idx}')
    print(f'[INFO] Total detections: {total_dets}')
    return 0


if __name__ == '__main__':
    sys.exit(main())