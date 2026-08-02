#!/usr/bin/env python3
"""
Видеоанализ с двухэтапной детекцией светофора:
1) yolopy (класс 9) находит светофор на всём кадре.
2) YOLOv4-tiny (OpenCV DNN) определяет working/notworking только внутри bbox светофора.
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
SOURCE = 0  # веб-камера или путь к файлу

# Зеленые зоны
TOP_FILL_H = 100
BOTTOM_FILL_Y = 640
FILL_X = 870
FILL_COLOR = (0, 255, 0)

# Класс светофора в classes.txt (индекс 9)
TRAFFIC_LIGHT_CLASS = 9

# Размер кадра для обработки
TARGET_SIZE = (1280, 960)

# Параметры кропа вокруг bbox yolopy
CROP_PADDING = 0.2         # 20% отступа с каждой стороны
DNN_INPUT_SIZE = 416       # входной размер для DNN модели

# HTTP streaming
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8089
JPEG_QUALITY = 70

# Директории поиска моделей
MODEL_DIR = Path("/home/arrma/Computer_vision_in_navigation_of_unmanned_robotic_systems/scripts/models")
DNN_MODEL_DIR = Path(__file__).parent / "model"  # папка model рядом со скриптом

jpeg_lock = threading.Lock()
current_jpeg = None


def fill_zones(frame):
    """Закрашивает зеленым: верх 100px, низ от 640px, лево до 870px"""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, min(TOP_FILL_H, h)), FILL_COLOR, -1)
    cv2.rectangle(frame, (0, min(BOTTOM_FILL_Y, h)), (w, h), FILL_COLOR, -1)
    cv2.rectangle(frame, (0, 0), (min(FILL_X, w), h), FILL_COLOR, -1)


def find_file(name, search_dirs):
    """Ищет файл по имени в списке директорий."""
    for d in search_dirs:
        p = Path(d) / name
        if p.exists():
            return p
    return None


# ======= Детектор yolopy (светофор, класс 9) =======
class TrafficLightDetector:
    def __init__(self, detect_class=TRAFFIC_LIGHT_CLASS):
        self.detect_class = detect_class
        self.model = None
        self.class_names = []

        if not YOLOPY_AVAILABLE:
            print("[INFO] yolopy not available, detector disabled")
            return

        search_paths = [Path.cwd(), MODEL_DIR, Path('/home/avt_user/Base_Code'), Path('/home/avt_user/PROGRAMMS')]
        classes_file = find_file('classes.txt', search_paths)
        model_file = find_file('yolo_uint8.tmfile', search_paths)

        if classes_file is None or model_file is None:
            print("[WARNING] yolopy model files not found, detector disabled")
            return

        with open(classes_file) as f:
            self.class_names = f.read().splitlines()

        try:
            self.model = yolopy.Model(
                str(model_file),
                use_uint8=True,
                use_timvx=True,
                cls_num=len(self.class_names)
            )
            self.model.set_anchors([18, 33, 33, 48, 25, 71, 58, 76, 40, 113, 87, 140])
            print(f"[INFO] yolopy model loaded: {model_file}")
        except Exception as e:
            print(f"[ERROR] Failed to load yolopy model: {e}")
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
                        'box': box,   # [x, y, w, h]
                        'label': self.class_names[classid] if classid < len(self.class_names) else f'class_{classid}'
                    })
            return results
        except Exception as e:
            print(f"[ERROR] yolopy detection failed: {e}")
            return []

    def draw_detections(self, frame, detections, color=(0, 255, 0)):
        for det in detections:
            box = det['box']
            label = f"{det['label']} [{det['score']*100:.1f}%]"
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


# ======= Детектор YOLOv4-tiny (working / notworking) =======
class DNNTrafficLightDetector:
    def __init__(self, cfg_path=None, weights_path=None, names_path=None, input_size=DNN_INPUT_SIZE):
        self.input_size = input_size
        self.net = None
        self.class_names = []
        self.out_layers = None

        search_dirs = [DNN_MODEL_DIR, Path.cwd(), MODEL_DIR]
        cfg = cfg_path or find_file('yolov4-tiny-svetofor.cfg', search_dirs)
        weights = weights_path or find_file('yolov4-tiny-svetofor_best_weights.weights', search_dirs)
        names = names_path or find_file('svetofor.names', search_dirs)

        missing = []
        for name, path in [('cfg', cfg), ('weights', weights), ('names', names)]:
            if not path or not Path(path).exists():
                missing.append(name)
        if missing:
            print(f"[WARNING] YOLOv4-tiny DNN model files missing: {missing}. Detector disabled.")
            return

        with open(names) as f:
            self.class_names = [line.strip() for line in f if line.strip()]

        self.net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
        self.out_layers = self.net.getUnconnectedOutLayersNames()
        print(f"[INFO] YOLOv4-tiny DNN model loaded: {weights}")

    def detect_on_crop(self, crop, conf_threshold=0.25, nms_threshold=0.4):
        """Запускает детекцию на кропе, возвращает список детекций в координатах кропа (после ресайза)."""
        if self.net is None:
            return []

        # Ресайз кропа до input_size
        resized = cv2.resize(crop, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(resized, 1/255.0, (self.input_size, self.input_size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)

        boxes_list, confidences, class_ids = [], [], []
        # Размеры после ресайза – всегда input_size x input_size
        h_resized, w_resized = self.input_size, self.input_size
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < conf_threshold:
                    continue
                # Координаты в масштабе resized
                cx, cy, bw, bh = detection[:4] * np.array([w_resized, h_resized, w_resized, h_resized])
                boxes_list.append([
                    int(cx - bw/2),
                    int(cy - bh/2),
                    int(bw),
                    int(bh)
                ])
                confidences.append(confidence)
                class_ids.append(class_id)

        results = []
        if boxes_list:
            indices = cv2.dnn.NMSBoxes(boxes_list, confidences, conf_threshold, nms_threshold)
            indices = np.array(indices).flatten() if len(indices) else []
            for i in indices:
                label = self.class_names[class_ids[i]]
                results.append({
                    'class': label,
                    'score': confidences[i],
                    'box': boxes_list[i],   # x, y, w, h в масштабе resized
                    'label': f"{label} {confidences[i]*100:.0f}%"
                })
        return results

    def draw_detections(self, frame, detections):
        colors = {'working': (0, 255, 0), 'notworking': (0, 0, 255)}
        for det in detections:
            box = det['box']
            color = colors.get(det['class'], (255, 255, 255))
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
            cv2.putText(frame, det['label'], (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


# ======= Вспомогательная функция: обработка региона светофора =======
def process_traffic_light_region(frame, bbox, dnn_detector):
    """
    Вырезает область вокруг bbox, прогоняет через DNN и возвращает детекции
    в координатах исходного кадра.
    bbox: [x, y, w, h]
    """
    x, y, w, h = bbox
    pad_w = int(w * CROP_PADDING)
    pad_h = int(h * CROP_PADDING)

    # Границы кропа с учётом padding, не выходя за пределы кадра
    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame.shape[1], x + w + pad_w)
    y2 = min(frame.shape[0], y + h + pad_h)

    if x2 <= x1 or y2 <= y1:
        return []

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    # Детекция на кропе
    detections_resized = dnn_detector.detect_on_crop(crop)  # box в масштабе input_size

    # Масштабируем координаты обратно на исходный кроп, затем добавляем смещение
    crop_w = x2 - x1
    crop_h = y2 - y1
    scale_x = crop_w / dnn_detector.input_size
    scale_y = crop_h / dnn_detector.input_size

    global_detections = []
    for det in detections_resized:
        bx, by, bw, bh = det['box']
        # В координатах кропа
        bx_crop = int(bx * scale_x)
        by_crop = int(by * scale_y)
        bw_crop = int(bw * scale_x)
        bh_crop = int(bh * scale_y)
        # В координатах полного кадра
        gx = x1 + bx_crop
        gy = y1 + by_crop
        global_detections.append({
            'class': det['class'],
            'score': det['score'],
            'box': [gx, gy, bw_crop, bh_crop],
            'label': det['label']
        })
    return global_detections


# ======= HTTP стриминг =======
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
    print(f'[INFO] Pipeline: yolopy -> traffic light bbox -> DNN (working/notworking) inside bbox')

    # Инициализация детекторов
    yolopy_detector = TrafficLightDetector()
    dnn_detector = DNNTrafficLightDetector()

    server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
    stream_thread = threading.Thread(target=server.serve_forever, daemon=True)
    stream_thread.start()
    print("[INFO] Streaming server started")

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(SOURCE, str) and SOURCE.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1
            if TARGET_SIZE:
                frame = cv2.resize(frame, TARGET_SIZE)

            fill_zones(frame)

            # --- Этап 1: поиск светофора yolopy ---
            tl_detections = yolopy_detector.detect(frame)
            # Рисуем bbox светофоров (зелёные)
            yolopy_detector.draw_detections(frame, tl_detections, color=(255, 255, 0))

            # --- Этап 2: для каждого светофора определяем working/notworking ---
            if tl_detections:
                print(f"\n--- Frame {frame_count} ---")
                for idx, tl in enumerate(tl_detections):
                    bbox = tl['box']
                    print(f"Светофор #{idx+1}: bbox={bbox}, score={tl['score']:.2f}")
                    # Запускаем DNN на кропе
                    status_dets = process_traffic_light_region(frame, bbox, dnn_detector)
                    if status_dets:
                        for det in status_dets:
                            print(f"  -> Статус: {det['class']} ({det['score']*100:.0f}%)")
                    else:
                        print("  -> Статус не определён")
                    # Рисуем результаты DNN (красные/зелёные боксы)
                    dnn_detector.draw_detections(frame, status_dets)
            else:
                print(f"Frame {frame_count}: светофор не найден")

            # Отправка в стрим
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