import atexit
import time
import random
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from Arduino_A26 import Arduino
from road_utils import *

""" Запускать на бортовом компьютере беспилотника.
    Рядом должны быть Arduino_A26.py и road_utils.py.

    Беспилотник движется по дорожной разметке.
    Красный знак «новые территории» (треугольник) детектируется на кадре
    средствами OpenCV, результат подтверждается трекингом центра,
    после чего беспилотник останавливается и отправляет команду «на взлёт»
    скрипту-имитатору квадрокоптера по UDP (см. quadcopter_sim.py).

    Кадр с отрисованной детекцией транслируется в браузер по MJPEG
    на http://<IP>:8088/

    В основном цикле организованы:
    поиск линий дорожной разметки,
    определение угла поворота колёс для движения к центру полосы,
    опрос микроконтроллера, для отслеживания дистанции, которую осталось проехать,
    детекция знака и подтверждение треком,
    трансляция кадра в браузер.

"""




CAR_SPEED = 1600  # скорость беспилотника
THRESHOLD = 230  # порог бинаризации для поиска линий разметки
CAMERA_ID = '/dev/video0'
# ARDUINO_PORT = 'COM3'
# ARDUINO_PORT = '/dev/ttyS0'
ARDUINO_PORT = '/dev/ttyUSB0'

# детектируемый знак (6 = Sign: artificial roughness) и порог пропусков для СТОП
DETECT_CLASS = 6
MISSED_FRAMES_LIMIT = 5

# Детекция знака «новые территории»: красный треугольник в полосе кадра
TOP_FILL_H = 300
BOTTOM_FILL_Y = 640
MIN_AREA = 400               # минимальная площадь контура (пиксели)
RED_RANGES = [               # диапазоны HSV для красного цвета
    ((0, 70, 50), (10, 255, 255)),
    ((170, 70, 50), (180, 255, 255)),
]

# Трекинг центра знака (фильтрует случайные срабатывания)
TRACK_DIST = 100     # макс. смещение центра между соседними кадрами, px
TRACK_MISS_MAX = 5   # через сколько кадров без детекта трек сбрасывается
TRACK_CONFIRM = 3    # сколько кадров подряд нужно для подтверждения
TRACK_STATE = {'x': None, 'y': None, 'missed': 0, 'confirmed': 0}

# Команда «на взлёт» для квадрокоптера (скрипт-имитатор, слушает UDP)
QUAD_HOST = '127.0.0.1'
QUAD_PORT = 9000

GO = 'GO'
STOP = 'STOP'

STATE = GO
PREV_STATE = None
PREV_SUBSTATE = None
SUBSTATE = None


def triangle_center(pts):
    return int(pts[:, 0].mean()), int(pts[:, 1].mean())


def dist_to_track(x, y):
    if TRACK_STATE['x'] is None:
        return None
    return ((x - TRACK_STATE['x']) ** 2 + (y - TRACK_STATE['y']) ** 2) ** 0.5


def update_tracker(centroids):
    """Обновляет трек. Возвращает индекс выбранного треугольника или None."""
    if not centroids:
        TRACK_STATE['missed'] += 1
        if TRACK_STATE['missed'] > TRACK_MISS_MAX:
            TRACK_STATE.update(x=None, y=None, missed=0, confirmed=0)
        return None

    if TRACK_STATE['x'] is None:
        cx, cy = centroids[0]
        TRACK_STATE.update(x=cx, y=cy, missed=0, confirmed=1)
        return 0

    distances = [dist_to_track(cx, cy) for cx, cy in centroids]
    idx = int(np.argmin(distances))
    if distances[idx] > TRACK_DIST:
        # рядом с треком ничего нет — случайный объект, игнорируем
        TRACK_STATE['missed'] += 1
        if TRACK_STATE['missed'] > TRACK_MISS_MAX:
            TRACK_STATE.update(x=None, y=None, missed=0, confirmed=0)
        return None

    cx, cy = centroids[idx]
    TRACK_STATE.update(x=cx, y=cy, missed=0, confirmed=TRACK_STATE['confirmed'] + 1)
    return idx


def find_triangular_red(frame):
    """Ищет красные области в видимой полосе, возвращает треугольники
    (массивы точек в координатах исходного кадра)."""
    h, w = frame.shape[:2]
    y1 = min(TOP_FILL_H, h)
    y2 = min(BOTTOM_FILL_Y, h)
    band = frame[y1:y2, :]

    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = np.zeros(band.shape[:2], dtype=np.uint8)
    for lower, upper in RED_RANGES:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) == 3:  # примерно треугольник
            pts = approx.reshape(-1, 2).astype(np.int32)
            pts[:, 1] += y1  # возвращаем в координаты всего кадра
            found.append(pts)
    return found


def draw_signs(frame, triangles):
    """Рисует рамку и вершины найденных треугольников (для трансляции в браузер)."""
    for pts in triangles:
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)


def draw_center(frame, cx, cy):
    """Рисует крест с кругом в центре знака."""
    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 2)
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 2)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 2)


def send_takeoff():
    """Отправляет команду «на взлёт» квадрокоптеру по UDP."""
    msg = 'на взлёт'.encode('utf-8')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(msg, (QUAD_HOST, QUAD_PORT))


arduino = None
video_orig = None

# HTTP streaming config
STREAM_HOST = '0.0.0.0'
STREAM_PORT = 8088
JPEG_QUALITY = 70

jpeg_lock = threading.Lock()
current_jpeg = None


@atexit.register
def exit_func(*args):
    global arduino
    if arduino is not None:
        try:
            arduino.close()
        finally:
            arduino = None
    if video_orig is not None:
        video_orig.close()
# cv2.destroyAllWindows()


arduino = Arduino(ARDUINO_PORT)
print("Arduino connected")

# астраиваем камеру
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

find_lines = centre_mass2 # название функции для поиска линий разметки

# HTTP streaming handler
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

# Start streaming server in background
server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
stream_thread = threading.Thread(target=server.serve_forever, daemon=True)
stream_thread.start()
print(f'[INFO] Streaming on http://{STREAM_HOST}:{STREAM_PORT}')

# пропускаем часть кадров, для стабилизации настроек камеры
for i in range(30):
    ret, frame = cap.read()

last_err = 0
ped_log_state_prev = None
last_ped = 0
while True:
    try:
        start_time = time.time()
        ret, frame = cap.read()
        end_frame = time.time()
        if not ret:
            break

        # Детекция знака «новые территории» на полном кадре.
        triangles = find_triangular_red(frame) if STATE != STOP else []
        centroids = [triangle_center(pts) for pts in triangles]
        idx = update_tracker(centroids)
        if idx is not None and TRACK_STATE['confirmed'] >= TRACK_CONFIRM:
            print('на взлёт')
            send_takeoff()
            TRACK_STATE.update(x=None, y=None, missed=0, confirmed=0)
            STATE = STOP

        # Кадр для трансляции в браузер: полный кадр с отрисованной детекцией
        stream_frame = frame.copy()
        draw_signs(stream_frame, triangles)
        if TRACK_STATE['x'] is not None:
            draw_center(stream_frame, TRACK_STATE['x'], TRACK_STATE['y'])
        if STATE == STOP:
            cv2.putText(stream_frame, 'НА ВЗЛЁТ', (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        frame = frame[-720:, :]  # для поиска разметки весь кадр не нужен
        orig_frame = frame.copy()
        frame = cv2.resize(frame, SIZE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Переводим изображение в чёрно-белое с градациями серого
        bin = cv2.inRange(gray, THRESHOLD, 255)  # Бинаризуем по порогу, должны остаться только белые линии разметки
        # bin = binarize(frame, THRESHOLD)

    
        wrapped = trans_perspective(bin, TRAP, RECT, SIZE)  # получаем область перед колёсами
        left, right = find_lines(wrapped)  # координаты левой и правой линий разметки
    
        # ПИД-регулятор для определения угла поворота колёс
        # ПИД старается удерживать центр кадра ровно между линиями дорожной разметки
        err = 0 - ((left + right) // 2 - wrapped.shape[1] // 2)
        err = -err  # Инвертирование направления поворота колёс
        angle = int(90 + KP * err + KD * (err - last_err))  # высчитываем угол
        last_err = err
    
        angle = min(max(45, angle), 135)
        print(angle)
    
    
       
    
        if PREV_STATE != STATE:
            print(f'STATE: {STATE})')
            PREV_STATE = STATE
    
        if STATE != STOP:
            arduino.set_speed(CAR_SPEED)
            arduino.set_angle(angle)
        else:
            arduino.set_speed(1500)  # Стоп-сигнал
            # arduino.stop()  # Ардуино подтвердит получение

        # Отдаём кадр с детекцией на стриминг в браузер
        ok, jpg = cv2.imencode('.jpg', stream_frame,
                              [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with jpeg_lock:
                current_jpeg = jpg.tobytes()
    
        end_time = time.time()
    
        fps = 1 / (end_time - start_time)
        if fps < 10:
            print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')
    except KeyboardInterrupt:
        print('\n[INFO] KeyboardInterrupt received, stopping the robot...')
        break


if arduino is not None:
    arduino.close()
    arduino = None
server.shutdown()
server.server_close()
cap.release()
cv2.destroyAllWindows()
#exit()
