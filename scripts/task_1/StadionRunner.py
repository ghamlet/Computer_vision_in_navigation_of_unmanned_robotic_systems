import atexit
import time
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from Arduino_A26 import Arduino
from road_utils import *

""" Запускать на бортовом компьютере беспилотника.
    Рядом должны быть Arduino_A26.py и road_utils.py.

    Беспилотник движется по дорожной разметке и использует нейросетевой детектор для
    обнаружения: пешеходов, знаков, светофоров.
    При появлении в кадре пешехода беспилотный автомобиль останавливается.

    В основном цикле организованы:
    поиск линий дорожной разметки,
    определение угла поворота колёс для движения к центру полосы,
    опрос микроконтроллера, для отслеживания дистанции, которую осталось проехать,
    обработка кадра нейросетевым детектором,
    анализ результатов работы детектора.

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

GO = 'GO'
STOP = 'STOP'

STATE = GO
PREV_STATE = None
PREV_SUBSTATE = None
SUBSTATE = None


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
