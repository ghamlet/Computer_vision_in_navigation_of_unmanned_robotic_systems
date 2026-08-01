import atexit
import time
import random
from pathlib import Path

import cv2
import numpy as np
import yolopy

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

CAR_SPEED = 1605  # скорость беспилотника
THRESHOLD = 200  # порог бинаризации для поиска линий разметки
CAMERA_ID = '/dev/video0'
# ARDUINO_PORT = 'COM3'
# ARDUINO_PORT = '/dev/ttyS0'
ARDUINO_PORT = '/dev/ttyUSB0'

GO = 'GO'
STOP = 'STOP'

STATE = GO
PREV_STATE = None
PREV_SUBSTATE = None
SUBSTATE = None

arduino = None
video_orig = None

@atexit.register
def exit_func(*args):
    if arduino is not None:
        arduino.close()
    if video_orig is not None:
        video_orig.close()
    # cv2.destroyAllWindows()


# загружаем модель машинного обучения из файлов
with open('classes.txt') as file:
    class_names = file.read().splitlines()
model_file = str('yolo_uint8.tmfile')  # задаем путь к файлу модели YOLO
model = yolopy.Model(model_file, use_uint8=True, use_timvx=True, cls_num=10)  # загружаем модель YOLO
model.set_anchors([18, 33, 33, 48, 25, 71, 58, 76, 40, 113, 87, 140])  # задаем якорные точки для модели YOLO

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

# пропускаем часть кадров, для стабилизации настроек камеры
for i in range(30):
    ret, frame = cap.read()

last_err = 0
ped_log_state_prev = None
last_ped = 0
while True:
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


    # Детектируем пешеходов на изображении с камеры #
    #################################################

    classes, scores, boxes = model.detect(orig_frame)  # отдаём кадр детектору
    # labeled_frame = orig_frame.copy()
    peds = []
    for classid, score, box in zip(classes, scores, boxes):
        if classid == 0: # если нашли пешехода, то
            label = f'PERSON [{score * 100:.2f}%]'
            x, y, w, h = box
            pd_area = w * h
            xc = x + w // 2
            yc = y + h // 2
            peds.append((pd_area, xc, yc)) # добавляем его координаты в список
        elif classid == 9:
            pass
        else:
            label = f'{class_names[classid]} [{score * 100:.2f}%]'

        # Отрисовка ограничивающих рамок
        # color = (0, 0, 255)
        # FONT = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.rectangle(labeled_frame, box, color, 2)
        # cv2.putText(labeled_frame, label, (box[0], box[1] - 10), FONT, 0.5, color, 2)
        # cv2.imwrite("Detection.png", labeled_frame)

    frame_width = orig_frame.shape[1]
    amount = 0
    # amount_on = 0
    ped_detected = False
    for pd_area, xc, xy in peds:
        if (frame_width / 5 < xc) and (xc < frame_width / 5 * 4):
            ped_detected = True
            last_ped = time.time()
            amount += 1  # число пешеходов в кадре

    if amount:  # Если есть пешеходы, то ...
        ped_log_state = f'{amount} pedestrians in my RoI'
        STATE = STOP
    else:
        ped_log_state = 'NO pedestrians in my RoI'
        STATE = GO

    if ped_log_state != ped_log_state_prev:
        print("Pedestrian info:", ped_log_state)
        ped_log_state_prev = ped_log_state
    # # --- DETECT PED END --- #

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
