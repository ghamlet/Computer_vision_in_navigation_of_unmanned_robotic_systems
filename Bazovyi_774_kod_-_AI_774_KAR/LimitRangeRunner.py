import atexit
import time
import random

import cv2
import numpy as np

from Arduino_A26 import Arduino
from road_utils import *

""" Запускать на бортовом компьютере беспилотника.
    Рядом должны быть Arduino_A26.py и road_utils.py.
    
    Беспилотник проезжает по полосе дорожной разметки определённую дистанцию и
    останавливается.
    
    В основном цикле организованы:
    поиск линий дорожной разметки,
    определение угла поворота колёс для движения к центру полосы,
    опрос микроконтроллера, для отслеживания дистанции, которую осталось проехать.

"""

NEED_DIST = 4000  #  число срабатываний энкодера, которое необходимо проехать
CAR_SPEED = 1605  # скорость беспилотника
THRESHOLD = 200  # порог бинаризации для поиска линий разметки
CAMERA_ID = '/dev/video0'
# ARDUINO_PORT = 'COM3'
# ARDUINO_PORT = '/dev/ttyS0'
ARDUINO_PORT = '/dev/ttyUSB0'

ARDUINO_CURRENT_DIST = 0  # последняя дистанция, которую сообщил микроконтроллер
PREV_ARDUINO_CURRENT_DIST = 0  # последняя дистанция, которую сообщил микроконтроллер

arduino = None
video_orig = None

@atexit.register
def exit_func(*args):
    if arduino is not None:
        arduino.close()
    if video_orig is not None:
        video_orig.close()
    # cv2.destroyAllWindows()


# пример обработки уже принятых сообщений от микроконтроллера
# интерпретирует текст сообщения и выполняет соответствующие действия
def parseMessage(msg):
    global ARDUINO_CURRENT_DIST

    # простое сообщение из одного текстового поля, проверяем целиком
    if msg == "OK_STOP":
        print("\n***Response from Arduino***")
        print(f"Text: <{msg}>")
        return True

    # остались сообщения из двух полей, текст и значение
    parts = msg.split(':')
    if len(parts) <=1: # если часть одна, то ":" не оказалось
        return False  # сообщение не соответствует формату
    value = int(parts[1])  # переводим в число вторую часть строки от пользователя

    if parts[0] == "STAT_DIST":
        print("\n***Response from Arduino***")
        print(f"Text: <{parts[0]}>")
        print(f"Value: <{value}>")

        # меняем значение, которое будем анализировать в основном цикле
        ARDUINO_CURRENT_DIST = value
        return True

    if parts[0] == "OK_DIST":
        print("\n***Response from Arduino***")
        print(f"Text: <{parts[0]}>")
        print(f"Value: <{value}>")

        # меняем значение, которое будем анализировать в основном цикле
        ARDUINO_CURRENT_DIST = value
        return True

    return False  # сообщение не соответствует формату протокола

arduino = Arduino(ARDUINO_PORT)
print("Arduino connected")

# настраиваем камеру
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

##############################################################
####   Указываем дистанцию, которую необходимо проехать   ####
##############################################################

arduino.set_dist(NEED_DIST)

# принимаем подтверждение от Arduino
start_time = time.monotonic()
while arduino.receive_message() is False: # крутим цикл, пока не получим сообщение
    if time.monotonic() - start_time > 1: # через секунду выходим из цикла
        print("!!! The DIST command is missing !!!")
        arduino.close()  # закрываем соединение
        exit()
else:
    msg = arduino.get_msg_from_buffer()  # получаем сообщение и обрабатываем его
    parseMessage(msg)  # обрабатываем сообщения

arduino.set_speed(CAR_SPEED)  # задаём скорость движения вперёд

last_err = 0
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
    arduino.set_angle(angle)

    arduino.get_dist()  # запрашиваем текущее значение дистанции для Arduino

    # пример того, как ожидать сообщения не блокируя обработку кадров
    while arduino.receive_message():  # если сообщение есть, то
        msg = arduino.get_msg_from_buffer()  # получаем его текст
        parseMessage(msg)  # обрабатываем сообщения

    # print(ARDUINO_CURRENT_DIST, PREV_ARDUINO_CURRENT_DIST)
    if (ARDUINO_CURRENT_DIST == 0) and (PREV_ARDUINO_CURRENT_DIST > ARDUINO_CURRENT_DIST):
        print("Distance covered")
        arduino.stop()
        break
    PREV_ARDUINO_CURRENT_DIST = ARDUINO_CURRENT_DIST

    end_time = time.time()

    fps = 1 / (end_time - start_time)
    if fps < 10:
        print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')

# строка ниже не нужна, соединение закроется в exit_func
# arduino.close()  # закрываем соединение