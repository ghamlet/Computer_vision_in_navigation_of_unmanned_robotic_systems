""" Версия StadionRunner.py из Bazovyi_774_kod_-_AI_774_KAR
    БЕЗ логики управления колёсами, БЕЗ Arduino и БЕЗ нейросетевых моделей.

    Запускать на бортовом компьютере беспилотника.
    Рядом должен быть road_utils.py.

    Осталось только:
    - захват кадра с камеры,
    - предобработка (кроп, resize, бинаризация, перспектива),
    - поиск линий дорожной разметки,
    - визуализация найденных линий,
    - вывод FPS.

"""

import time

import cv2

from road_utils import SIZE, THRESHOLD, TRAP, RECT, trans_perspective, centre_mass2

CAMERA_ID = '/dev/video4'

# настраиваем камеру
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

find_lines = centre_mass2  # название функции для поиска линий разметки

# пропускаем часть кадров, для стабилизации настроек камеры
for i in range(30):
    ret, frame = cap.read()

while True:
    start_time = time.time()
    try:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[-720:, :]  # для поиска разметки весь кадр не нужен
        orig_frame = frame.copy()
        frame = cv2.resize(frame, SIZE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # переводим изображение в чёрно-белое с градациями серого
        bin = cv2.inRange(gray, THRESHOLD, 255)  # бинаризуем по порогу, должны остаться только белые линии разметки

        
        wrapped = trans_perspective(bin, TRAP, RECT, SIZE)  # получаем область перед колёсами
        left, right = find_lines(wrapped)  # координаты левой и правой линий разметки

        # визуализация найденных линий
        viz = cv2.cvtColor(wrapped, cv2.COLOR_GRAY2BGR)
        cv2.line(viz, (left, 0), (left, viz.shape[0]), (0, 0, 255), 2)
        cv2.line(viz, (right, 0), (right, viz.shape[0]), (0, 0, 255), 2)
        cv2.line(viz, ((left + right) // 2, 0), ((left + right) // 2, viz.shape[0]), (0, 255, 0), 2)
        cv2.imshow('wrapped', viz)
        cv2.imshow('frame', orig_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC - выход
            break

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f'left={left} right={right} fps={fps:.1f}')
        if fps < 10:
            print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')
    except KeyboardInterrupt:
        print('\n[INFO] KeyboardInterrupt received, stopping...')
        break

cap.release()
cv2.destroyAllWindows()
