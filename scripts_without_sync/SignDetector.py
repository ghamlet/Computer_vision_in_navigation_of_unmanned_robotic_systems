""" Детекция ТОЛЬКО дорожных знаков.
    На основе StadionRunner.py из Bazovyi_774_kod_-_AI_774_KAR,
    но БЕЗ логики движения (без Arduino, без управления колёсами, без ПИД).

    Запускать на бортовом компьютере беспилотника.
    Рядом должны лежать: classes.txt, yolo_uint8.tmfile, установлен yolopy.

    В кадре детектируются только знаки (классы 1-8).
    Пешеходы (0) и светофоры (9) игнорируются.

"""

import time

import cv2
import yolopy

CAMERA_ID = '/dev/video0'

# загружаем модель машинного обучения из файлов
with open('classes.txt') as file:
    class_names = file.read().splitlines()
model_file = str('yolo_uint8.tmfile')  # путь к файлу модели YOLO
model = yolopy.Model(model_file, use_uint8=True, use_timvx=True, cls_num=10)  # загружаем модель YOLO
model.set_anchors([18, 33, 33, 48, 25, 71, 58, 76, 40, 113, 87, 140])  # якорные точки для модели YOLO

# классы дорожных знаков (все, кроме pedestrian=0 и traffic light=9)
SIGN_CLASSES = set(range(1, 9))

# настраиваем камеру
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print('[ERROR] Cannot open camera ID:', CAMERA_ID)
    quit()

# пропускаем часть кадров, для стабилизации настроек камеры
for i in range(30):
    ret, frame = cap.read()

FONT = cv2.FONT_HERSHEY_SIMPLEX
SIGN_COLOR = (0, 255, 0)

while True:
    start_time = time.time()
    try:
        ret, frame = cap.read()
        if not ret:
            break

        # Детектируем объекты нейросетевым детектором
        classes, scores, boxes = model.detect(frame)

        labeled_frame = frame.copy()
        sign_count = 0
        for classid, score, box in zip(classes, scores, boxes):
            if classid not in SIGN_CLASSES:
                continue  # не знак - пропускаем
            sign_count += 1
            label = f'{class_names[classid]} [{score * 100:.2f}%]'
            # Отрисовка ограничивающей рамки
            cv2.rectangle(labeled_frame, box, SIGN_COLOR, 2)
            cv2.putText(labeled_frame, label, (box[0], box[1] - 10), FONT, 0.5, SIGN_COLOR, 2)

        if sign_count:
            print(f'{sign_count} sign(s) detected')
            for classid, score, box in zip(classes, scores, boxes):
                if classid in SIGN_CLASSES:
                    print(f'  - {class_names[classid]}: {score * 100:.2f}%')

        cv2.imshow('Sign detection', labeled_frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC - выход
            break

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f'fps={fps:.1f}')
        if fps < 10:
            print(f'[WARNING] FPS is too low! ({fps:.1f} fps)')
    except KeyboardInterrupt:
        print('\n[INFO] KeyboardInterrupt received, stopping...')
        break

cap.release()
cv2.destroyAllWindows()
