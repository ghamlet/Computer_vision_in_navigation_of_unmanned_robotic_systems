#!/usr/bin/env python3
"""Поиск красных объектов треугольной формы (знаков) в видимой полосе кадра.

Закрашенные зоны (верх и низ) игнорируются — ищем только в полосе
y = [TOP_FILL_H, BOTTOM_FILL_Y]. Без yolopy и нейросети, чистый OpenCV.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Закрашенные зоны (в пикселях по высоте кадра)
TOP_FILL_H = 300
BOTTOM_FILL_Y = 640
FILL_COLOR = (0, 255, 0)  # BGR

# Параметры поиска красного треугольного знака
SIGN_COLOR = (0, 0, 255)     # обводка знака (красный)
CENTER_COLOR = (255, 255, 255)  # центр треугольника
FONT = cv2.FONT_HERSHEY_SIMPLEX
MIN_AREA = 400               # минимальная площадь контура (пиксели)
RED_RANGES = [               # диапазоны HSV для красного цвета
    ((0, 70, 50), (10, 255, 255)),
    ((170, 70, 50), (180, 255, 255)),
]

# Трекинг центра знака (фильтрует случайные срабатывания)
TRACK_DIST = 100     # макс. смещение центра между соседними кадрами, px
TRACK_MISS_MAX = 5   # через сколько кадров без детекта трек сбрасывается
TRACK_CONFIRM = 3    # сколько кадров подряд нужнно для подтверждения
STOP_DELAY = 0.0     # после скольких секунд без трека выводится СТОП
TRACK_STATE = {'x': None, 'y': None, 'missed': 0, 'confirmed': 0}


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
        # трека нет — начинаем с первого объекта
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


def resolve_source(source):
    if source != 'auto':
        return source
    video = next((f for f in os.listdir(SCRIPT_DIR)
                  if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))), None)
    return os.path.join(SCRIPT_DIR, video) if video else 0


def fill_zones(frame):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, min(TOP_FILL_H, h)), FILL_COLOR, -1)
    cv2.rectangle(frame, (0, min(BOTTOM_FILL_Y, h)), (w, h), FILL_COLOR, -1)


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
    for pts in triangles:
        x, y, w, h = cv2.boundingRect(pts)
        cv2.rectangle(frame, (x, y), (x + w, y + h), SIGN_COLOR, 2)
        cv2.putText(frame, 'Sign: movement prohibition', (x, y - 10),
                    FONT, 0.5, SIGN_COLOR, 2)


def draw_center(frame, cx, cy):
    cv2.drawMarker(frame, (cx, cy), CENTER_COLOR, cv2.MARKER_CROSS, 24, 2)
    cv2.circle(frame, (cx, cy), 4, CENTER_COLOR, -1)


def main():
    parser = argparse.ArgumentParser(description='Video with red triangle sign detection')
    parser.add_argument('--source', '-s', default='auto',
                        help='Video source (file, camera index, URL); default: video from this dir')
    args = parser.parse_args()

    source = ""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video source: {source}')
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    paused = False
    start = time.time()
    had_track = False
    lost_since = None
    stop_printed = False
    try:
        while True:
            if paused:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key in (ord('f'), ord('F')):
                    paused = False
                continue

            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            triangles = find_triangular_red(frame)
            centroids = [triangle_center(pts) for pts in triangles]
            idx = update_tracker(centroids)
            confirmed = idx is not None and TRACK_STATE['confirmed'] >= TRACK_CONFIRM

            if confirmed:
                had_track = True
                lost_since = None
                stop_printed = False
            elif had_track:
                if lost_since is None:
                    lost_since = time.time()
                if not stop_printed and time.time() - lost_since >= STOP_DELAY:
                    print('на взлёт')
                    stop_printed = True

            fill_zones(frame)
            if confirmed:
                draw_signs(frame, [triangles[idx]])
                cx, cy = TRACK_STATE['x'], TRACK_STATE['y']
                draw_center(frame, cx, cy)

            fps = frame_count / (time.time() - start)
            cv2.putText(frame, f'Signs: {len(triangles)}  FPS: {fps:.1f}', (10, 330), FONT, 0.7, (255, 255, 255), 2)
            cv2.imshow('Video (F - pause, ESC/q - stop)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key in (ord('f'), ord('F')):
                paused = True
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
