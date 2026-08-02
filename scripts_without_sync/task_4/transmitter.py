#!/usr/bin/env python3
"""Передатчик (с квадрокоптера): шлёт айкару 3 пакета с интервалом 5 секунд.

Пакет 1 — новые станции со случайными координатами (метры).
Пакет 2 — существующие станции с изменённым статусом (working -> non_working).
Пакет 3 — завершение программы.

Система координат карты: (0;0) — правый верхний угол, ось x влево, ось y вниз.
Координаты станций генерируются в метрах в пределах карты.
"""

import json
import random
import socket
import time

AI_CAR_HOST = '127.0.0.1'
AI_CAR_PORT = 9100

WORLD_W_M = 7.975
WORLD_H_M = 6.9

INTERVAL = 5.0
NEW_STATIONS_NUM = 4

# Фиксированные станции (должны совпадать с FIXED_STATIONS в visualizer.py)
FIXED_STATIONS = [
    {'id': 'F1', 'x_m': 1.2, 'y_m': 1.5},
    {'id': 'F2', 'x_m': 5.0, 'y_m': 2.3},
    {'id': 'F3', 'x_m': 3.1, 'y_m': 5.4},
]


def send(sock, msg):
    text = json.dumps(msg, ensure_ascii=False)
    sock.sendto(text.encode('utf-8'), (AI_CAR_HOST, AI_CAR_PORT))
    print(f'[QUAD] -> {AI_CAR_HOST}:{AI_CAR_PORT}: {text[:160]}')


def random_station(idx):
    return {
        'id': f'S{idx}',
        'x_m': round(random.uniform(0, WORLD_W_M), 2),
        'y_m': round(random.uniform(0, WORLD_H_M), 2),
    }


def make_packet1():
    return {
        'type': 'stations_new',
        'stations': [random_station(i) for i in range(1, NEW_STATIONS_NUM + 1)],
    }


def make_packet2(stations):
    chosen = random.sample(stations, max(1, len(stations) // 2))
    updates = []
    for s in chosen:
        updates.append({
            'id': s['id'],
            'x_m': s['x_m'],
            'y_m': s['y_m'],
            'status': 'non_working',
        })
    return {'type': 'stations_status', 'updates': updates}


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        time.sleep(1.0)  # ждём, пока айкар поднимет приёмник

        packet1 = make_packet1()
        send(sock, packet1)
        time.sleep(INTERVAL)

        existing = FIXED_STATIONS + packet1['stations']
        packet2 = make_packet2(existing)
        send(sock, packet2)
        time.sleep(INTERVAL)

        send(sock, {'type': 'end'})
    finally:
        sock.close()
    print('[QUAD] Done')


if __name__ == '__main__':
    main()
