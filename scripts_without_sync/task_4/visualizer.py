#!/usr/bin/env python3
"""Визуализатор: принимает карту от айкара и рисует станции и их статусы.

- После получения карты сразу рисует фиксированные станции (зелёные — рабочие).
- Пакет stations_new добавляет новые станции (зелёные).
- Пакет stations_status меняет цвет: working = зелёный, non_working = красный.
- Пакет end закрывает окно.

Система координат карты: (0;0) — правый верхний угол, ось x влево, ось y вниз.
"""

import base64
import json
import socket

import cv2
import numpy as np

VISUALIZER_RECV_PORT = 9200

WORLD_W_M = 7.975
WORLD_H_M = 6.9

FONT = cv2.FONT_HERSHEY_SIMPLEX
WORKING_COLOR = (0, 200, 0)
NON_WORKING_COLOR = (0, 0, 255)
START_COLOR = (0, 255, 255)
ROUTE_COLOR = (255, 0, 255)

# Фиксированные станции (должны совпадать с FIXED_STATIONS в transmitter.py)
FIXED_STATIONS = [
    {'id': 'F1', 'x_m': 1.2, 'y_m': 1.5},
    {'id': 'F2', 'x_m': 5.0, 'y_m': 2.3},
    {'id': 'F3', 'x_m': 3.1, 'y_m': 5.4},
]


def meters_to_pixels(img_w, img_h, x_m, y_m):
    """Метры -> пиксели. (0;0) — правый верхний угол, x влево, y вниз."""
    px_per_m_x = img_w / WORLD_W_M
    px_per_m_y = img_h / WORLD_H_M
    px = img_w - x_m * px_per_m_x
    py = y_m * px_per_m_y
    return int(round(px)), int(round(py))


def draw_station(img, sid, px, status):
    color = WORKING_COLOR if status == 'working' else NON_WORKING_COLOR
    x, y = px
    cv2.circle(img, (x, y), 12, color, 2)
    cv2.circle(img, (x, y), 3, color, -1)
    label = f'{sid}: рабочий' if status == 'working' else f'{sid}: НЕ работает'
    cv2.putText(img, label, (x + 16, y + 4), FONT, 0.6, color, 2)


def match_by_coords(stations, px, tol=30):
    """Ищет станцию по ближайшим пиксельным координатам (если id неизвестен)."""
    best_id, best_d = None, tol
    for sid, st in stations.items():
        dx = st['px'][0] - px[0]
        dy = st['px'][1] - px[1]
        d = (dx * dx + dy * dy) ** 0.5
        if d < best_d:
            best_id, best_d = sid, d
    return best_id


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', VISUALIZER_RECV_PORT))
    sock.settimeout(0.5)
    print(f'[VIZ] Listening on udp://127.0.0.1:{VISUALIZER_RECV_PORT}')

    map_parts = {}
    img = None
    stations = {}
    start_px = None
    route_wps = None
    route_stops = None
    route_ids = None
    route_junctions = None

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                data = None

            if data:
                try:
                    msg = json.loads(data.decode('utf-8'))
                except (ValueError, UnicodeDecodeError):
                    continue
                mtype = msg.get('type')

                if mtype == 'map_chunk':
                    map_parts[msg['idx']] = msg['data']
                    if len(map_parts) == msg['total']:
                        b64 = ''.join(map_parts[i] for i in range(msg['total']))
                        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        h, w = img.shape[:2]
                        for s in FIXED_STATIONS:
                            stations[s['id']] = {
                                'px': meters_to_pixels(w, h, s['x_m'], s['y_m']),
                                'status': 'working',
                            }
                        print(f'[VIZ] Map received ({w}x{h}), fixed: {sorted(stations)}')

                elif mtype == 'start':
                    start_px = (msg['x'], msg['y'])
                    print(f'[VIZ] Start point: {start_px}')

                elif mtype == 'stations_new':
                    if img is None:
                        print('[VIZ][WARN] stations_new before map, ignored')
                    else:
                        h, w = img.shape[:2]
                        for s in msg['stations']:
                            stations[s['id']] = {
                                'px': meters_to_pixels(w, h, s['x_m'], s['y_m']),
                                'status': 'working',
                            }
                        print(f'[VIZ] New stations: {[s["id"] for s in msg["stations"]]}')

                elif mtype == 'stations_status':
                    if img is None:
                        print('[VIZ][WARN] stations_status before map, ignored')
                    else:
                        h, w = img.shape[:2]
                        for u in msg['updates']:
                            sid = u['id']
                            if sid not in stations:
                                sid = match_by_coords(
                                    stations, meters_to_pixels(w, h, u['x_m'], u['y_m']))
                            if sid is not None:
                                stations[sid]['status'] = u['status']
                                print(f'[VIZ] Status: {sid} -> {u["status"]}')
                            else:
                                print(f'[VIZ][WARN] Unknown station: {u["id"]}')

                elif mtype == 'route':
                    route_wps = msg.get('waypoints')
                    route_stops = msg.get('stops')
                    route_ids = msg.get('station_ids')
                    route_junctions = msg.get('junctions')
                    print(f'[VIZ] Route received: {len(route_wps)} waypoints, '
                          f'junctions {route_junctions}')

                elif mtype == 'end':
                    print('[VIZ] End packet received, closing')
                    break

            if img is not None:
                display = img.copy()
                for sid, st in stations.items():
                    draw_station(display, sid, st['px'], st['status'])
                if start_px is not None:
                    x, y = start_px
                    cv2.drawMarker(display, (x, y), START_COLOR, cv2.MARKER_CROSS, 30, 3)
                    cv2.circle(display, (x, y), 12, START_COLOR, 2)
                    cv2.putText(display, 'СТАРТ', (x + 20, y), FONT, 0.7, START_COLOR, 2)
                if route_wps:
                    wps = np.array(route_wps, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(display, [wps], False, ROUTE_COLOR, 3)
                    if route_stops:
                        for i, (sx, sy) in enumerate(route_stops):
                            cv2.drawMarker(display, (sx, sy), ROUTE_COLOR,
                                           cv2.MARKER_DIAMOND, 22, 3)
                            label = str(i + 1)
                            if route_ids and i < len(route_ids):
                                label = f'{i + 1}:{route_ids[i]}'
                            cv2.putText(display, label, (sx + 16, sy + 4),
                                        FONT, 0.6, ROUTE_COLOR, 2)
                if route_junctions:
                    txt = 'Маршрут: через перекрёстки ' + '-'.join(
                        map(str, route_junctions))
                    cv2.putText(display, txt, (30, 60),
                                FONT, 0.8, ROUTE_COLOR, 2)
                try:
                    cv2.imshow('visualizer', display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                except cv2.error:
                    pass
    finally:
        sock.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print('[VIZ] Exited')


if __name__ == '__main__':
    main()
