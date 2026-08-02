#!/usr/bin/env python3
"""Айкар (бортовая программа беспилотного автомобиля).

По регламенту «Актуализация карты электростанций и построение маршрута по ней»:

1. Показывает базовую карту, клик мышью ставит точку старта (Enter — отправить).
2. Отправляет базовую карту и точку старта визуализатору.
3. Принимает от передатчика три пакета (UDP 9100):
   - stations_new     — новые станции, добавляет к своим данным;
   - stations_status  — изменение статусов станций, обновляет свои данные;
   - end              — завершение разведки.
4. Первые два пакета ретранслирует визуализатору (UDP 9200).
5. После пакета end прокладывает маршрут через все исправные станции
   по дорожному графу (road_graph.py) и отправляет визуализатору
   пакет, кодирующий маршрут.
"""

import base64
import json
import os
import socket
import threading

import cv2
import numpy as np

import road_graph

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CAR_RECV_PORT = 9100            # приём от квадрокоптера (transmitter.py)
VISUALIZER_HOST = '127.0.0.1'
VISUALIZER_PORT = 9200          # отправка визуализатору (visualizer.py)

# Размеры мира в метрах (должны совпадать с transmitter.py и visualizer.py).
WORLD_W_M = 7.975
WORLD_H_M = 6.9

# Станции на базовой карте (должны совпадать с FIXED_STATIONS в
# transmitter.py и visualizer.py).
FIXED_STATIONS = [
    {'id': 'F1', 'x_m': 1.2, 'y_m': 1.5},
    {'id': 'F2', 'x_m': 5.0, 'y_m': 2.3},
    {'id': 'F3', 'x_m': 3.1, 'y_m': 5.4},
]

MAP_EXTS = ('.png', '.jpg', '.jpeg')
JPEG_QUALITY = 85
CHUNK_SIZE = 50000              # размер одного чанка base64 (UDP <= ~64 КБ)

ROUTE_COLOR = (255, 0, 255)     # цвет маршрута на карте

stop_event = threading.Event()
start_point = None              # пиксели на карте
img_w = img_h = 0               # размеры карты

# Список станций, известных айкару: id -> {'x_m', 'y_m', 'status'}.
stations = {}
stations_lock = threading.Lock()


def meters_to_pixels(x_m, y_m):
    """Метры -> пиксели. (0;0) — правый верхний угол, x влево, y вниз."""
    px_per_m_x = img_w / WORLD_W_M
    px_per_m_y = img_h / WORLD_H_M
    px = img_w - x_m * px_per_m_x
    py = y_m * px_per_m_y
    return int(round(px)), int(round(py))


def resolve_map():
    """Возвращает путь к первой картинке карты в этой папке или None."""
    for f in os.listdir(SCRIPT_DIR):
        if f.lower().endswith(MAP_EXTS):
            return os.path.join(SCRIPT_DIR, f)
    return None


def on_mouse(event, x, y, flags, param):
    global start_point
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        print(f'[CAR] Start point: {x}, {y} (Enter — подтвердить, Esc — сброс)')


def send_map(sock, img, start_px):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        print('[CAR][ERROR] Cannot encode map')
        return
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    total = (len(b64) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(total):
        chunk = b64[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        packet = json.dumps({'type': 'map_chunk', 'idx': i, 'total': total, 'data': chunk})
        sock.sendto(packet.encode('utf-8'), (VISUALIZER_HOST, VISUALIZER_PORT))
    packet = json.dumps({'type': 'start', 'x': start_px[0], 'y': start_px[1]})
    sock.sendto(packet.encode('utf-8'), (VISUALIZER_HOST, VISUALIZER_PORT))
    print(f'[CAR] Map sent to visualizer ({total} chunks), start {start_px}')


def handle_stations_new(msg):
    """Добавляет новые станции в свои данные и возвращает список id."""
    added = []
    with stations_lock:
        for s in msg['stations']:
            stations[s['id']] = {
                'x_m': s['x_m'],
                'y_m': s['y_m'],
                'status': 'working',
            }
            added.append(s['id'])
    return added


def handle_stations_status(msg):
    """Обновляет статусы станций по id (если id неизвестен — по координатам)."""
    updated = []
    with stations_lock:
        for u in msg['updates']:
            sid = u['id']
            if sid not in stations:
                sid = match_station_by_coords(u['x_m'], u['y_m'])
            if sid is not None:
                stations[sid]['status'] = u['status']
                updated.append((sid, u['status']))
            else:
                print(f'[CAR][WARN] Unknown station: {u["id"]}')
    return updated


def match_station_by_coords(x_m, y_m, tol=0.4):
    """Ищет станцию по ближайшим метрическим координатам (если id неизвестен)."""
    best_id, best_d = None, tol
    for sid, st in stations.items():
        d = ((st['x_m'] - x_m) ** 2 + (st['y_m'] - y_m) ** 2) ** 0.5
        if d < best_d:
            best_id, best_d = sid, d
    return best_id


def build_and_send_route(out):
    """Строит маршрут через все исправные станции и шлёт его визуализатору."""
    with stations_lock:
        working = [(sid, dict(st)) for sid, st in stations.items()
                   if st['status'] == 'working']

    if not working:
        print('[CAR][WARN] No working stations, route is empty')
        packet = json.dumps({'type': 'route', 'waypoints': [], 'stops': [],
                             'station_ids': [], 'junctions': []})
        out.sendto(packet.encode('utf-8'), (VISUALIZER_HOST, VISUALIZER_PORT))
        return

    origin = start_point
    if origin is None:
        print('[CAR][WARN] Start point not set, using map center')
        origin = (img_w // 2, img_h // 2)

    ids = [sid for sid, _ in working]
    station_px = [meters_to_pixels(st['x_m'], st['y_m']) for _, st in working]
    route = road_graph.build_route(origin, station_px)

    order_ids = [ids[i] for i in route['order']]
    print(f'[CAR] Route through working stations {ids}: order {order_ids}')
    print(f'[CAR] Junctions on route: {route["junctions"]}')

    packet = json.dumps({
        'type': 'route',
        'waypoints': route['waypoints'],
        'stops': route['stops'],
        'station_ids': order_ids,
        'junctions': route['junctions'],
    })
    out.sendto(packet.encode('utf-8'), (VISUALIZER_HOST, VISUALIZER_PORT))
    print('[CAR] Route packet sent to visualizer')


def relay_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', CAR_RECV_PORT))
    sock.settimeout(0.5)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f'[CAR] Relay: udp://127.0.0.1:{CAR_RECV_PORT} -> {VISUALIZER_HOST}:{VISUALIZER_PORT}')
    try:
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            try:
                msg = json.loads(data.decode('utf-8'))
            except (ValueError, UnicodeDecodeError):
                continue

            mtype = msg.get('type')
            print(f'[CAR] <- quad: {mtype}')

            if mtype == 'stations_new':
                added = handle_stations_new(msg)
                print(f'[CAR] New stations: {added}')
                # Первый пакет транслируем визуализатору.
                out.sendto(data, (VISUALIZER_HOST, VISUALIZER_PORT))
                print(f'[CAR] -> viz:   {mtype}')

            elif mtype == 'stations_status':
                updated = handle_stations_status(msg)
                for sid, status in updated:
                    print(f'[CAR] Status: {sid} -> {status}')
                # Второй пакет транслируем визуализатору.
                out.sendto(data, (VISUALIZER_HOST, VISUALIZER_PORT))
                print(f'[CAR] -> viz:   {mtype}')

            elif mtype == 'end':
                # Третий пакет не транслируем: строим маршрут и шлём его.
                print('[CAR] Aerial recon finished, building route')
                build_and_send_route(out)
                stop_event.set()

            # Прочие пакеты игнорируем.
    finally:
        out.close()
        sock.close()


def main():
    global img_w, img_h

    map_path = resolve_map()
    img = cv2.imread(map_path) if map_path else None
    if img is None:
        print(f'[CAR][WARN] Map image not found in {SCRIPT_DIR}, using blank canvas')
        img = np.zeros((960, 1280, 3), dtype=np.uint8)
        cv2.putText(img, 'MAP NOT FOUND - add image to task_4/', (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        print(f'[CAR] Map loaded: {map_path}')

    img_h, img_w = img.shape[:2]

    # Айкар знает станции базовой карты.
    with stations_lock:
        for s in FIXED_STATIONS:
            stations[s['id']] = {
                'x_m': s['x_m'],
                'y_m': s['y_m'],
                'status': 'working',
            }

    relay = threading.Thread(target=relay_loop, daemon=True)
    relay.start()

    sent = False
    while not stop_event.is_set():
        display = img.copy()
        if start_point is not None:
            x, y = start_point
            cv2.drawMarker(display, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
            cv2.circle(display, (x, y), 12, (0, 255, 255), 2)
            cv2.putText(display, 'СТАРТ (Enter - отправить)', (x + 20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display, 'Кликните по точке старта', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if sent:
            cv2.putText(display, 'Карта отправлена визуализатору', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        try:
            cv2.imshow('ai_car', display)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error:
            key = -1

        if key in (27, ord('q')):
            break
        if start_point is not None and key in (13, 10) and not sent:
            send_map(socket.socket(socket.AF_INET, socket.SOCK_DGRAM), img, start_point)
            sent = True

    stop_event.set()
    relay.join(timeout=2)
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    print('[CAR] Exited')


if __name__ == '__main__':
    main()
