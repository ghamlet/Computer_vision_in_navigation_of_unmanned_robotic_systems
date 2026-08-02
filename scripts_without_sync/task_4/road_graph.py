#!/usr/bin/env python3
"""Дорожный граф карты и построение маршрута через вершины графа.

Граф согласован с eval.py: те же перекрёстки 1..5 и те же межперекрёстковые
связи (j1: j2,j3,j4; j2: j1,j3,j5; j3: j1,j2,j4,j5; j4: j1,j3,j5;
j5: j2,j3,j4). Дороги — сетка по полосам движения из eval.py
(upright_lanes / sideways_lanes): вертикальные дороги x = 44/307/570,
горизонтальные y = 39/283/526. Координаты — в пикселях карты (1204x891).

Маршрут строится как ломаная по этой сети: каждая станция соединяется с
ближайшей точкой дороги, далее путь идёт по рёбрам графа через перекрёстки.
"""

import heapq
import math

# Центры перекрёстков (номера соответствуют тегам eval.py).
JUNCTIONS = {
    1: (307, 39),
    2: (44, 283),
    3: (307, 283),
    4: (570, 283),
    5: (307, 526),
}

# Дороги как ломаные линии (пиксельные координаты).
# Три вертикальные и три горизонтальные дороги образуют сетку.
ROADS = [
    # Вертикальная центральная (j1 -> j3 -> j5).
    [(307, 39), (307, 106), (307, 215), (307, 283), (307, 350), (307, 459), (307, 526)],
    # Вертикальная левая (j1 -> j2 -> низ).
    [(44, 39), (44, 106), (44, 215), (44, 283), (44, 350), (44, 459), (44, 526)],
    # Вертикальная правая (j1 -> j4 -> низ).
    [(570, 39), (570, 106), (570, 215), (570, 283), (570, 350), (570, 459), (570, 526)],
    # Горизонтальная верхняя (через j1).
    [(44, 39), (111, 39), (240, 39), (307, 39), (375, 39), (503, 39), (570, 39)],
    # Горизонтальная средняя (j2 -> j3 -> j4).
    [(44, 283), (111, 283), (240, 283), (307, 283), (375, 283), (503, 283), (570, 283)],
    # Горизонтальная нижняя (через j5).
    [(44, 526), (111, 526), (240, 526), (307, 526), (375, 526), (503, 526), (570, 526)],
]

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# Граф маршрутизации: вершины — точки сетки, рёбра — соседние точки дорог.
_GRAPH = {}
_SEGMENTS = []
for road in ROADS:
    for a, b in zip(road, road[1:]):
        _SEGMENTS.append((a, b))
        _GRAPH.setdefault(a, []).append((b, _dist(a, b)))
        _GRAPH.setdefault(b, []).append((a, _dist(a, b)))


def point_on_segment(p, a, b):
    """Ближайшая к p точка отрезка ab и расстояние до неё."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0:
        return a, _dist(p, a)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    q = (ax + t * dx, ay + t * dy)
    return q, _dist(p, q)


def nearest_road(p):
    """Ближайшая к точке p точка дорожной сети.

    Возвращает (точка, сегмент (a, b), расстояние).
    """
    best = None
    for a, b in _SEGMENTS:
        q, d = point_on_segment(p, a, b)
        if best is None or d < best[2]:
            best = (q, (a, b), d)
    return best


def shortest_path(u, v):
    """Кратчайший путь между узлами графа u и v (список узлов)."""
    if u == v:
        return [u]
    dist = {n: math.inf for n in _GRAPH}
    prev = {n: None for n in _GRAPH}
    dist[u] = 0.0
    pq = [(0.0, u)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        if node == v:
            break
        for nxt, w in _GRAPH[node]:
            nd = d + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                prev[nxt] = node
                heapq.heappush(pq, (nd, nxt))
    if math.isinf(dist[v]):
        return []
    path = []
    node = v
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1]


def _node_cost(u, v):
    """Длина кратчайшего пути по графу между узлами."""
    if u == v:
        return 0.0
    path = shortest_path(u, v)
    return sum(_dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def _access_options(q1, seg1, q2, seg2):
    """Все варианты подключения точек доступа к графу.

    Возвращает список (u1, u2, длина_пути_по_графу).
    """
    if seg1 == seg2:
        return [(None, None, _dist(q1, q2))]
    options = []
    for u1 in seg1:
        for u2 in seg2:
            road_len = _node_cost(u1, u2)
            options.append((u1, u2, road_len))
    return options


def access_distance(p1, seg1, p2, seg2):
    """Длина пути по дорогам от точки p1 (на сегменте seg1) до p2 (seg2)."""
    q1, d1 = point_on_segment(p1, seg1[0], seg1[1])
    q2, d2 = point_on_segment(p2, seg2[0], seg2[1])
    best = math.inf
    for u1, u2, road_len in _access_options(q1, seg1, q2, seg2):
        if u1 is None:
            total = d1 + road_len + d2
        else:
            total = d1 + _dist(q1, u1) + road_len + _dist(u2, q2) + d2
        if total < best:
            best = total
    return best


def polyline_between(p1, seg1, p2, seg2):
    """Ломаная по дорогам от точки p1 до точки p2.

    p1/p2 соединяются с ближайшими точками своих дорог (подъезды), далее
    путь идёт по рёбрам графа. Возвращает список точек-кортежей.
    """
    q1, _ = point_on_segment(p1, seg1[0], seg1[1])
    q2, _ = point_on_segment(p2, seg2[0], seg2[1])
    if seg1 == seg2:
        return [p1, q1, q2, p2]
    # Выбираем вариант подключения с минимальной общей длиной пути.
    best = None
    for u1, u2, road_len in _access_options(q1, seg1, q2, seg2):
        total = _dist(q1, u1) + road_len + _dist(u2, q2)
        if best is None or total < best[0]:
            best = (total, u1, u2)
    _, u1, u2 = best
    pts = [p1, q1]
    pts.extend(shortest_path(u1, u2))
    pts.append(q2)
    pts.append(p2)
    return pts


def _junction_tags(waypoints):
    """Номера перекрёстков, через которые проходит ломаная маршрута."""
    tags = []
    for p in waypoints:
        x, y = int(round(p[0])), int(round(p[1]))
        for tag, (jx, jy) in JUNCTIONS.items():
            if (x - jx) ** 2 + (y - jy) ** 2 <= 2:
                if not tags or tags[-1] != tag:
                    tags.append(tag)
    return tags


def build_route(start_px, station_px_list):
    """Маршрут через все станции по дорожной сети.

    Порядок станций — ближайший сосед по длине дорожного пути от старта.
    Возвращает словарь:
      waypoints  — ломаная маршрута (пиксели),
      stops      — координаты станций в порядке посещения (пиксели),
      order      — индексы станций в исходном списке в порядке посещения,
      junctions  — последовательность перекрёстков на маршруте.
    """
    if not station_px_list:
        return {'waypoints': [], 'stops': [], 'order': [], 'junctions': []}

    start_road = nearest_road(start_px)
    st_roads = [nearest_road(s) for s in station_px_list]

    # Порядок посещения станций (ближайший сосед).
    remaining = list(range(len(st_roads)))
    order = []
    cur_p, cur_seg = start_px, start_road[1]
    while remaining:
        best_i, best_d = None, math.inf
        for i in remaining:
            p, seg, _ = st_roads[i]
            d = access_distance(cur_p, cur_seg, p, seg)
            if d < best_d:
                best_d, best_i = d, i
        order.append(best_i)
        cur_p, cur_seg, _ = st_roads[best_i]
        remaining.remove(best_i)

    # Собираем ломаную.
    waypoints = []
    stops = []
    cur_p, cur_seg = start_px, start_road[1]
    for i in order:
        st_px = station_px_list[i]
        st_access, st_seg, _ = st_roads[i]
        seg = polyline_between(cur_p, cur_seg, st_access, st_seg)
        for p in seg[1:]:
            waypoints.append([int(round(p[0])), int(round(p[1]))])
        waypoints.append([int(round(st_px[0])), int(round(st_px[1]))])
        stops.append([int(round(st_px[0])), int(round(st_px[1]))])
        cur_p, cur_seg = st_px, st_seg

    # Убираем повторяющиеся подряд точки.
    deduped = []
    for p in waypoints:
        if not deduped or deduped[-1] != p:
            deduped.append(p)

    return {
        'waypoints': deduped,
        'stops': stops,
        'order': order,
        'junctions': _junction_tags(deduped),
    }
