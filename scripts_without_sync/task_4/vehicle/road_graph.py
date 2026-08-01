from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class JunctionPoint:
    tag: Any
    neighbors: List[Tuple['JunctionPoint', int]] = None

    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = []

    def link(self, point: 'JunctionPoint', dist: int):
        self.neighbors.append((point, dist))

    def __hash__(self):
        return hash(self.tag)

    def __eq__(self, other):
        if isinstance(other, JunctionPoint):
            return self.tag == other.tag
        return False


class RoadGraph:
    def __init__(self):
        self.junctions: Dict[Any, JunctionPoint] = {}
        self._build_graph()
        self._build_inter_junction_links()
        self.lane_width = 27

        self.upright_lanes = {
            'up_left': [44, 106, 215],
            'up_center': [307, 106, 215],
            'up_right': [570, 106, 215],
            'down_left': [44, 350, 459],
            'down_center': [307, 350, 459],
            'down_right': [570, 350, 459]
        }

        self.sideways_lanes = {
            'west_up': [39, 111, 240],
            'west_mid': [283, 111, 240],
            'west_low': [526, 111, 240],
            'east_up': [39, 375, 503],
            'east_mid': [283, 375, 503],
            'east_low': [526, 375, 503]
        }

    def _build_graph(self):
        j1_entry_r = JunctionPoint(1)
        j1_entry_b = JunctionPoint(1)
        j1_entry_l = JunctionPoint(1)

        j1_exit_r = JunctionPoint('j1_exit_r')
        j1_exit_b = JunctionPoint('j1_exit_b')
        j1_exit_l = JunctionPoint('j1_exit_l')

        j1_entry_r.link(j1_exit_b, 14)
        j1_entry_r.link(j1_exit_l, 13)

        j1_entry_l.link(j1_exit_b, 9)
        j1_entry_l.link(j1_exit_r, 13)

        j1_entry_b.link(j1_exit_r, 9)
        j1_entry_b.link(j1_exit_l, 14)

        j2_entry_r = JunctionPoint(2)
        j2_entry_b = JunctionPoint(2)
        j2_entry_t = JunctionPoint(2)

        j2_exit_r = JunctionPoint('j2_exit_r')
        j2_exit_b = JunctionPoint('j2_exit_b')
        j2_exit_t = JunctionPoint('j2_exit_t')

        j2_entry_r.link(j2_exit_b, 14)
        j2_entry_r.link(j2_exit_t, 9)

        j2_entry_b.link(j2_exit_t, 13)
        j2_entry_b.link(j2_exit_r, 9)

        j2_entry_t.link(j2_exit_b, 13)
        j2_entry_t.link(j2_exit_r, 14)

        j3_entry_l = JunctionPoint(3)
        j3_entry_r = JunctionPoint(3)
        j3_entry_b = JunctionPoint(3)
        j3_entry_t = JunctionPoint(3)

        j3_exit_l = JunctionPoint('j3_exit_l')
        j3_exit_r = JunctionPoint('j3_exit_r')
        j3_exit_b = JunctionPoint('j3_exit_b')
        j3_exit_t = JunctionPoint('j3_exit_t')

        j3_entry_l.link(j3_exit_r, 13)
        j3_entry_l.link(j3_exit_t, 14)
        j3_entry_l.link(j3_exit_b, 9)

        j3_entry_r.link(j3_exit_l, 13)
        j3_entry_r.link(j3_exit_t, 9)
        j3_entry_r.link(j3_exit_b, 14)

        j3_entry_b.link(j3_exit_r, 9)
        j3_entry_b.link(j3_exit_t, 13)
        j3_entry_b.link(j3_exit_l, 14)

        j3_entry_t.link(j3_exit_r, 14)
        j3_entry_t.link(j3_exit_l, 9)
        j3_entry_t.link(j3_exit_b, 13)

        j4_entry_l = JunctionPoint(4)
        j4_entry_b = JunctionPoint(4)
        j4_entry_t = JunctionPoint(4)

        j4_exit_l = JunctionPoint('j4_exit_l')
        j4_exit_b = JunctionPoint('j4_exit_b')
        j4_exit_t = JunctionPoint('j4_exit_t')

        j4_entry_l.link(j4_exit_b, 9)
        j4_entry_l.link(j4_exit_t, 14)

        j4_entry_b.link(j4_exit_t, 13)
        j4_entry_b.link(j4_exit_l, 14)

        j4_entry_t.link(j4_exit_b, 13)
        j4_entry_t.link(j4_exit_l, 9)

        j5_entry_l = JunctionPoint(5)
        j5_entry_r = JunctionPoint(5)
        j5_entry_t = JunctionPoint(5)

        j5_exit_l = JunctionPoint('j5_exit_l')
        j5_exit_r = JunctionPoint('j5_exit_r')
        j5_exit_t = JunctionPoint('j5_exit_t')

        j5_entry_l.link(j5_exit_r, 13)
        j5_entry_l.link(j5_exit_t, 14)

        j5_entry_r.link(j5_exit_l, 13)
        j5_entry_r.link(j5_exit_t, 9)

        j5_entry_t.link(j5_exit_r, 14)
        j5_entry_t.link(j5_exit_l, 9)

        for jp in [j1_entry_r, j1_entry_b, j1_entry_l, j1_exit_r, j1_exit_b, j1_exit_l,
                   j2_entry_r, j2_entry_b, j2_entry_t, j2_exit_r, j2_exit_b, j2_exit_t,
                   j3_entry_l, j3_entry_r, j3_entry_b, j3_entry_t, j3_exit_l, j3_exit_r, j3_exit_b, j3_exit_t,
                   j4_entry_l, j4_entry_b, j4_entry_t, j4_exit_l, j4_exit_b, j4_exit_t,
                   j5_entry_l, j5_entry_r, j5_entry_t, j5_exit_l, j5_exit_r, j5_exit_t]:
            self.junctions[jp.tag] = jp

    def _build_inter_junction_links(self):
        j1_exit_l = self.junctions['j1_exit_l']
        j1_exit_r = self.junctions['j1_exit_r']
        j1_exit_b = self.junctions['j1_exit_b']

        j2_exit_t = self.junctions['j2_exit_t']
        j2_exit_r = self.junctions['j2_exit_r']
        j2_exit_b = self.junctions['j2_exit_b']

        j3_exit_l = self.junctions['j3_exit_l']
        j3_exit_r = self.junctions['j3_exit_r']
        j3_exit_b = self.junctions['j3_exit_b']
        j3_exit_t = self.junctions['j3_exit_t']

        j4_exit_l = self.junctions['j4_exit_l']
        j4_exit_t = self.junctions['j4_exit_t']
        j4_exit_b = self.junctions['j4_exit_b']

        j5_exit_l = self.junctions['j5_exit_l']
        j5_exit_r = self.junctions['j5_exit_r']
        j5_exit_t = self.junctions['j5_exit_t']

        j2_entry_t = self.junctions[2]
        j2_entry_r = self.junctions[2]
        j2_entry_b = self.junctions[2]

        j3_entry_l = self.junctions[3]
        j3_entry_r = self.junctions[3]
        j3_entry_b = self.junctions[3]
        j3_entry_t = self.junctions[3]

        j4_entry_l = self.junctions[4]
        j4_entry_b = self.junctions[4]
        j4_entry_t = self.junctions[4]

        j5_entry_l = self.junctions[5]
        j5_entry_r = self.junctions[5]
        j5_entry_t = self.junctions[5]

        j1_entry_l = self.junctions[1]
        j1_entry_r = self.junctions[1]
        j1_entry_b = self.junctions[1]

        j1_exit_l.link(j2_entry_t, 33)
        j1_exit_r.link(j4_entry_t, 28)
        j1_exit_b.link(j3_entry_t, 10)

        j2_exit_t.link(j1_entry_l, 28)
        j2_exit_r.link(j3_entry_l, 12)
        j2_exit_b.link(j5_entry_l, 33)

        j3_exit_l.link(j2_entry_r, 12)
        j3_exit_r.link(j4_entry_l, 12)
        j3_exit_b.link(j5_entry_t, 10)
        j3_exit_t.link(j1_entry_b, 10)

        j4_exit_l.link(j3_entry_r, 12)
        j4_exit_t.link(j1_entry_r, 33)
        j4_exit_b.link(j5_entry_r, 28)

        j5_exit_l.link(j2_entry_b, 28)
        j5_exit_r.link(j4_entry_b, 33)
        j5_exit_t.link(j3_entry_b, 10)

    def get_entry_points(self) -> List[JunctionPoint]:
        return [jp for tag, jp in self.junctions.items() if isinstance(tag, int)]

    def get_junction_by_tag(self, tag: Any) -> Optional[JunctionPoint]:
        return self.junctions.get(tag)

    def find_nearest_entry(self, x: int, y: int) -> Optional[JunctionPoint]:
        best_entry = None
        best_dist = float('inf')

        for lane_id, (lx, l_start, l_end) in self.upright_lanes.items():
            if lx - self.lane_width <= x <= lx + self.lane_width and l_start <= y <= l_end:
                if x > lx:
                    heading = 'north'
                else:
                    heading = 'south'

                entry = self._lane_to_entry(lane_id, heading)
                if entry:
                    dist = abs(x - lx) + abs(y - (l_start + l_end) // 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_entry = entry

        for lane_id, (ly, l_start, l_end) in self.sideways_lanes.items():
            if ly - self.lane_width <= y <= ly + self.lane_width and l_start <= x <= l_end:
                if y > ly:
                    heading = 'east'
                else:
                    heading = 'west'

                entry = self._lane_to_entry(lane_id, heading)
                if entry:
                    dist = abs(y - ly) + abs(x - (l_start + l_end) // 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_entry = entry

        return best_entry

    def _lane_to_entry(self, lane_id: str, heading: str) -> Optional[JunctionPoint]:
        mapping = {
            ('up_left', 'north'): 1,
            ('up_left', 'south'): 2,
            ('up_center', 'north'): 1,
            ('up_center', 'south'): 3,
            ('up_right', 'north'): 1,
            ('up_right', 'south'): 4,
            ('down_left', 'north'): 2,
            ('down_left', 'south'): 5,
            ('down_center', 'north'): 3,
            ('down_center', 'south'): 5,
            ('down_right', 'north'): 4,
            ('down_right', 'south'): 5,
            ('west_up', 'east'): 1,
            ('west_up', 'west'): 2,
            ('west_mid', 'east'): 3,
            ('west_mid', 'west'): 2,
            ('west_low', 'east'): 5,
            ('west_low', 'west'): 2,
            ('east_up', 'east'): 4,
            ('east_up', 'west'): 1,
            ('east_mid', 'east'): 4,
            ('east_mid', 'west'): 3,
            ('east_low', 'east'): 4,
            ('east_low', 'west'): 5,
        }
        tag = mapping.get((lane_id, heading))
        if tag:
            return self.junctions.get(tag)
        return None

    def shortest_path(self, start: JunctionPoint, goal: JunctionPoint) -> Tuple[int, List[Any]]:
        from collections import deque

        queue = deque([(start, 0, [start.tag])])
        visited = {start: 0}
        best_path = None
        best_dist = float('inf')

        while queue:
            current, dist, path = queue.popleft()

            if current == goal:
                if dist < best_dist:
                    best_dist = dist
                    best_path = path
                continue

            for neighbor, edge_dist in current.neighbors:
                new_dist = dist + edge_dist
                if neighbor not in visited or new_dist < visited[neighbor]:
                    visited[neighbor] = new_dist
                    queue.append((neighbor, new_dist, path + [neighbor.tag]))

        return (best_dist, best_path) if best_path else (float('inf'), [])

    def distance_between_stations(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, List[Any]]:
        entry1 = self.find_nearest_entry(x1, y1)
        entry2 = self.find_nearest_entry(x2, y2)

        if not entry1 or not entry2:
            return (float('inf'), [])

        return self.shortest_path(entry1, entry2)