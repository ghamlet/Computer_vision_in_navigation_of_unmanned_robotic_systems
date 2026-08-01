from typing import List, Dict, Tuple
from dataclasses import dataclass
from vehicle.road_graph import RoadGraph


@dataclass
class Station:
    id: str
    x: int
    y: int
    status: str
    description: str
    is_base: bool


class RoutePlanner:
    def __init__(self):
        self.graph = RoadGraph()
        self._distance_cache: Dict[Tuple[str, str], Tuple[int, List]] = {}

    def plan_route(self, stations: List[Station]) -> List[str]:
        functional = [s for s in stations if s.status == "functional"]
        
        if len(functional) <= 1:
            return [s.id for s in functional]

        dist_matrix = self._compute_distance_matrix(functional)
        order = self._solve_tsp_nearest_neighbor(functional, dist_matrix)
        return [s.id for s in order]

    def _compute_distance_matrix(self, stations: List[Station]) -> Dict[Tuple[str, str], int]:
        matrix = {}
        for i, s1 in enumerate(stations):
            for j, s2 in enumerate(stations):
                if i == j:
                    matrix[(s1.id, s2.id)] = 0
                elif (s1.id, s2.id) not in matrix:
                    dist, _ = self.graph.distance_between_stations(s1.x, s1.y, s2.x, s2.y)
                    matrix[(s1.id, s2.id)] = dist
                    matrix[(s2.id, s1.id)] = dist
        return matrix

    def _solve_tsp_nearest_neighbor(self, stations: List[Station], dist_matrix: Dict[Tuple[str, str], int]) -> List[Station]:
        unvisited = set(s.id for s in stations)
        current = stations[0].id
        unvisited.remove(current)
        path = [stations[0]]

        station_map = {s.id: s for s in stations}

        while unvisited:
            nearest = min(unvisited, key=lambda sid: dist_matrix.get((current, sid), float('inf')))
            path.append(station_map[nearest])
            unvisited.remove(nearest)
            current = nearest

        return path

    def get_path_between(self, station_a: Station, station_b: Station) -> List:
        dist, path = self.graph.distance_between_stations(station_a.x, station_a.y, station_b.x, station_b.y)
        return path