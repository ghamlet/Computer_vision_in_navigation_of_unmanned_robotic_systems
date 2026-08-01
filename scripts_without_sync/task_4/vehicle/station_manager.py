from typing import List, Dict, Optional
from dataclasses import dataclass
from common.protocol import StationInfo, BaseMapPacket, NewStationPacket, StatusUpdatePacket


@dataclass
class Station:
    id: str
    x: int
    y: int
    status: str
    description: str
    is_base: bool


class StationManager:
    def __init__(self):
        self.stations: Dict[str, Station] = {
            "base_0": Station("base_0", 200, 300, "unknown", "Базовая ПС 1", True),
            "base_1": Station("base_1", 400, 300, "unknown", "Базовая ПС 2", True),
        }
        self.discovered_counter = 0
        self.base_station_ids = {"base_0", "base_1"}

    def get_base_map_packet(self) -> BaseMapPacket:
        station_infos = [
            StationInfo(
                id=s.id,
                x=s.x,
                y=s.y,
                status=s.status,
                description=s.description,
                is_base=s.is_base
            )
            for s in self.stations.values()
        ]
        return BaseMapPacket(stations=station_infos)

    def handle_new_station(self, packet: NewStationPacket) -> Station:
        if packet.is_base:
            if packet.id in self.stations:
                station = self.stations[packet.id]
                station.x = packet.x
                station.y = packet.y
                station.description = packet.description
                station.status = packet.status
                return station
            else:
                station = Station(
                    id=packet.id,
                    x=packet.x,
                    y=packet.y,
                    status=packet.status,
                    description=packet.description,
                    is_base=True
                )
                self.stations[packet.id] = station
                self.base_station_ids.add(packet.id)
                return station
        else:
            self.discovered_counter += 1
            station_id = packet.id or f"disc_{self.discovered_counter:03d}"
            station = Station(
                id=station_id,
                x=packet.x,
                y=packet.y,
                status=packet.status,
                description=packet.description,
                is_base=False
            )
            self.stations[station_id] = station
            return station

    def handle_status_update(self, packet: StatusUpdatePacket):
        for update in packet.updates:
            if update.id in self.stations:
                self.stations[update.id].status = update.status

    def get_functional_stations(self) -> List[Station]:
        return [s for s in self.stations.values() if s.status == "functional"]

    def get_all_stations(self) -> List[Station]:
        return list(self.stations.values())

    def get_station(self, station_id: str) -> Optional[Station]:
        return self.stations.get(station_id)

    def get_base_stations(self) -> List[Station]:
        return [s for s in self.stations.values() if s.is_base]