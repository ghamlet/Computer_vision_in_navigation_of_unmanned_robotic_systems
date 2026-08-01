import cv2
import numpy as np
import asyncio
import threading
import tkinter as tk
from tkinter import simpledialog
from typing import Dict, List, Optional
from dataclasses import dataclass

from common.protocol import (
    BasePacket, BaseMapPacket, NewStationPacket, StatusUpdatePacket, 
    RoutePacket, decode_packet
)
from common.network import TCPClient


@dataclass
class Station:
    id: str
    x: int
    y: int
    status: str
    description: str
    is_base: bool


COLORS = {
    "unknown": (0, 255, 255),      # yellow
    "functional": (0, 255, 0),     # green
    "broken": (0, 0, 255),         # red
    "discovered": (255, 0, 255),   # magenta
}
RADIUS = 15
ROUTE_COLOR = (255, 255, 0)  # cyan
ROUTE_THICKNESS = 2


class Visualizer:
    def __init__(self, vehicle_host: str = "localhost", vehicle_port: int = 8002, map_path: str = "image.jpg"):
        self.vehicle_host = vehicle_host
        self.vehicle_port = vehicle_port
        self.map_path = map_path
        self.client = TCPClient(vehicle_host, vehicle_port)
        self.stations: Dict[str, Station] = {}
        self.route_order: List[str] = []
        self.original_map: Optional[np.ndarray] = None
        self.working_map: Optional[np.ndarray] = None
        self.refresh_flag = True
        self.lock = threading.Lock()
        self.running = True
        self._description_dialog = None

    def load_map(self):
        self.original_map = cv2.imread(self.map_path)
        if self.original_map is None:
            self.original_map = np.zeros((891, 1204, 3), dtype=np.uint8)
        self.working_map = self.original_map.copy()

    def start(self):
        self.load_map()
        
        cv2.namedWindow("Map")
        cv2.setMouseCallback("Map", self.on_mouse)

        network_thread = threading.Thread(target=self.run_network_loop, daemon=True)
        network_thread.start()

        self.display_loop()

    def run_network_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.network_loop())

    async def network_loop(self):
        connected = await self.client.connect()
        if not connected:
            print("[Visualizer] Failed to connect to vehicle")
            return

        try:
            while self.running:
                packet = await self.client.receive()
                self.handle_packet(packet)
        except Exception as e:
            print(f"[Visualizer] Network error: {e}")

    def handle_packet(self, packet: BasePacket):
        with self.lock:
            if packet.type == "base_map":
                self.stations.clear()
                for si in packet.stations:
                    self.stations[si.id] = Station(
                        id=si.id, x=si.x, y=si.y, status=si.status,
                        description=si.description, is_base=si.is_base
                    )
                print(f"[Visualizer] Received base map with {len(self.stations)} stations")
                self.refresh_flag = True

            elif packet.type == "new_station":
                station = Station(
                    id=packet.id, x=packet.x, y=packet.y, status=packet.status,
                    description=packet.description, is_base=packet.is_base
                )
                self.stations[packet.id] = station
                print(f"[Visualizer] New station: {station.id} at ({station.x}, {station.y})")
                self.refresh_flag = True

            elif packet.type == "status_update":
                for update in packet.updates:
                    if update.id in self.stations:
                        self.stations[update.id].status = update.status
                print(f"[Visualizer] Status updates applied")
                self.refresh_flag = True

            elif packet.type == "route":
                self.route_order = packet.station_order
                print(f"[Visualizer] Route received: {self.route_order}")
                self.refresh_flag = True

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.add_base_station_dialog(x, y)

    def add_base_station_dialog(self, x: int, y: int):
        def get_description():
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            desc = simpledialog.askstring("Новая базовая станция", 
                f"Координаты: ({x}, {y})\nВведите описание:", parent=root)
            root.destroy()
            if desc:
                asyncio.run_coroutine_threadsafe(
                    self.send_new_base_station(x, y, desc), 
                    asyncio.get_event_loop()
                )

        threading.Thread(target=get_description, daemon=True).start()

    async def send_new_base_station(self, x: int, y: int, description: str):
        station_id = f"base_{len([s for s in self.stations.values() if s.is_base])}"
        packet = NewStationPacket(
            id=station_id, x=x, y=y, status="unknown",
            description=description, is_base=True
        )
        await self.client.send(packet)
        print(f"[Visualizer] Sent new base station: {station_id}")

    def draw_map(self):
        if self.original_map is None:
            return

        self.working_map = self.original_map.copy()

        for station in self.stations.values():
            color = COLORS.get(station.status, (128, 128, 128))
            cv2.circle(self.working_map, (station.x, station.y), RADIUS, color, -1)
            cv2.circle(self.working_map, (station.x, station.y), RADIUS, (255, 255, 255), 2)
            
            label = f"{station.id}"
            if station.description:
                label += f": {station.description}"
            cv2.putText(self.working_map, label, (station.x - 40, station.y - RADIUS - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if len(self.route_order) >= 2:
            for i in range(len(self.route_order) - 1):
                s1 = self.stations.get(self.route_order[i])
                s2 = self.stations.get(self.route_order[i + 1])
                if s1 and s2:
                    cv2.line(self.working_map, (s1.x, s1.y), (s2.x, s2.y), ROUTE_COLOR, ROUTE_THICKNESS)
                    cv2.arrowedLine(self.working_map, (s1.x, s1.y), (s2.x, s2.y), ROUTE_COLOR, ROUTE_THICKNESS, tipLength=0.05)

        cv2.putText(self.working_map, "PКМ - добавить базовую станцию", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def display_loop(self):
        while self.running:
            with self.lock:
                if self.refresh_flag:
                    self.draw_map()
                    self.refresh_flag = False

            if self.working_map is not None:
                cv2.imshow("Map", self.working_map)

            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # ESC
                break

        self.running = False
        cv2.destroyAllWindows()


def main():
    visualizer = Visualizer()
    visualizer.start()


if __name__ == "__main__":
    main()