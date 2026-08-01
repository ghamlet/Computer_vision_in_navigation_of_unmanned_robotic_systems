import cv2
import numpy as np
import asyncio
import threading
import queue
from typing import Dict, List, Optional
from dataclasses import dataclass

from common.protocol import (
    BasePacket, BaseMapPacket, NewStationPacket, StatusUpdatePacket, 
    RoutePacket, decode_packet, encode_packet
)
from common.network import TCPServer


@dataclass
class Station:
    id: str
    x: int
    y: int
    status: str
    description: str
    is_base: bool


COLORS = {
    "unknown": (0, 255, 255),
    "functional": (0, 255, 0),
    "broken": (0, 0, 255),
    "discovered": (255, 0, 255),
}
RADIUS = 15
ROUTE_COLOR = (255, 255, 0)
ROUTE_THICKNESS = 2


class Visualizer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8002, map_path: str = "image.jpg"):
        self.host = host
        self.port = port
        self.map_path = map_path
        self.server = TCPServer(host, port, self.handle_client)
        self.stations: Dict[str, Station] = {}
        self.route_order: List[str] = []
        self.original_map: Optional[np.ndarray] = None
        self.working_map: Optional[np.ndarray] = None
        self.refresh_flag = True
        self.running = True
        self._client_writer = None
        
        # Right-click dialog state
        self.dialog_active = False
        self.dialog_x = 0
        self.dialog_y = 0
        self.dialog_text = ""
        self.dialog_cursor_pos = 0
        
        # Thread-safe queues for communication between threads
        self._send_queue: queue.Queue = queue.Queue()
        self._recv_queue: queue.Queue = queue.Queue()

    def load_map(self):
        self.original_map = cv2.imread(self.map_path)
        if self.original_map is None:
            self.original_map = np.zeros((891, 1204, 3), dtype=np.uint8)
        self.working_map = self.original_map.copy()

    def start(self):
        self.load_map()
        
        cv2.namedWindow("Map")
        cv2.setMouseCallback("Map", self.on_mouse)

        # Run network loop in background thread
        network_thread = threading.Thread(target=self._run_network_thread, daemon=True)
        network_thread.start()

        # Run display loop in main thread (required for OpenCV/Qt)
        self.display_loop()

    def _run_network_thread(self):
        """Run the asyncio network loop in a dedicated thread."""
        asyncio.run(self.network_loop())

    async def network_loop(self):
        await self.server.start()
        print("[Visualizer] Network server started")
        
        # Process send queue periodically
        while self.running:
            # Handle outgoing packets
            try:
                while True:
                    packet = self._send_queue.get_nowait()
                    if self._client_writer:
                        self._client_writer.write(encode_packet(packet))
                        await self._client_writer.drain()
                        print(f"[Visualizer] Sent: {packet.type}")
            except queue.Empty:
                pass
            
            await asyncio.sleep(0.05)

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        print(f"[Visualizer] Vehicle connected from {addr}")
        self._client_writer = writer
        
        try:
            while self.running:
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, 'big')
                data = await reader.readexactly(length)
                packet = decode_packet(data)
                # Put received packet in queue for display thread
                self._recv_queue.put(packet)
        except asyncio.IncompleteReadError:
            print("[Visualizer] Vehicle disconnected")
        except Exception as e:
            print(f"[Visualizer] Network error: {e}")
        finally:
            self._client_writer = None
            writer.close()
            await writer.wait_closed()

    def handle_packet(self, packet: BasePacket):
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
        if event == cv2.EVENT_RBUTTONDOWN and not self.dialog_active:
            self.dialog_active = True
            self.dialog_x = x
            self.dialog_y = y
            self.dialog_text = ""
            self.dialog_cursor_pos = 0
            print(f"[Visualizer] Right-click at ({x}, {y}) - enter description, press Enter to confirm, Esc to cancel")

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

        # Draw dialog if active
        if self.dialog_active:
            self.draw_dialog()

        cv2.putText(self.working_map, "PКМ - добавить базовую станцию", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_dialog(self):
        # Draw semi-transparent overlay
        overlay = self.working_map.copy()
        cv2.rectangle(overlay, (100, 300), (1100, 500), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.working_map, 0.3, 0, self.working_map)
        
        # Draw dialog box
        cv2.rectangle(self.working_map, (100, 300), (1100, 500), (255, 255, 255), 2)
        cv2.rectangle(self.working_map, (102, 302), (1098, 498), (50, 50, 50), -1)
        
        # Title
        cv2.putText(self.working_map, f"Новая базовая станция: ({self.dialog_x}, {self.dialog_y})", 
                   (120, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(self.working_map, "Введите описание:", 
                   (120, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Input field
        cv2.rectangle(self.working_map, (120, 400), (1080, 440), (100, 100, 100), -1)
        cv2.rectangle(self.working_map, (120, 400), (1080, 440), (255, 255, 255), 1)
        
        # Text with cursor
        display_text = self.dialog_text
        import time
        if int(time.time() * 2) % 2 == 0:
            display_text = display_text[:self.dialog_cursor_pos] + "|" + display_text[self.dialog_cursor_pos:]
        else:
            display_text = display_text[:self.dialog_cursor_pos] + " " + display_text[self.dialog_cursor_pos:]
            
        cv2.putText(self.working_map, display_text, 
                   (130, 428), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(self.working_map, "Enter - подтвердить | Esc - отмена", 
                   (120, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    def handle_key(self, key):
        if self.dialog_active:
            if key == 13:  # Enter
                if self.dialog_text.strip():
                    station_id = f"base_{len([s for s in self.stations.values() if s.is_base])}"
                    packet = NewStationPacket(
                        id=station_id, x=self.dialog_x, y=self.dialog_y, status="unknown",
                        description=self.dialog_text.strip(), is_base=True
                    )
                    self._send_queue.put(packet)
                self.dialog_active = False
            elif key == 27:  # Esc
                self.dialog_active = False
            elif key == 8:  # Backspace
                if self.dialog_cursor_pos > 0:
                    self.dialog_text = self.dialog_text[:self.dialog_cursor_pos-1] + self.dialog_text[self.dialog_cursor_pos:]
                    self.dialog_cursor_pos -= 1
            elif 32 <= key <= 126:  # Printable characters
                char = chr(key)
                self.dialog_text = self.dialog_text[:self.dialog_cursor_pos] + char + self.dialog_text[self.dialog_cursor_pos:]
                self.dialog_cursor_pos += 1

    def display_loop(self):
        while self.running:
            # Process received packets from network thread
            try:
                while True:
                    packet = self._recv_queue.get_nowait()
                    self.handle_packet(packet)
            except queue.Empty:
                pass

            if self.refresh_flag:
                self.draw_map()
                self.refresh_flag = False

            if self.working_map is not None:
                cv2.imshow("Map", self.working_map)

            key = cv2.waitKey(50) & 0xFF
            if key == 27 and not self.dialog_active:  # ESC to quit
                break
            elif key != 255:  # Any key pressed
                self.handle_key(key)

        self.running = False
        cv2.destroyAllWindows()


def main():
    visualizer = Visualizer()
    visualizer.start()


if __name__ == "__main__":
    main()