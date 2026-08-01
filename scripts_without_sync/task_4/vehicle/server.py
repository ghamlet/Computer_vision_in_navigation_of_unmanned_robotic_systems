import asyncio
from typing import Optional
from common.protocol import (
    BasePacket, NewStationPacket, StatusUpdatePacket, 
    ReconCompletePacket, RoutePacket, decode_packet, encode_packet
)
from common.network import TCPClient
from vehicle.station_manager import StationManager
from vehicle.route_planner import RoutePlanner


class VehicleServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8001, visualizer_host: str = "localhost", visualizer_port: int = 8002):
        self.host = host
        self.port = port
        self.visualizer_host = visualizer_host
        self.visualizer_port = visualizer_port
        self.station_manager = StationManager()
        self.route_planner = RoutePlanner()
        self.visualizer_client = TCPClient(visualizer_host, visualizer_port)
        self.server = None
        self._recon_received = False
        self._visualizer_connected = False

    async def _try_connect_visualizer(self):
        while not self._visualizer_connected:
            try:
                connected = await self.visualizer_client.connect()
                if connected:
                    self._visualizer_connected = True
                    base_map_packet = self.station_manager.get_base_map_packet()
                    await self.visualizer_client.send(base_map_packet)
                    print("[Vehicle] Sent base map to visualizer")
                    break
            except Exception as e:
                print(f"[Vehicle] Visualizer connection failed, retrying in 2s: {e}")
                await asyncio.sleep(2)

    async def start(self):
        asyncio.create_task(self._try_connect_visualizer())

        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addr = self.server.sockets[0].getsockname()
        print(f"[Vehicle] Server listening on {addr[0]}:{addr[1]}")

    async def _send_to_visualizer(self, packet: BasePacket):
        if self._visualizer_connected:
            try:
                await self.visualizer_client.send(packet)
            except Exception as e:
                print(f"[Vehicle] Failed to send to visualizer: {e}")
                self._visualizer_connected = False
                asyncio.create_task(self._try_connect_visualizer())

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        print(f"[Vehicle] Transmitter connected from {addr}")

        try:
            while True:
                length_bytes = await reader.readexactly(4)
                length = int.from_bytes(length_bytes, 'big')
                data = await reader.readexactly(length)
                packet = decode_packet(data)

                print(f"[Vehicle] Received packet: {packet.type}")

                if packet.type == "new_station":
                    station = self.station_manager.handle_new_station(packet)
                    print(f"[Vehicle] New station: {station.id} at ({station.x}, {station.y}) status={station.status} is_base={station.is_base}")
                    await self._send_to_visualizer(packet)

                elif packet.type == "status_update":
                    self.station_manager.handle_status_update(packet)
                    print(f"[Vehicle] Status updates: {[(u.id, u.status) for u in packet.updates]}")
                    await self._send_to_visualizer(packet)

                elif packet.type == "recon_complete":
                    print("[Vehicle] Recon complete received, planning route...")
                    self._recon_received = True
                    functional = self.station_manager.get_functional_stations()
                    if len(functional) >= 2:
                        station_order = self.route_planner.plan_route(functional)
                    else:
                        station_order = [s.id for s in functional]
                    route_packet = RoutePacket(station_order=station_order)
                    await self._send_to_visualizer(route_packet)
                    print(f"[Vehicle] Route planned: {station_order}")

        except asyncio.IncompleteReadError:
            print("[Vehicle] Transmitter disconnected")
        except Exception as e:
            print(f"[Vehicle] Error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def serve_forever(self):
        await self.start()
        async with self.server:
            await self.server.serve_forever()


async def main():
    server = VehicleServer()
    try:
        await server.serve_forever()
    except KeyboardInterrupt:
        print("[Vehicle] Shutting down...")
    finally:
        await server.visualizer_client.close()


if __name__ == "__main__":
    asyncio.run(main())