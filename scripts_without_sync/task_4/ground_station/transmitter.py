import asyncio
import random
import uuid
from typing import List

from common.protocol import NewStationPacket, StatusUpdatePacket, ReconCompletePacket
from common.network import TCPClient


class Transmitter:
    def __init__(self, vehicle_host: str = "localhost", vehicle_port: int = 8001):
        self.vehicle_host = vehicle_host
        self.vehicle_port = vehicle_port
        self.client = TCPClient(vehicle_host, vehicle_port)
        self.running = True
        self.cycle = 0

        self.discovered_locations = [
            (650, 450), (150, 600), (900, 200), (300, 700), (1000, 500),
            (100, 100), (1100, 800), (500, 100), (200, 800), (800, 700),
        ]
        self.discovered_index = 0

    async def connect(self) -> bool:
        return await self.client.connect()

    async def send_new_station(self):
        if self.discovered_index < len(self.discovered_locations):
            x, y = self.discovered_locations[self.discovered_index]
            self.discovered_index += 1
        else:
            x = random.randint(50, 1150)
            y = random.randint(50, 840)

        station_id = f"disc_{self.cycle:03d}_{random.randint(100, 999)}"
        status = random.choice(["functional", "broken"])
        descriptions = [
            "Электростанция на возвышенности",
            "Солнечная панель у дороги",
            "Ветряк рядом с лесом",
            "Гидростанция на реке",
            "Мобильная зарядка",
        ]
        description = random.choice(descriptions)

        packet = NewStationPacket(
            id=station_id, x=x, y=y, status=status,
            description=description, is_base=False
        )
        await self.client.send(packet)
        print(f"[Transmitter] Sent new_station: {station_id} at ({x}, {y}) status={status}")

    async def send_status_update(self):
        base_ids = ["base_0", "base_1"]
        updates = []
        for bid in base_ids:
            status = random.choice(["functional", "broken"])
            updates.append({"id": bid, "status": status})

        packet = StatusUpdatePacket(updates=updates)
        await self.client.send(packet)
        print(f"[Transmitter] Sent status_update: {[(u['id'], u['status']) for u in updates]}")

    async def send_recon_complete(self):
        packet = ReconCompletePacket()
        await self.client.send(packet)
        print(f"[Transmitter] Sent recon_complete")

    async def run_cycle(self):
        await self.send_new_station()
        await asyncio.sleep(0.1)
        await self.send_status_update()
        await asyncio.sleep(0.1)
        await self.send_recon_complete()

    async def run(self):
        connected = await self.connect()
        if not connected:
            print("[Transmitter] Failed to connect to vehicle")
            return

        while self.running:
            self.cycle += 1
            print(f"\n[Transmitter] === Cycle {self.cycle} ===")
            await self.run_cycle()
            await asyncio.sleep(5)


async def main():
    transmitter = Transmitter()
    try:
        await transmitter.run()
    except KeyboardInterrupt:
        print("[Transmitter] Stopping...")
    finally:
        await transmitter.client.close()


if __name__ == "__main__":
    asyncio.run(main())