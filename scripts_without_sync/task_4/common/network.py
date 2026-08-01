import asyncio
from typing import Callable, Awaitable, Optional
from common.protocol import BasePacket


class TCPServer:
    def __init__(self, host: str, port: int, handler: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]):
        self.host = host
        self.port = port
        self.handler = handler
        self.server: Optional[asyncio.Server] = None

    async def start(self):
        self.server = await asyncio.start_server(self.handler, self.host, self.port)
        addr = self.server.sockets[0].getsockname()
        print(f"[Server] Listening on {addr[0]}:{addr[1]}")

    async def serve_forever(self):
        await self.start()
        async with self.server:
            await self.server.serve_forever()

    def close(self):
        if self.server:
            self.server.close()


class TCPClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self._connected = False

    async def connect(self) -> bool:
        try:
            self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
            self._connected = True
            print(f"[Client] Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[Client] Connection failed: {e}")
            return False

    async def send(self, packet: BasePacket):
        if not self._connected or self.writer is None:
            raise RuntimeError("Not connected")
        from common.protocol import encode_packet
        data = encode_packet(packet)
        self.writer.write(data)
        await self.writer.drain()

    async def receive(self) -> BasePacket:
        if not self._connected or self.reader is None:
            raise RuntimeError("Not connected")
        from common.protocol import recv_packet
        return await recv_packet(self.reader)

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected