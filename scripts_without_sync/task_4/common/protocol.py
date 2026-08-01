import json
import struct
import asyncio
from dataclasses import dataclass, asdict
from typing import List, Optional, Literal, Any
from abc import ABC


@dataclass
class StationInfo:
    id: str
    x: int
    y: int
    status: Literal["unknown", "functional", "broken"]
    description: str
    is_base: bool = False


@dataclass
class StationStatusUpdate:
    id: str
    status: Literal["functional", "broken"]


class BasePacket(ABC):
    type: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class BaseMapPacket(BasePacket):
    type: Literal["base_map"] = "base_map"
    image_path: str = "image.jpg"
    stations: List[StationInfo] = None

    def __post_init__(self):
        if self.stations is None:
            self.stations = []


@dataclass
class NewStationPacket(BasePacket):
    type: Literal["new_station"] = "new_station"
    id: str = ""
    x: int = 0
    y: int = 0
    status: Literal["functional", "broken"] = "functional"
    description: str = ""
    is_base: bool = False


@dataclass
class StatusUpdatePacket(BasePacket):
    type: Literal["status_update"] = "status_update"
    updates: List[StationStatusUpdate] = None

    def __post_init__(self):
        if self.updates is None:
            self.updates = []


@dataclass
class ReconCompletePacket(BasePacket):
    type: Literal["recon_complete"] = "recon_complete"


@dataclass
class RoutePacket(BasePacket):
    type: Literal["route"] = "route"
    station_order: List[str] = None

    def __post_init__(self):
        if self.station_order is None:
            self.station_order = []


PACKET_TYPES = {
    "base_map": BaseMapPacket,
    "new_station": NewStationPacket,
    "status_update": StatusUpdatePacket,
    "recon_complete": ReconCompletePacket,
    "route": RoutePacket,
}


def encode_packet(packet: BasePacket) -> bytes:
    json_str = packet.to_json()
    data = json_str.encode("utf-8")
    length = struct.pack(">I", len(data))
    return length + data


def decode_packet(data: bytes) -> BasePacket:
    json_str = data.decode("utf-8")
    obj = json.loads(json_str)
    packet_type = obj.get("type")
    cls = PACKET_TYPES.get(packet_type)
    if cls is None:
        raise ValueError(f"Unknown packet type: {packet_type}")
    return cls(**obj)


async def send_packet(writer: asyncio.StreamWriter, packet: BasePacket):
    data = encode_packet(packet)
    writer.write(data)
    await writer.drain()


async def recv_packet(reader: asyncio.StreamReader) -> BasePacket:
    length_bytes = await reader.readexactly(4)
    length = struct.unpack(">I", length_bytes)[0]
    data = await reader.readexactly(length)
    return decode_packet(data)