import json
import struct
import asyncio
from dataclasses import dataclass, asdict, fields
from typing import List, Optional, Literal, Any, get_type_hints, get_origin, get_args
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


def _convert_value(value: Any, target_type: Any) -> Any:
    """Recursively convert dict values to appropriate dataclass instances."""
    if value is None:
        return None
    
    # Handle dataclass directly
    if hasattr(target_type, '__dataclass_fields__'):
        if isinstance(value, dict):
            from common.protocol import _instantiate_dataclass
            return _instantiate_dataclass(target_type, value)
        return value
    
    # Handle List[SomeDataclass]
    origin = get_origin(target_type)
    if origin is list or origin is List or origin == list:
        args = get_args(target_type)
        if args:
            item_type = args[0]
            if hasattr(item_type, '__dataclass_fields__'):
                return [_convert_value(v, item_type) for v in value]
    return value


def _instantiate_dataclass(cls: type, data: dict) -> Any:
    """Create dataclass instance with proper nested object conversion."""
    if not hasattr(cls, '__dataclass_fields__'):
        return data
    
    field_types = get_type_hints(cls)
    kwargs = {}
    for field in fields(cls):
        field_name = field.name
        if field_name in data:
            field_type = field_types.get(field_name, field.type)
            kwargs[field_name] = _convert_value(data[field_name], field_type)
        elif field.default is not None or hasattr(field, 'default_factory'):
            # Use default
            pass
        else:
            kwargs[field_name] = data.get(field_name)
    return cls(**kwargs)


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
    return _instantiate_dataclass(cls, obj)


async def send_packet(writer: asyncio.StreamWriter, packet: BasePacket):
    data = encode_packet(packet)
    writer.write(data)
    await writer.drain()


async def recv_packet(reader: asyncio.StreamReader) -> BasePacket:
    length_bytes = await reader.readexactly(4)
    length = struct.unpack(">I", length_bytes)[0]
    data = await reader.readexactly(length)
    return decode_packet(data)