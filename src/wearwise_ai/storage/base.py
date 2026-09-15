from abc import ABC, abstractmethod


class ObjectStorage(ABC):
    @abstractmethod
    async def read_bytes(self, key: str) -> bytes:
        """Read an object by storage key."""

    @abstractmethod
    async def write_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        """Write an object by storage key."""
