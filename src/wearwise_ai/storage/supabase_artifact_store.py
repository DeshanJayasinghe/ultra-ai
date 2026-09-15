from __future__ import annotations

from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from wearwise_ai.storage.artifact_store import ArtifactStoreError


@dataclass(frozen=True, slots=True)
class SupabaseArtifactStoreConfig:
    base_url: str
    service_role_key: str
    bucket: str
    timeout_seconds: float = 30.0


class SupabaseArtifactStore:
    def __init__(self, config: SupabaseArtifactStoreConfig) -> None:
        self._config = config

    def read_bytes(self, key: str) -> bytes:
        self._validate_config()
        request = Request(
            url=self._object_url(key),
            method="GET",
            headers={
                "apikey": self._config.service_role_key,
                "Authorization": f"Bearer {self._config.service_role_key}",
            },
        )

        try:
            with urlopen(request, timeout=self._config.timeout_seconds) as response:
                if response.status < 200 or response.status >= 300:
                    raise ArtifactStoreError(
                        f"Supabase artifact read failed with HTTP {response.status}: {key}."
                    )
                return response.read()
        except HTTPError as exc:
            raise ArtifactStoreError(
                f"Supabase artifact read failed with HTTP {exc.code}: {key}."
            ) from exc
        except URLError as exc:
            raise ArtifactStoreError(f"Supabase artifact read failed: {key}.") from exc

    def write_bytes(self, key: str, payload: bytes, content_type: str) -> None:
        self._validate_config()
        request = Request(
            url=self._object_url(key),
            data=payload,
            method="POST",
            headers={
                "apikey": self._config.service_role_key,
                "Authorization": f"Bearer {self._config.service_role_key}",
                "Content-Type": content_type,
                "x-upsert": "true",
            },
        )

        try:
            with urlopen(request, timeout=self._config.timeout_seconds) as response:
                if response.status < 200 or response.status >= 300:
                    raise ArtifactStoreError(
                        f"Supabase artifact upload failed with HTTP {response.status}: {key}."
                    )
        except HTTPError as exc:
            raise ArtifactStoreError(
                f"Supabase artifact upload failed with HTTP {exc.code}: {key}."
            ) from exc
        except URLError as exc:
            raise ArtifactStoreError(f"Supabase artifact upload failed: {key}.") from exc

    def _validate_config(self) -> None:
        if (
            not self._config.base_url
            or not self._config.service_role_key
            or not self._config.bucket
        ):
            raise ArtifactStoreError("Supabase artifact store is not configured.")

    def _object_url(self, key: str) -> str:
        path = "/".join(quote(part, safe="") for part in key.split("/"))
        return (
            f"{self._config.base_url.rstrip('/')}/storage/v1/object/"
            f"{quote(self._config.bucket, safe='')}/{path}"
        )
