from __future__ import annotations

import unittest
from unittest.mock import patch

from wearwise_ai.storage.artifact_store import ArtifactStoreError
from wearwise_ai.storage.supabase_artifact_store import (
    SupabaseArtifactStore,
    SupabaseArtifactStoreConfig,
)


class FakeResponse:
    status = 200
    payload = b""

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class SupabaseArtifactStoreTests(unittest.TestCase):
    def test_write_bytes_uploads_to_supabase_storage_path(self) -> None:
        store = SupabaseArtifactStore(
            SupabaseArtifactStoreConfig(
                base_url="https://example.supabase.co",
                service_role_key="service-role",
                bucket="wearwise-media",
            )
        )

        with patch(
            "wearwise_ai.storage.supabase_artifact_store.urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            store.write_bytes(
                "ai-outputs/asset-1/background_removed_v4.webp",
                b"cutout",
                "image/webp",
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://example.supabase.co/storage/v1/object/"
            "wearwise-media/ai-outputs/asset-1/background_removed_v4.webp",
        )
        self.assertEqual(request.data, b"cutout")
        self.assertEqual(request.headers["Apikey"], "service-role")
        self.assertEqual(request.headers["Authorization"], "Bearer service-role")
        self.assertEqual(request.headers["Content-type"], "image/webp")
        self.assertEqual(request.headers["X-upsert"], "true")

    def test_write_bytes_requires_configuration(self) -> None:
        store = SupabaseArtifactStore(
            SupabaseArtifactStoreConfig(base_url="", service_role_key="", bucket="")
        )

        with self.assertRaises(ArtifactStoreError):
            store.write_bytes("output.webp", b"cutout", "image/webp")

    def test_read_bytes_downloads_from_supabase_storage_path(self) -> None:
        store = SupabaseArtifactStore(
            SupabaseArtifactStoreConfig(
                base_url="https://example.supabase.co",
                service_role_key="service-role",
                bucket="wearwise-media",
            )
        )
        response = FakeResponse()
        response.payload = b"cutout"

        with patch(
            "wearwise_ai.storage.supabase_artifact_store.urlopen",
            return_value=response,
        ) as urlopen:
            payload = store.read_bytes("ai-outputs/asset-1/background_removed_v4.webp")

        request = urlopen.call_args.args[0]
        self.assertEqual(payload, b"cutout")
        self.assertEqual(
            request.full_url,
            "https://example.supabase.co/storage/v1/object/"
            "wearwise-media/ai-outputs/asset-1/background_removed_v4.webp",
        )
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.headers["Apikey"], "service-role")
        self.assertEqual(request.headers["Authorization"], "Bearer service-role")


if __name__ == "__main__":
    unittest.main()
