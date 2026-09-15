from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from wearwise_ai.storage import ArtifactStoreError, LocalArtifactStore


class LocalArtifactStoreTests(unittest.TestCase):
    def test_reads_and_writes_artifacts_under_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(root=Path(tmpdir))

            store.write_bytes("ai-outputs/asset-1/cutout.webp", b"payload", "image/webp")

            self.assertEqual(store.read_bytes("ai-outputs/asset-1/cutout.webp"), b"payload")

    def test_rejects_path_traversal(self) -> None:
        store = LocalArtifactStore(root=Path("/tmp"))

        with self.assertRaises(ArtifactStoreError):
            store.read_bytes("../secret")


if __name__ == "__main__":
    unittest.main()
