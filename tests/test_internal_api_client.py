from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from wearwise_ai.infrastructure.internal_api_client import (
    InternalApiConfig,
    InternalApiError,
    LaravelInternalApiClient,
)


class FakeResponse:
    status = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class LaravelInternalApiClientTests(unittest.TestCase):
    def test_claim_next_job_posts_worker_and_job_types(self) -> None:
        client = LaravelInternalApiClient(
            InternalApiConfig(
                base_url="http://api.test",
                bearer_token="internal-token",
            )
        )

        with patch(
            "wearwise_ai.infrastructure.internal_api_client.urlopen",
            return_value=FakeResponse({"data": {"id": "job-1"}}),
        ) as urlopen:
            response = client.claim_next_job(
                request_id="request-1",
                worker_id="worker-a",
                job_types=["background_removal"],
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(response["data"], {"id": "job-1"})
        self.assertEqual(request.full_url, "http://api.test/internal/v1/ai/jobs/claim-next")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Authorization"], "Bearer internal-token")
        self.assertEqual(request.headers["X-request-id"], "request-1")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {
                "worker_id": "worker-a",
                "job_types": ["background_removal"],
            },
        )

    def test_timeout_is_wrapped_as_internal_api_error(self) -> None:
        client = LaravelInternalApiClient(
            InternalApiConfig(
                base_url="http://api.test",
                bearer_token="internal-token",
                retry_backoff_seconds=0,
            )
        )

        with patch(
            "wearwise_ai.infrastructure.internal_api_client.urlopen",
            side_effect=TimeoutError(),
        ), self.assertRaisesRegex(InternalApiError, "timed out"):
            client.claim_next_job(
                request_id="request-1",
                worker_id="worker-a",
                job_types=["background_removal"],
            )

    def test_progress_timeout_is_logged_and_does_not_raise(self) -> None:
        client = LaravelInternalApiClient(
            InternalApiConfig(
                base_url="http://api.test",
                bearer_token="internal-token",
                max_attempts=1,
                retry_backoff_seconds=0,
            )
        )

        with patch(
            "wearwise_ai.infrastructure.internal_api_client.urlopen",
            side_effect=TimeoutError(),
        ):
            response = client.report_progress(
                "job-1",
                request_id="request-1",
                worker_id="worker-a",
                progress_percent=45,
                status_message="fetching cleaned garment image",
            )

        self.assertEqual(response, {})

    def test_timeout_error_includes_method_and_path(self) -> None:
        client = LaravelInternalApiClient(
            InternalApiConfig(
                base_url="http://api.test",
                bearer_token="internal-token",
                max_attempts=1,
                retry_backoff_seconds=0,
            )
        )

        with patch(
            "wearwise_ai.infrastructure.internal_api_client.urlopen",
            side_effect=TimeoutError(),
        ), self.assertRaisesRegex(
            InternalApiError,
            r"POST /internal/v1/ai/jobs/claim-next timed out",
        ):
            client.claim_next_job(
                request_id="request-1",
                worker_id="worker-a",
                job_types=["background_removal"],
            )


if __name__ == "__main__":
    unittest.main()
