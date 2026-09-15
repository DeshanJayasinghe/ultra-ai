from __future__ import annotations

import os
from pathlib import Path

from wearwise_ai.core.env import load_environment_file


def test_load_environment_file_reads_project_dotenv(tmp_path: Path) -> None:
    project = tmp_path / "wearwise-ai"
    nested = project / "src" / "wearwise_ai"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname = 'wearwise-ai'\n")
    (project / ".env").write_text("WEARWISE_TEST_ENV_VALUE=from-dotenv\n")

    previous = os.environ.pop("WEARWISE_TEST_ENV_VALUE", None)
    try:
        loaded = load_environment_file(start=nested)

        assert loaded is True
        assert os.environ["WEARWISE_TEST_ENV_VALUE"] == "from-dotenv"
    finally:
        if previous is None:
            os.environ.pop("WEARWISE_TEST_ENV_VALUE", None)
        else:
            os.environ["WEARWISE_TEST_ENV_VALUE"] = previous


def test_load_environment_file_does_not_override_existing_env(tmp_path: Path) -> None:
    project = tmp_path / "wearwise-ai"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'wearwise-ai'\n")
    (project / ".env").write_text("WEARWISE_TEST_ENV_VALUE=from-dotenv\n")

    previous = os.environ.get("WEARWISE_TEST_ENV_VALUE")
    os.environ["WEARWISE_TEST_ENV_VALUE"] = "from-shell"
    try:
        loaded = load_environment_file(start=project)

        assert loaded is True
        assert os.environ["WEARWISE_TEST_ENV_VALUE"] == "from-shell"
    finally:
        if previous is None:
            os.environ.pop("WEARWISE_TEST_ENV_VALUE", None)
        else:
            os.environ["WEARWISE_TEST_ENV_VALUE"] = previous
