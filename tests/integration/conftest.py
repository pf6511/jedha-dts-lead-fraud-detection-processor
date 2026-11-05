"""Integration tests: they call the real services configured in secrets.env
(database, MLflow, inference API, payment API, Discord).

They are excluded from the default run (see pyproject.toml) and are run
from the repository root with:

    uv run pytest -m integration

Without secrets.env, the folder is not collected at all.
"""

from pathlib import Path

collect_ignore_glob = [] if Path("secrets.env").is_file() else ["test_*.py"]
