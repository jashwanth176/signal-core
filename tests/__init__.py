"""The tests run against TEST_DATABASE_URL, a throwaway Postgres, never the app database."""
import os

from backend import _load_env

_load_env()
_url = os.environ.get("TEST_DATABASE_URL", "")
if not _url or _url == os.environ.get("DATABASE_URL"):
    raise SystemExit("TEST_DATABASE_URL must name a throwaway Postgres, not the app database: see the README")
os.environ["DATABASE_URL"] = _url
