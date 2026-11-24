"""Backend: the app database (db), the registry graph (graph, kg), collectors, enrichment (ctx, people, intel),
scoring (engine, model, proof), CRM and deals, the email brief, the CLI (python -m backend) and the Vercel build."""
import os
from pathlib import Path


def _load_env(path=Path(__file__).resolve().parent.parent / ".env"):
    """KEY=value lines from the repo's .env into the environment. A variable already set wins."""
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().removeprefix("export ").partition("=")
            if sep and k and not k.startswith("#"):
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env()
