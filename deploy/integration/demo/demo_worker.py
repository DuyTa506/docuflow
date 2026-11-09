#!/usr/bin/env python3
"""Integration-queue worker for the demo only.

Serves just the IntegrationJobWorkflow, on the queue and with the MSSQL target
the demo's environment sets (start_demo.sh), so the demo neither needs the
production worker restarted with demo settings nor competes with it: OCR,
digest and translation still run on the production workers.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from temporalio.worker import Worker  # noqa: E402

from config.settings import settings  # noqa: E402
from services.pipeline.temporal_client import get_temporal_client  # noqa: E402
from workers.temporal_worker import _integration_worker_config  # noqa: E402


async def main() -> None:
    client = await get_temporal_client()
    cfg = _integration_worker_config()
    print(
        f"demo worker: queue={cfg['task_queue']} mssql={'on' if settings.integration_mssql_url else 'off'} "
        f"base_url={settings.integration_public_base_url}",
        flush=True,
    )
    await Worker(client, **cfg).run()


if __name__ == "__main__":
    asyncio.run(main())
