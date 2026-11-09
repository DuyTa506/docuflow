#!/usr/bin/env python3
"""Issue, list and revoke API keys for library systems, from the server shell.

Same as the admin endpoints under /api/v2/integration/clients, for operators
without an admin JWT at hand.

  python scripts/integration_client.py create "Thư viện Trường" --callback-url http://lib:9099/cb
  python scripts/integration_client.py create "Thư viện Trường" --json   # machine-readable
  python scripts/integration_client.py list
  python scripts/integration_client.py revoke CLI_001

The API key and webhook secret are printed once and not stored in clear.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.database import get_db_manager, init_database  # noqa: E402
from data.db_models import IntegrationClient  # noqa: E402
from services import integration_service as svc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    create = sub.add_parser("create")
    create.add_argument("name")
    create.add_argument("--callback-url")
    create.add_argument("--json", action="store_true")
    sub.add_parser("list")
    revoke = sub.add_parser("revoke")
    revoke.add_argument("client_id")
    args = parser.parse_args()

    init_database()
    with get_db_manager().session() as db:
        if args.cmd == "create":
            client, key, secret = svc.create_client(db, args.name, callback_url=args.callback_url)
            out = {"client_id": client.id, "name": client.name, "api_key": key, "webhook_secret": secret}
            if args.json:
                print(json.dumps(out, ensure_ascii=False))
            else:
                for k, v in out.items():
                    print(f"{k:15} {v}")
                print("\nLưu API key và webhook secret ngay: chúng không hiển thị lại.")
        elif args.cmd == "list":
            for c in db.query(IntegrationClient).order_by(IntegrationClient.created_at):
                print(f"{c.id:10} {c.status:8} {c.api_key_prefix}…  {c.name}  {c.callback_url or ''}")
        else:
            client = db.get(IntegrationClient, args.client_id)
            if client is None:
                sys.exit(f"{args.client_id}: not found")
            client.status = "REVOKED"
            print(f"{client.id} revoked")


if __name__ == "__main__":
    main()
