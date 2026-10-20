#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket


def main() -> int:
    parser = argparse.ArgumentParser(description="Gui 1 tin hieu chup den 07_run_live --mode trigger")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    try:
        with socket.create_connection((args.host, args.port), timeout=args.timeout) as client:
            client.sendall(b"STATUS\n" if args.status else b"TRIGGER\n")
            with client.makefile("r", encoding="utf-8") as stream:
                for line in stream:
                    response = json.loads(line)
                    print(json.dumps(response, ensure_ascii=False, indent=2))
                    if response.get("status") != "queued":
                        return 0 if response.get("status") in ("done", "ready") else 1
        print("Ket noi dong truoc khi co ket qua. Khong tu dong gui lai trigger.")
        return 1
    except (OSError, ValueError) as exc:
        print(f"Loi tin hieu chup: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
