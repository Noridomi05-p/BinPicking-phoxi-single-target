#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from phoxi_yolo3d.config import DEFAULT_CLASSES_PATH


def parse_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("Danh sach lop dang rong.")
    if len(names) != len(set(names)):
        raise ValueError("Ten lop bi trung.")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="Dat danh sach lop theo thu tu co dinh")
    parser.add_argument(
        "--classes",
        help='Cac ten cach nhau boi dau phay, vi du: "chai_nhua,hop_giay,lon_nhom"',
    )
    parser.add_argument("--output", default=str(DEFAULT_CLASSES_PATH))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw = args.classes or input("Nhap ten cac lop, cach nhau boi dau phay: ")
    names = parse_names(raw)
    output = Path(args.output).expanduser().resolve()
    existing = [
        line.strip()
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ] if output.exists() else []
    if existing and not args.force:
        raise SystemExit(
            f"{output} da co {len(existing)} lop. Them --force neu that su muon thay doi."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Da luu {len(names)} lop vao {output}")
    for index, name in enumerate(names):
        print(f"  {index}: {name}")
    print("Khong doi thu tu lop sau khi da bat dau gan nhan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

