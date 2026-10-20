#!/usr/bin/env python3
"""Read-only machine checks, plus an optional real CUDA model forward test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiem tra driver / CUDA / extension GraspNet")
    parser.add_argument("--repo", default=str(ROOT / "third_party/graspnet-baseline"))
    parser.add_argument("--checkpoint", default=str(ROOT / "models/graspnet/checkpoint-rs.tar"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Forward tren cloud tong hop, khong tao pose cho robot")
    parser.add_argument("--num-points", type=int, default=20000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    from phoxi_yolo3d.grasp_backend import GraspNetBackend, environment_report
    repo, checkpoint = Path(args.repo).expanduser().resolve(), Path(args.checkpoint).expanduser().resolve()
    report = environment_report(repo, checkpoint, args.device)
    if args.smoke and report["ready_for_smoke_test"]:
        import numpy as np
        rng = np.random.default_rng(42)
        cloud = rng.uniform([-.1, -.1, .9], [.1, .1, 1.1], (args.num_points, 3)).astype(np.float32)
        try:
            backend = GraspNetBackend(repo, checkpoint, {"device": args.device})
            rows, profile = backend.predict(cloud, np.full_like(cloud, .5))
            report.update(full_forward_tested=True, smoke_profile=profile,
                          smoke_proposal_count=len(rows), smoke_input="synthetic diagnostic only")
        except Exception as exc:
            report["smoke_error"] = f"{type(exc).__name__}: {exc}"
    formatted = json.dumps(report, ensure_ascii=False, indent=2)
    print(formatted)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(formatted+"\n", encoding="utf-8")
    return 0 if (report["full_forward_tested"] if args.smoke else report["ready_for_smoke_test"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
