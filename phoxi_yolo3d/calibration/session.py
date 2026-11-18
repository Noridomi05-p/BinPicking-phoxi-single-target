"""Atomic, resumable sessions. Original frames are never removed by excluding a sample."""
from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..storage import load_capture, save_bgr_png, save_capture
from .board import BoardDetection, BoardSpec, overlay
from .math3d import check_transform, pose_error, pose_zyz


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with pending.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def frame_identity(frame):
    acquisition = frame.info.get("acquisition", {})
    device_id = acquisition.get("device_id")
    space = acquisition.get("coordinate_space")
    if not device_id or space != "CameraSpace/PrimaryCamera":
        raise ValueError("Frame thiếu định danh camera hoặc chưa xác nhận CameraSpace/PrimaryCamera.")
    return {"device_id": str(device_id), "coordinate_space": space,
            "image_shape_hw": list(frame.texture_bgr.shape[:2]),
            "source_shape_hw": acquisition.get("source_shape_hw"),
            "roi_xyxy": acquisition.get("roi_xyxy"),
            "calibration_id": acquisition.get("calibration_id")}


class Session:
    def __init__(self, root, document):
        self.root, self.doc = Path(root).resolve(), document
        b = self.doc["board"]
        self.spec = BoardSpec(int(b["cols"]), int(b["rows"]), float(b["square_mm"]))

    @classmethod
    def create(cls, root, spec: BoardSpec, detection_settings=None, solver_settings=None, demo=False):
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=False)
        doc = {"schema_version": 1, "kind": "phoxi_eye_to_hand", "created_utc": utc_now(),
               "updated_utc": utc_now(), "demo": bool(demo), "board": spec.to_dict(),
               "robot_reference": "flange", "robot_frame": "BASE", "euler": "intrinsic_ZYZ",
               "distance_unit": "millimetre", "angle_unit": "degree", "camera_identity": None,
               "detection_settings": detection_settings or {}, "solver_settings": solver_settings or {},
               "samples": [], "current_result": None}
        s = cls(root, doc)
        s.save()
        return s

    @classmethod
    def load(cls, root):
        root = Path(root).expanduser().resolve()
        if root.is_file():
            root = root.parent
        doc = json.loads((root / "session.json").read_text(encoding="utf-8"))
        if doc.get("schema_version") != 1 or doc.get("kind") != "phoxi_eye_to_hand":
            raise ValueError("Không phải phiên PhoXi calibration được hỗ trợ.")
        if (doc.get("robot_reference"), doc.get("robot_frame"), doc.get("euler")) != ("flange", "BASE", "intrinsic_ZYZ"):
            raise ValueError("Phiên phải sử dụng pose mặt bích / BASE / ZYZ.")
        recovered = False
        known = {item["id"] for item in doc["samples"]}
        # A completed sample directory is a small journal. Recover the pair if
        # interruption happened after its rename but before session.json commit.
        for folder in sorted((root/"samples").glob("*")):
            if folder.name in known or not re.fullmatch(r"[CV]\d{4}",folder.name):
                continue
            meta = json.loads((folder/"sample.json").read_text(encoding="utf-8"))
            if (meta.get("board") != doc["board"] or meta.get("demo") != doc["demo"] or
                    (doc.get("camera_identity") is not None and meta.get("camera_identity") != doc["camera_identity"])):
                raise ValueError("Phát hiện mẫu chưa ghi danh nhưng cấu hình không khớp phiên.")
            if not (folder/"scans/frame.npz").is_file() or not (folder/"corners.npz").is_file():
                raise ValueError("Mẫu chưa ghi danh thiếu dữ liệu gốc; không tự khôi phục.")
            doc["camera_identity"] = meta["camera_identity"]
            doc["samples"].append(meta);doc["current_result"] = None;recovered = True
        for item in doc["samples"]:
            if not re.fullmatch(r"[CV]\d{4}", item["id"]):
                raise ValueError("Mã mẫu không hợp lệ.")
            if item["role"] not in ("fit", "validation"):
                raise ValueError("Vai trò mẫu không hợp lệ.")
            check_transform(item["T_base_flange"])
            check_transform(item["T_camera_board"])
        session = cls(root, doc)
        if recovered:
            session.save()
        return session

    def save(self):
        self.doc["updated_utc"] = utc_now()
        atomic_json(self.root / "session.json", self.doc)

    def fingerprint(self):
        data = {k: self.doc[k] for k in ("board", "camera_identity", "demo", "samples", "solver_settings")}
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def add(self, frame, detection, robot_pose, role, stationary_confirmed, origin_confirmed):
        if role not in ("fit", "validation"):
            raise ValueError("Vai trò mẫu không hợp lệ.")
        if not stationary_confirmed or not origin_confirmed:
            raise ValueError("Cần xác nhận robot đứng yên và gốc bảng đúng dấu trên bảng.")
        t_robot = pose_zyz(robot_pose)
        ident = frame_identity(frame)
        if bool(frame.info.get("calibration_demo", False)) != self.doc["demo"]:
            raise ValueError("Không được trộn frame DEMO và frame camera thật.")
        if self.doc["camera_identity"] is not None and self.doc["camera_identity"] != ident:
            raise ValueError("Camera / kích thước ảnh / hệ tọa độ đã đổi. Tạo phiên mới; không trộn dữ liệu.")
        for old in self.doc["samples"]:
            if not old.get("enabled", True):
                continue
            mm, deg = pose_error(old["T_base_flange"], t_robot)
            if mm < .5 and deg < .5:
                raise ValueError(f"Pose gần trùng mẫu {old['id']}. Di chuyển đến pose mới trước khi lấy mẫu.")
        prefix = "C" if role == "fit" else "V"
        next_id = 1 + max([int(x["id"][1:]) for x in self.doc["samples"] if x["id"].startswith(prefix)] or [0])
        sample_id = prefix + f"{next_id:04d}"
        pending = self.root / (".pending_" + uuid.uuid4().hex)
        final = self.root / "samples" / sample_id
        meta = {"id": sample_id, "role": role, "enabled": True,
                "captured_at_utc": frame.info.get("capture_started_utc", frame.timestamp_utc),
                "confirmed_at_utc": utc_now(), "frame_id": frame.frame_id,
                "robot_pose_xyzabc": [float(v) for v in robot_pose],
                "T_base_flange": t_robot.tolist(), **detection.to_dict(),
                "stationary_confirmed": True, "origin_confirmed": True,
                "camera_identity": ident, "board": self.doc["board"], "demo": self.doc["demo"]}
        before = copy.deepcopy(self.doc)
        try:
            save_capture(frame, pending, stem="frame", save_3d=True)
            save_bgr_png(pending / "overlay.png", overlay(frame, detection, self.spec))
            np.savez_compressed(pending / "corners.npz", uv=detection.corners_uv,
                                xyz=detection.corners_camera_mm, valid=detection.valid, inliers=detection.inliers)
            atomic_json(pending / "sample.json", meta)
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                raise FileExistsError(f"Mẫu {sample_id} đã tồn tại; không ghi đè ảnh gốc.")
            pending.rename(final)
            self.doc["camera_identity"] = ident
            self.doc["samples"].append(meta)
            self.doc["current_result"] = None
            self.save()
        except Exception:
            self.doc = before
            # Keep completed raw evidence; load() recovers its metadata journal.
            raise
        finally:
            if pending.exists():
                shutil.rmtree(pending)
        return sample_id

    def toggle(self, sample_id):
        item = next(s for s in self.doc["samples"] if s["id"] == sample_id)
        item["enabled"] = not item.get("enabled", True)
        self.doc["current_result"] = None
        self.save()

    def read_sample(self, sample_id):
        item = next(s for s in self.doc["samples"] if s["id"] == sample_id)
        folder = self.root / "samples" / item["id"]
        frame = load_capture(folder / "scans" / "frame.npz")
        with np.load(folder / "corners.npz", allow_pickle=False) as arr:
            detection = BoardDetection(arr["uv"].copy(), arr["xyz"].copy(), arr["valid"].copy(),
                arr["inliers"].copy(), np.array(item["T_camera_board"]), item["rms_mm"],
                item["max_error_mm"], item["measured_square_mm"], item["origin_flipped_180"])
        return frame, detection, item

    def store_result(self, result):
        result = dict(result)
        result.update(created_utc=utc_now(), dataset_fingerprint=self.fingerprint(),
                      camera_identity=self.doc["camera_identity"], board=self.doc["board"],
                      demo=self.doc["demo"], robot_ready=False)
        if self.doc["demo"]:
            result["status"] = "DEMO_NOT_FOR_ROBOT"
        stem = "calibration_" + result["dataset_fingerprint"][:16]
        destination = self.root / "results" / (stem + ".json")
        atomic_json(destination, result)
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=["id", "role", "position_error_mm", "rotation_error_deg", "review_required"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in result["residuals"])
        destination.with_suffix(".csv").write_text(text.getvalue(), encoding="utf-8")
        self.doc["current_result"] = str(destination.relative_to(self.root))
        self.save()
        return result

    def current_result(self):
        relative = self.doc.get("current_result")
        if not relative:
            return None
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Đường dẫn kết quả nằm ngoài phiên.")
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if result.get("dataset_fingerprint") == self.fingerprint() else None

    def export(self, path):
        result = self.current_result()
        if result is None:
            raise ValueError("Chưa có kết quả hoặc bộ mẫu đã thay đổi. Tính lại trước khi xuất.")
        atomic_json(path, result)
