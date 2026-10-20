from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import list_images, parse_yolo_segmentation, write_yolo_segmentation


@dataclass
class PolygonObject:
    class_id: int
    points_px: list[tuple[float, float]]


class OpenCVPolygonAnnotator:
    """Minimal local polygon labeller that writes native YOLO-seg labels."""

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        class_names: list[str],
        auto_labels_dir: str | Path | None = None,
        review_all: bool = False,
        max_view_width: int = 1500,
        max_view_height: int = 850,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Can cai opencv-python de dung cong cu gan nhan.") from exc
        self.cv2 = cv2
        self.images = list_images(images_dir)
        self.labels_dir = Path(labels_dir)
        self.auto_labels_dir = Path(auto_labels_dir) if auto_labels_dir else None
        self.class_names = class_names
        self.review_all = review_all
        self.max_view_width = max_view_width
        self.max_view_height = max_view_height
        self.index = 0
        self.class_id = 0
        self.objects: list[PolygonObject] = []
        self.current_points: list[tuple[float, float]] = []
        self.image = None
        self.scale = 1.0
        self.display_height = 0
        self.window_name = "PhoXi YOLO Segmentation Labeler"

    def _manual_path(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def _source_label_path(self, image_path: Path) -> Path | None:
        manual = self._manual_path(image_path)
        if manual.exists():
            return manual
        if self.auto_labels_dir:
            automatic = self.auto_labels_dir / f"{image_path.stem}.txt"
            if automatic.exists():
                return automatic
        return None

    def _next_unlabelled_index(self, start: int) -> int | None:
        if self.review_all:
            return min(max(start, 0), len(self.images) - 1) if self.images else None
        for index in range(max(start, 0), len(self.images)):
            if not self._manual_path(self.images[index]).exists():
                return index
        return None

    def _load(self, index: int) -> None:
        self.index = index
        image_path = self.images[index]
        self.image = self.cv2.imread(str(image_path), self.cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Khong doc duoc anh: {image_path}")
        height, width = self.image.shape[:2]
        source_label = self._source_label_path(image_path)
        self.objects = []
        if source_label:
            for class_id, coordinates in parse_yolo_segmentation(source_label):
                points = [
                    (coordinates[i] * width, coordinates[i + 1] * height)
                    for i in range(0, len(coordinates), 2)
                ]
                self.objects.append(PolygonObject(class_id, points))
        self.current_points = []
        self.scale = min(
            self.max_view_width / width,
            self.max_view_height / height,
            1.0,
        )
        self.display_height = int(round(height * self.scale))

    def _save(self) -> None:
        image_path = self.images[self.index]
        height, width = self.image.shape[:2]
        serialised = []
        for obj in self.objects:
            coordinates: list[float] = []
            for x, y in obj.points_px:
                coordinates.extend(
                    [float(np.clip(x / width, 0, 1)), float(np.clip(y / height, 0, 1))]
                )
            serialised.append((obj.class_id, coordinates))
        write_yolo_segmentation(self._manual_path(image_path), serialised)

    def _finish_polygon(self) -> None:
        if len(self.current_points) >= 3:
            self.objects.append(PolygonObject(self.class_id, self.current_points.copy()))
        self.current_points = []

    def _mouse(self, event, x, y, flags, param) -> None:
        del flags, param
        if y >= self.display_height:
            return
        original = (x / self.scale, y / self.scale)
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self.current_points.append(original)
        elif event == self.cv2.EVENT_RBUTTONDOWN:
            self._finish_polygon()

    def _render(self) -> np.ndarray:
        cv2 = self.cv2
        height, width = self.image.shape[:2]
        display_width = int(round(width * self.scale))
        view = cv2.resize(
            self.image, (display_width, self.display_height), interpolation=cv2.INTER_AREA
        )
        overlay = view.copy()
        palette = [
            (64, 220, 255),
            (255, 160, 64),
            (80, 230, 100),
            (230, 90, 220),
            (80, 120, 255),
            (255, 220, 70),
        ]
        for obj in self.objects:
            colour = palette[obj.class_id % len(palette)]
            points = np.array(
                [[round(x * self.scale), round(y * self.scale)] for x, y in obj.points_px],
                dtype=np.int32,
            )
            if len(points) >= 3:
                cv2.fillPoly(overlay, [points], colour)
                cv2.polylines(view, [points], True, colour, 2)
                anchor = tuple(points[0])
                cv2.putText(
                    view,
                    self.class_names[obj.class_id],
                    anchor,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
        view = cv2.addWeighted(overlay, 0.28, view, 0.72, 0)
        if self.current_points:
            points = np.array(
                [
                    [round(x * self.scale), round(y * self.scale)]
                    for x, y in self.current_points
                ],
                dtype=np.int32,
            )
            cv2.polylines(view, [points], False, (255, 255, 255), 2)
            for point in points:
                cv2.circle(view, tuple(point), 4, (255, 255, 255), -1)

        panel_height = 135
        canvas = np.zeros((self.display_height + panel_height, display_width, 3), dtype=np.uint8)
        canvas[: self.display_height] = view
        image_name = self.images[self.index].name
        status = (
            f"Anh {self.index + 1}/{len(self.images)}: {image_name} | "
            f"Lop [{self.class_id}] {self.class_names[self.class_id]} | "
            f"Objects={len(self.objects)}"
        )
        help_lines = [
            status,
            "Trai: them diem | Phai/Enter: dong polygon | U: lui diem | X: xoa object cuoi",
            "[/]: doi lop | 0-9: chon lop | S/Space: luu + anh tiep | P: anh truoc | Q: thoat",
            "Anh khong co vat: khong ve gi, bam S de luu nhan rong.",
        ]
        for row, text in enumerate(help_lines):
            cv2.putText(
                canvas,
                text,
                (12, self.display_height + 27 + row * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def run(self) -> None:
        if not self.images:
            raise RuntimeError("Khong co anh trong thu muc gan nhan.")
        first = self._next_unlabelled_index(0)
        if first is None:
            print("Tat ca anh da co nhan. Dung --review de xem lai.")
            return
        self._load(first)
        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)
        self.cv2.setMouseCallback(self.window_name, self._mouse)
        while True:
            self.cv2.imshow(self.window_name, self._render())
            key = self.cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key in (13, 10):
                self._finish_polygon()
            elif key == ord("u"):
                if self.current_points:
                    self.current_points.pop()
            elif key == ord("x"):
                self.current_points = []
                if self.objects:
                    self.objects.pop()
            elif key == ord("r"):
                self.current_points = []
                self.objects = []
            elif key == ord("["):
                self.class_id = (self.class_id - 1) % len(self.class_names)
            elif key == ord("]"):
                self.class_id = (self.class_id + 1) % len(self.class_names)
            elif ord("0") <= key <= ord("9"):
                selected = key - ord("0")
                if selected < len(self.class_names):
                    self.class_id = selected
            elif key in (ord("s"), ord(" ")):
                self._finish_polygon()
                self._save()
                next_index = self._next_unlabelled_index(self.index + 1)
                if next_index is None:
                    print("Da gan nhan xong tat ca anh.")
                    break
                self._load(next_index)
            elif key == ord("p"):
                self._load(max(0, self.index - 1))
            elif key == ord("q"):
                print("Thoat. Anh hien tai chi duoc luu neu ban da bam S/Space.")
                break
        self.cv2.destroyAllWindows()

