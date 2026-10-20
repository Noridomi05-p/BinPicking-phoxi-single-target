# PhoXi Gen2 M + YOLO Instance Segmentation + đo 3D

Bản cập nhật 0.2: xem **[CAP_NHAT_SINGLE_TARGET.md](CAP_NHAT_SINGLE_TARGET.md)** để
chạy chọn một mục tiêu, đo chiều cao theo mặt phẳng, đề xuất tư thế camera và
kiểm tra khoảng trống. Gói đã tích hợp bản ROI/3D và GraspNet ngoại tuyến trước đó.
Tư thế mới luôn có `robot_ready: false`; TCP, hình học kẹp thật và quỹ đạo robot
chưa được hiệu chuẩn trong gói này.

Dự án này nhận diện và phân loại từng đồ vật đang xếp chồng trong hộp, sau đó dùng
`PointCloud` của PhoXi để đo khoảng cách đến từng vật. Toàn bộ nhận diện, đo lường,
kiểm tra dữ liệu và xuất kết quả được thực hiện bằng Python. PhoXi Control chỉ đóng
vai trò driver/API.

## Kết quả đầu ra cho mỗi vật

Mỗi instance có:

- `class_name`, `class_id`, độ tin cậy và bounding box;
- mask riêng của phần vật đang nhìn thấy;
- tâm bề mặt nhìn thấy `(X,Y,Z)` trong hệ camera, đơn vị mm;
- `depth_z_median_mm`: chiều sâu dọc trục quang học Z;
- `distance_3d_median_mm`: khoảng cách Euclid từ camera;
- `nearest_surface_mm`: khoảng cách ổn định đến phần gần camera nhất;
- kích thước XYZ của phần bề mặt nhìn thấy và số điểm 3D hợp lệ.

## Phân chia công việc

Phần đã được tự động hóa trong mã nguồn:

- kết nối và software-trigger PhoXi;
- đọc Texture, DepthMap, PointCloud, ConfidenceMap và NormalMap;
- chuẩn hóa texture 16-bit thành ảnh dùng cho YOLO;
- lưu ảnh/scan, kiểm tra nhãn và chia train/val/test;
- huấn luyện, đánh giá, auto-label gợi ý và export model;
- nhận diện trực tiếp, đo 3D chống nhiễu, hiển thị và xuất JSONL;
- demo và bộ kiểm thử không cần camera.

Việc người vận hành bắt buộc phải làm:

1. Gắn, cấp nguồn và nối Ethernet cho camera.
2. Đặt hoặc đảo vị trí vật thể để tạo nhiều cảnh khác nhau rồi giữ chúng đứng yên khi quét.
3. Cung cấp tên các loại vật.
4. Vẽ/review mask thủ công vì chỉ người dùng mới biết nhãn đúng của vật bị che khuất.

## 0. Bố trí PhoXi Gen2 M

Với PhoXi M, khoảng cách từ camera tới mọi bề mặt cần đo nên nằm trong vùng làm việc
của thiết bị. Mặc định dự án chỉ giữ điểm có `400 <= Z <= 1300 mm`; sau lần chụp thử,
hãy điều chỉnh `measurement.min_z_mm` và `measurement.max_z_mm` trong
`config/project.yaml` sát với hộp thực tế.

Camera phải cố định và vật phải đứng yên trong thời gian chiếu mẫu. Nên bố trí mặt vật
thường gặp quanh khoảng cách tối ưu của camera, tránh rung, bề mặt gương, trong suốt và
kim loại bóng.

Dự án hiện được cấu hình cho camera `JEJ-043` và ROI cố định của hộp
`[620, 280, 1420, 1080]` trên frame gốc 2048×1532. Mọi thành phần Texture, Depth,
PointCloud, Confidence và Normal đều được crop cùng tọa độ nên phép đo 3D vẫn thẳng
hàng với mask. Nếu camera hoặc hộp bị dịch chuyển, phải chỉnh lại `camera.roi_xyxy`
trước khi thu dataset mới.

## 1. Chuẩn bị PhoXi Control và Python

Yêu cầu:

- Ubuntu 22.04/24.04 hoặc Windows 10/11;
- Python 3.10 trở lên;
- PhoXi Control 1.17 trở lên đang chạy;
- camera xuất hiện ở trạng thái `Ready`;
- GPU NVIDIA được khuyên dùng khi train.

Trên Ubuntu, mở terminal tại thư mục dự án rồi chạy:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PhoXi installer thường đặt biến `PHOXI_CONTROL_PATH`. Kiểm tra:

```bash
echo "$PHOXI_CONTROL_PATH"
python scripts/00_diagnose.py
```

Kết quả đúng phải cho thấy PhoXi Control đang chạy, phiên bản API và ít nhất một camera.
Nếu chỉ có một camera, chương trình tự chọn. Nếu có nhiều camera, truyền
`--device-id ID_CUA_CAMERA` hoặc điền `camera.device_id` trong config.

## 2. Khai báo lớp đúng một lần

Năm lớp đã chốt cho bộ vật hiện tại:

```bash
python scripts/01_set_classes.py --classes \
  "lo_nhua_nap_xanh,lo_nhua_nap_do,khoi_tru,khoi_lap_phuong,hop_anibox"
```

Ba khối trụ khác màu cùng thuộc `khoi_tru`; ba khối lập phương khác màu cùng thuộc
`khoi_lap_phuong`. Mỗi vật vẫn là một instance/mask riêng.

Không đổi thứ tự lớp sau khi bắt đầu gán nhãn. Nếu muốn thay toàn bộ danh sách có chủ
đích, thêm `--force` và phải gán lại dữ liệu cũ.

## 3. Thu dữ liệu

### Giai đoạn pilot

Chụp khoảng 50–100 cảnh trước để kiểm tra toàn bộ pipeline:

```bash
python scripts/02_capture.py --count 80 --mode manual
```

Mỗi lần chương trình chờ, hãy thay đổi vị trí, góc quay, mức chồng lấp và số lượng vật;
giữ cảnh đứng yên rồi nhấn Enter. Ảnh dùng để train được lưu tại
`data/raw/images`.

Chụp thêm 5–10 cảnh có dữ liệu 3D để kiểm tra đo chiều sâu offline:

```bash
python scripts/02_capture.py --count 10 --mode manual --save-3d
```

Scan nén nằm trong `data/raw/scans`. Một scan 3D có thể khá lớn, vì vậy không cần lưu
3D cho toàn bộ ảnh huấn luyện.

Nên có:

- cảnh ít vật, cảnh rất lộn xộn và nhiều mức che khuất;
- mọi hướng quay thường xuất hiện;
- vật sát thành hộp, ở giữa và ở các độ cao khác nhau;
- 5–10% ảnh hộp trống hoặc không có lớp cần nhận diện;
- tối thiểu khoảng 300–500 instance nhìn thấy cho mỗi lớp ở bản huấn luyện chính.

Không chụp hàng loạt ảnh gần như giống hệt nhau rồi chia ngẫu nhiên vào train/test.

## 4. Gán nhãn thủ công

Chạy công cụ polygon viết bằng Python/OpenCV:

```bash
python scripts/03_annotate.py
```

Điều khiển:

- chuột trái: thêm điểm polygon;
- chuột phải hoặc Enter: đóng polygon;
- `[` / `]`: đổi lớp;
- `0`–`9`: chọn nhanh class ID;
- `U`: bỏ điểm vừa vẽ;
- `X`: xóa object cuối;
- `R`: xóa toàn bộ nhãn của ảnh hiện tại;
- `S` hoặc Space: lưu và sang ảnh tiếp theo;
- `P`: quay lại ảnh trước;
- `Q`: thoát.

Ảnh không có vật vẫn phải nhấn `S` để tạo file nhãn rỗng. Quy tắc chi tiết nằm trong
`HUONG_DAN_GAN_NHAN.md`.

## 5. Kiểm tra và chia dataset

Chỉ kiểm tra lỗi:

```bash
python scripts/04_prepare_dataset.py --check-only
```

Khi không còn `errors`, tạo dataset 70/20/10:

```bash
python scripts/04_prepare_dataset.py
```

Nếu chủ động muốn chia lại dataset đã tồn tại:

```bash
python scripts/04_prepare_dataset.py --overwrite
```

File dùng cho Ultralytics được tạo tự động tại `data/dataset/dataset.yaml`.

## 6. Huấn luyện pilot

Train nhanh để kiểm tra nhãn:

```bash
python scripts/05_train.py --epochs 50 --imgsz 640
```

Train chính theo config:

```bash
python scripts/05_train.py
```

Model tốt nhất được tự động sao chép về `models/best.pt`. Cấu hình mặc định dùng
`yolo11n-seg.pt`, ảnh 1024 và batch 4. Trên máy `ai_lab` RTX 4060 có thể thử
`yolo11s-seg.pt`; nếu thiếu VRAM, giảm `--batch 2` hoặc `--imgsz 768`.

## 7. Giảm công gán nhãn bằng model pilot

Sau khi có model pilot, chụp thêm ảnh rồi tạo nhãn gợi ý:

```bash
python scripts/09_autolabel.py --confidence 0.60
python scripts/03_annotate.py
```

Nhãn tự động nằm ở `data/raw/auto_labels`. Công cụ gán nhãn sẽ mở nhãn đó làm gợi ý;
khi bạn nhấn `S`, bản đã review được lưu vào `data/raw/labels`. Không đưa nhãn tự động
chưa review thẳng vào train.

## 8. Đánh giá

```bash
python scripts/06_validate.py --split test
```

Kết quả mask mAP được lưu tại `reports/validation_metrics.json`. Ngoài mAP, cần quan
sát riêng các cảnh che khuất nặng, vật sát thành hộp và các cặp lớp dễ nhầm.

## 9. Chạy trực tiếp với camera

```bash
python scripts/07_run_live.py --config config/inspection.yaml
```

Lần chạy đầu, chương trình tự mở cửa sổ để vẽ polygon ROI của lòng thùng. Nhấn chuột
trái để thêm điểm, chuột phải hoặc Enter để đóng ROI, rồi nhấn `S` để lưu. ROI được
lưu tại `config/bin_roi.json`; không cần sửa tọa độ bằng tay.

Sau khi ROI đã được lưu, bỏ hết vật khỏi thùng rồi nhấn `B`. Chương trình chụp 15
frame, lấy depth trung vị làm bề mặt thùng trống và lưu tại
`data/calibration/bin_surface.npz`. Phải chụp lại bề mặt nếu camera hoặc thùng bị
dịch chuyển.

Mặc định chụp một lần rồi giữ ảnh. Hướng dẫn cập nhật: `CAI_DAT_ROI_3D.md`.

Điều khiển cửa sổ live:

- `1/2/3/4`: manual / continuous / interval / trigger TCP;
- `Space/N`: chụp một ảnh; `P`: tạm dừng; `S`: lưu ảnh gốc và point cloud;

- `R`: vẽ hoặc chỉnh lại ROI;
- `B`: chụp lại bề mặt thùng trống;
- `D`: bật/tắt các mask bị loại và lý do loại;
- `M`: bật/tắt màu phủ trong mask;
- `Q` hoặc Esc: thoát.

Cửa sổ bên trái chỉ vẽ contour sát theo mask, tên lớp và số đo; không vẽ bounding
box. Bên phải là depth map. Kết quả từng frame được ghi vào
`results/live_.../objects.jsonl`, gồm cả danh sách vật được giữ và các ứng viên bị
loại.

Chạy không mở cửa sổ và chỉ xử lý 20 scan:

```bash
python scripts/07_run_live.py --config config/inspection.yaml --mode continuous --headless --count 20
```

Muốn lưu tất cả ảnh đã vẽ kết quả:

```bash
python scripts/07_run_live.py --save-images
```

## 10. Kiểm tra scan offline và tọa độ từng điểm

```bash
python scripts/08_run_offline.py data/raw/scans
python scripts/11_inspect_depth.py data/raw/scans/TEN_SCAN.npz
```

Trong công cụ inspect, di chuột trên vật để xem trung vị 5×5 của `X,Y,Z,D` theo mm.
`Z` là khoảng cách dọc trục quang học; `D=sqrt(X²+Y²+Z²)` là khoảng cách thực đến gốc
camera.
Khi nhận diện một mask, `Z` và `D` trên ảnh kết quả lần lượt là trung vị Z và
trung vị khoảng cách Euclid của các điểm `PointCloud` hợp lệ thuộc mask.
`D` được mã nguồn tính từ XYZ do PhoXi trả về, không lấy trực tiếp từ `DepthMap`.
Nếu đã chụp nền thùng trống bằng phím `B` và `SURFACE PASS`, ảnh kết quả hiện
thêm `Hmax` (chênh lệch
Z lớn nhất giữa nền và vật tại các pixel hợp lệ trong mask) và `H99` (phân vị 99,
ít nhạy với một điểm lỗi). `result.json` / `offline_results.json` chứa
`postprocess.surface_height_max_mm` và `surface_height_p99_mm` tương ứng.
Phần tối thiếu điểm 3D không được tính vào hai giá trị này; nếu không có nền hoặc
thiếu cặp điểm hợp lệ thì giá trị là `null`.

## 11. Export model

ONNX:

```bash
python scripts/10_export_model.py --format onnx
```

TensorRT trên máy NVIDIA:

```bash
python scripts/10_export_model.py --format engine --half --device 0
```

## 12. Demo và kiểm thử không cần camera

```bash
python scripts/99_demo_measurement.py
python -m unittest discover -s tests -v
```

Demo tạo hai vật giả ở khoảng Z 700 mm và 860 mm, chạy đúng cùng hàm đo dùng cho camera
thật và lưu kết quả vào `demo_output`.

## Cách đo 3D chống nhiễu

Với mỗi mask, chương trình thực hiện:

1. Co mask 3 pixel để bỏ vùng biên lẫn vật phía dưới/thành hộp.
2. Loại điểm `(0,0,0)`, `NaN`, vô cực và điểm ngoài khoảng Z cấu hình.
3. Có thể lọc thêm bằng ConfidenceMap sau khi quan sát phân bố thật.
4. Bỏ 1% cực trị ở hai đầu theo Z.
5. Dùng median thay vì mean/min để giảm ảnh hưởng của nhiễu.
6. Dùng percentile 5% thay cho điểm nhỏ nhất khi tính bề mặt gần nhất.

Trước khi chấp nhận một mask, tầng hậu xử lý còn thực hiện hai cổng kiểm tra:

1. Tâm mask phải nằm trong ROI và ít nhất 80% diện tích mask phải thuộc ROI.
2. So sánh Z với nền thùng trống: ít nhất 30 điểm nổi và số điểm nổi / tổng pixel mask >= 2% thì đạt kiểm tra nền. Ngưỡng nổi tại mỗi pixel là `max(5 mm, 3 × nhiễu nền)`.
3. Chỉ loại sát đáy khi đủ cặp điểm 3D ở ít nhất 50% diện tích mask. Nếu thiếu dữ liệu, giữ nhãn 2D với trạng thái UNKNOWN. `verified_3d` chỉ đúng khi phép đo và kiểm tra nền đều đạt.
4. Lọc các mask cùng lớp chồng lấn mạnh để giảm nhãn trùng. Xem ngưỡng tại `config/inspection.yaml`.

Vì vậy dự đoán ngoài thùng bị ROI loại, còn vết xước hoặc hình in trên đáy thùng bị
bộ lọc 3D loại khi có đủ dữ liệu đối chiếu. Thiếu 3D không được coi là bằng chứng đó là đáy thùng.

Mã chủ động ép PhoXi về `CameraSpace` để `X,Y,Z,D` thật sự xuất phát từ camera. Tắt
`camera.force_camera_space` chỉ khi bạn cố ý làm việc trong Marker/Robot/Custom space.

## Cấu trúc chính

```text
config/project.yaml        Cấu hình camera, đo 3D, YOLO và training
config/classes.txt         Danh sách lớp theo thứ tự cố định
phoxi_yolo3d/camera.py     Giao tiếp phoxi-api
phoxi_yolo3d/measurement.py Đo XYZ/Z/D từ từng mask
phoxi_yolo3d/detector.py   YOLO instance segmentation
phoxi_yolo3d/roi.py        Vẽ, lưu và kiểm tra polygon ROI
phoxi_yolo3d/surface_reference.py Bề mặt depth của thùng trống
phoxi_yolo3d/postprocess.py Lọc ROI và độ cao 3D
phoxi_yolo3d/annotator.py  Công cụ gán polygon nội bộ
phoxi_yolo3d/dataset.py    Kiểm tra/chia dataset
scripts/                   Các bước chạy theo số thứ tự
tests/                     Kiểm thử phần không phụ thuộc phần cứng
```

## Giới hạn đúng của bài toán

- Camera chỉ đo được bề mặt đang nhìn thấy; code không giả đo phần bị che hoàn toàn.
- `visible_extent_xyz_mm` là kích thước phần nhìn thấy, không phải kích thước đầy đủ của
  vật bị che.
- PhoXi Scanner Gen2 phù hợp với cảnh tĩnh; chuyển động trong lúc chiếu mẫu gây sai dữ
  liệu 3D.
- Vật gương, kim loại bóng, trong suốt hoặc hấp thụ ánh sáng có thể tạo lỗ point cloud;
  YOLO vẫn có thể nhận diện texture nhưng phép đo 3D có thể bị đánh dấu `INVALID`.
