# GraspNet offline cho PhoXi + Robotiq 2F-85

## 1. Phạm vi bản cập nhật

Bản này thêm đường chạy độc lập trên scan đã lưu. Nó dùng XYZ của PhoXi trong `frame.npz` và mask YOLO trong `masks.npz`, không chạy lại YOLO và không điều khiển robot. Luồng hiện tại trong `07_run_live.py` và `08_run_offline.py` vẫn chạy như trước.

- Đọc mask của các vật trong `OBJECTS` của **cùng** `result.json`, chỉ đưa vật đã `verified_3d=True` vào chọn tư thế gắp.
- Co mask để hạn chế điểm ở mép; bỏ cụm 3D nhỏ không nối được với bề mặt lân cận. Không dùng quy tắc cố định kiểu “Z dưới 900 mm là nhiễu”.
- Chuyển XYZ từ mm sang m trực tiếp. Không cần tái dựng point cloud bằng intrinsics của ảnh đã crop.
- Dùng checkpoint GraspNet baseline để đề xuất tư thế 6D; lọc độ mở phù hợp ngón tiêu chuẩn 2F-85 (85 mm, mặc định chừa thêm 4 mm).
- Kiểm tra điểm thuộc vật ở vùng giữa hai ngón và va chạm với **toàn bộ điểm cảnh quan sát được**, gồm cả thành thùng ngoài ROI.
- Lưu tối đa 5 đề xuất mỗi vật, vị trí và ma trận quay trong hệ tọa độ camera.

**Đây là đề xuất để kiểm tra.** Bộ kiểm tra va chạm dùng mô hình kẹp song song tổng quát của GraspNet; chưa có CAD của 2F-85, ngón chuyển động thích nghi, gá, cổ tay hoặc đường đi robot. Điểm không có depth vẫn là vùng chưa biết. `robot_ready=False` luôn được ghi trong kết quả.

## 2. Các file được thêm và vị trí

Giải nén vào thư mục gốc `~/phoxi_yolo_3d`. ZIP này chỉ chứa file mới:

| File | Đặt cùng cấp với |
|---|---|
| `phoxi_yolo3d/grasp_data.py` | `measurement.py`, `surface_reference.py` |
| `phoxi_yolo3d/grasp_selection.py` | `postprocess.py` |
| `phoxi_yolo3d/grasp_backend.py` | `detector.py` |
| `phoxi_yolo3d/grasp_visualization.py` | `visualization.py` |
| `scripts/13_run_grasp_offline.py` | `08_run_offline.py` |
| `scripts/14_check_grasp_env.py` | `00_diagnose.py` |
| `config/grasp.yaml` | `project.yaml`, `inspection.yaml` |
| `tests/test_grasp_offline.py` | Các file test khác trong `tests/` |
| `requirements-grasp-prepare.txt` | `requirements.txt` |
| `CAI_DAT_GRASPNET.md` | `README.md` |

Lệnh giải nén (giả sử trình duyệt lưu ZIP trong Downloads):

```bash
cd ~/phoxi_yolo_3d
unzip ~/Downloads/phoxi_graspnet_update.zip -d ~/phoxi_yolo_3d
```

## 3. Kiểm tra laptop trước khi chọn môi trường CUDA

Chạy trong `.venv` đang dùng và gửi nội dung in ra:

```bash
cd ~/phoxi_yolo_3d
python scripts/14_check_grasp_env.py --output results/grasp_env.json
```

Script chỉ đọc driver/GPU, PyTorch, `nvcc`, GCC và thử import baseline nếu đã cài. Lần đầu `checkpoint_exists=False` và `imports_ok=False` là bình thường. Mã thoát 2 có nghĩa môi trường chưa sẵn sàng; báo cáo vẫn được ghi.

GTX 1650 là cấu hình nên thử với **batch 1, một scan mỗi lần**. Số điểm bắt đầu là 20.000. Nếu forward thực tế báo hết VRAM, thử 16.384 hoặc 8.192 điểm; việc giảm số điểm có thể làm mất chi tiết vật nhỏ. Chưa đo tốc độ hay VRAM trên laptop của bạn, nên chưa cam kết khả năng chạy liên tục.

Baseline chính thức công bố môi trường PyTorch 1.6 và có extension PointNet++/KNN phải biên dịch. `.venv` của bạn đã báo PyTorch 2.14 + CUDA 13.0; điều này chưa chứng minh extension cũ chạy được. **Không hạ PyTorch trong `.venv` nhận diện.** Chọn phiên bản Python/PyTorch/CUDA toolkit tương thích sau khi có báo cáo trên, rồi cài vào môi trường riêng.

`CUDA Version` trên `nvidia-smi`, CUDA runtime của PyTorch và `nvcc --version` có ý nghĩa khác nhau. Có PyTorch CUDA wheel chưa có nghĩa đã có toolkit để build extension. Nếu đổi PyTorch/CUDA, extension cần build lại. Với GPU này, kiểm tra capability thực tế trước khi chọn `TORCH_CUDA_ARCH_LIST`.

## 4. Chạy kiểm tra dữ liệu ngay, chưa cần mạng/GPU

Có thể tạo môi trường dành riêng cho bước này:

```bash
cd ~/phoxi_yolo_3d
python3 -m venv .venv_grasp
source .venv_grasp/bin/activate
python -m pip install -r requirements-grasp-prepare.txt
```

Không cần `pip install -e .`: hai script mới tự thêm thư mục dự án vào Python path và không import PhoXi API/Ultralytics.

Đặt biến trỏ đến inspection gần nhất của bạn:

```bash
INSPECTION="$HOME/phoxi_yolo_3d/results/live_20261002T082612_232059Z/inspection_20261002T082816_573555Z"
python scripts/13_run_grasp_offline.py "$INSPECTION/scans/frame.npz" --prepare-only
```

Script tự tìm `masks.npz`, `result.json` và `reference/bin_roi.json` trong cùng inspection. File `result.json` phải thuộc cùng frame; script kiểm tra frame ID và timestamp nếu có.

Nếu chỉ có scan và masks, không có result.json, chọn rõ ID đã xác nhận là vật:

```bash
python scripts/13_run_grasp_offline.py /duong/dan/frame.npz --masks /duong/dan/masks.npz --instance-ids 0 1 2 --prepare-only
```

Trường hợp này không có tên lớp/confidence/3D verification gốc. Kiểm tra ảnh trước khi dùng inference. Không chọn ID của vết xước hoặc vật đã bị loại.

`--prepare-only` tạo **0 grasp** đúng chủ ý: nó kiểm tra dữ liệu chứ không chạy mạng. Kết quả nằm trong thư mục mới dưới `results/grasp_offline/`; mở `grasp_preview.png` để xem pixel bị loại màu tím. Màu tím gồm cả pixel thiếu depth và phần mép bị co, không phải tất cả đều là nhiễu.

## 5. Cài baseline và checkpoint sau khi chốt môi trường

Đường dẫn mặc định:

- Source: `~/phoxi_yolo_3d/third_party/graspnet-baseline/`
- Checkpoint: `~/phoxi_yolo_3d/models/graspnet/checkpoint-rs.tar`

Có thể lấy source bằng:

```bash
cd ~/phoxi_yolo_3d
mkdir -p third_party models/graspnet
git clone https://github.com/graspnet/graspnet-baseline.git third_party/graspnet-baseline
```

Tải `checkpoint-rs.tar` từ mục **Pretrained models** trong README chính thức và đặt vào `models/graspnet/`. Đây là mô hình được huấn luyện trên RealSense, chưa được kiểm chứng trên PhoXi; so sánh thực nghiệm trước khi quyết định dùng trong sản xuất. Có thể thử checkpoint Kinect bằng `--checkpoint`. ZIP này không chứa source bên thứ ba hoặc trọng số.

Sau khi chọn và cài đúng PyTorch/CUDA toolkit trong môi trường riêng, làm theo mục **Installation** của repository để:

1. Cài các dependency của baseline (Open3D, SciPy, Pillow, tqdm, TensorBoard…). Giữ bộ NumPy tương thích với PyTorch/API đã chọn; file requirements bước chuẩn bị không giải quyết toàn bộ môi trường GPU.
2. Build/install `pointnet2/` và `knn/` trong chính môi trường đó.
3. Cài [GraspNetAPI chính thức](https://github.com/graspnet/graspnetAPI).

Không đưa một lệnh cài Torch/CUDA cố định ở đây vì chưa có phiên bản driver/toolkit trên laptop và chưa test build các extension. Nếu gặp lỗi `THC/THC.h`, `sklearn`, ABI hoặc kiến trúc CUDA không được hỗ trợ, gửi log để chọn bản tương thích hoặc sửa extension; không tiếp tục coi môi trường đã đạt.

Kiểm tra cả CUDA **và forward thực tế**, không chỉ import:

```bash
python scripts/14_check_grasp_env.py --smoke --output results/grasp_env_smoke.json
```

Smoke test dùng cloud tổng hợp để kiểm tra kernel/checkpoint. Nó không lưu tư thế cho robot và không xác nhận chất lượng gắp. Chỉ khi `full_forward_tested=true` mới coi phần forward mạng đã chạy được trên laptop.

## 6. Chạy đề xuất tư thế trên scan của bạn

Sau khi smoke test thành công:

```bash
INSPECTION="$HOME/phoxi_yolo_3d/results/live_20261002T082612_232059Z/inspection_20261002T082816_573555Z"
python scripts/13_run_grasp_offline.py "$INSPECTION/scans/frame.npz" --visualize
```

Nếu phải giảm số điểm:

```bash
python scripts/13_run_grasp_offline.py "$INSPECTION/scans/frame.npz" --num-points 16384 --visualize
```

Nếu mất ROI, cung cấp ROI đúng với scan/crop bằng `--roi`. Chỉ dùng `--allow-full-frame` khi chấp nhận xử lý toàn bộ ảnh đã lưu. ROI không khôi phục thành thùng đã bị crop khỏi ảnh.

Các file kết quả:

| File | Nội dung |
|---|---|
| `grasp_results.json` | Thống kê từng mask, lý do loại, profile GPU, các pose đã chọn |
| `grasps.npy` | N×17 theo định dạng GraspNet, đơn vị mét |
| `prepared_points.npz` | Điểm thực đã lấy mẫu, điểm/mask sạch từng vật để kiểm tra |
| `grasp_preview.png` | Viền mask, vùng pixel bỏ, điểm gắp và hướng đóng ngón gần đúng |

Tên lớp và YOLO confidence lấy từ inspection. `grasp_score` là điểm xếp hạng của mạng grasp; không phải confidence YOLO hoặc xác suất chắc chắn gắp thành công. `translation_camera_mm` là điểm tham chiếu GraspNet; **chưa phải TCP robot**. `grasp_height_mm`/`grasp_depth_mm` là tham số hình học grasp, không phải H99 hay độ cao vật so với đáy thùng.

`T_camera_grasp_m` biến đổi điểm từ hệ grasp sang camera: `p_camera = R @ p_grasp + t`. Ma trận quay được giữ trực tiếp để tránh chọn nhầm quy ước góc Euler của robot. Hướng jaw/approach trong viewer theo quy ước GraspNet; cần xác nhận trên model/TCP thật trước khi nối robot.

Ảnh 2D dùng điểm XYZ gần nhất để hiển thị hướng jaw, không phải phép chiếu camera đã hiệu chuẩn. Dùng Open3D để kiểm tra pose đầy đủ trong 3D.

## 7. Kiểm tra đã thực hiện cho bản cập nhật

- 11 bài kiểm tra mới: loại cụm spike, giữ bề mặt lớn, depth rỗng, lấy mẫu điểm thực, mask bị reject và frame mismatch, giới hạn mở có margin, pose lỗi, vật không đủ support, nhiều vật giữa jaw, đơn vị và NMS đối xứng jaw.
- 36 bài kiểm tra cũ của dự án.
- Chạy `--prepare-only` trên scan 800×800 bạn gửi: mask 0/1/2 giữ 26.068 / 37.324 / 23.172 điểm, tương ứng 96,0% / 99,3% / 99,9% số điểm hợp lệ sau co mask. Ba mask vẫn đủ điều kiện đầu vào. Đây chưa chứng minh chúng có tư thế gắp phù hợp 2F-85.
- Chưa chạy checkpoint/CUDA thật: môi trường kiểm tra không có GPU, PyTorch, baseline hoặc trọng số. Chưa kiểm chứng va chạm hay gắp thật trên Robotiq.

Chạy lại test riêng sau khi giải nén nếu cần:

```bash
python -m unittest discover -s tests -p test_grasp_offline.py -v
```

## 8. Trước khi nối chuyển động robot

Cần biết model robot, camera gắn cố định hay trên cổ tay, cấu hình ngón và TCP. Sau đó hiệu chuẩn `T_base_camera` (hoặc hand-eye nếu camera trên cổ tay), đo biến đổi grasp-frame ↔ TCP, kiểm tra CAD kẹp/gá/thùng, IK và đường đi. Ma trận robot cuối cùng phụ thuộc cả calibration camera **và** grasp-to-TCP, không chỉ đổi hệ camera sang base.

Không dùng D, Hmax hoặc H99 làm tọa độ gắp Z trực tiếp. Nếu mask thiếu nhiều depth, cần chụp lại/đổi góc trước khi chọn pose.

Nguồn kỹ thuật:

- [GraspNet baseline chính thức: môi trường, cài đặt, pretrained checkpoint](https://github.com/graspnet/graspnet-baseline)
- [Demo baseline: tham số mạng và kiểm tra va chạm](https://github.com/graspnet/graspnet-baseline/blob/main/demo.py)
- [GraspNetAPI: định dạng 6D, N×17](https://graspnetapi.readthedocs.io/en/latest/grasp_format.html)
- [Robotiq: độ mở mặc định 2F-85 và cấu hình ngón thay đổi](https://blog.robotiq.com/knowledge/2f85-and-2f140-default-opening-capability-5-1736280842906)
