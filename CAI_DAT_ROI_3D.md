# PhoXi — kiểm tra từng ảnh, lọc 3D và nhận tín hiệu chụp

Bản cập nhật 01/10/2026. Hướng dẫn này thay cho hướng dẫn ROI trước đó.

## 1. Kết luận từ ba ảnh bạn gửi

- Cả ba ảnh đều hiện `SURFACE: MISSING`: chưa có nền thùng trống được nạp. Ở trạng thái này, bộ lọc chiều cao chưa chạy.
- Khối lập phương vẫn được YOLO nhận diện. Các dòng `0<80`, `14<80`, `18<80` là số điểm đủ điều kiện đo 3D so với ngưỡng 80 điểm trong `measurement`; không phải chiều cao thấp hơn đáy thùng.
- Ảnh thứ ba hiện `REJECT=0`: khối trụ không bị hậu xử lý loại. Cần kiểm tra dự đoán YOLO trên ảnh gốc và ngưỡng confidence.
- Hai mask Anibox trên cùng một hộp là vấn đề dự đoán trùng/phân mảnh. Lọc theo độ chồng lấn có thể xử lý một phần, nhưng không bảo đảm sửa được hai mask rời nhau.

## 2. Cài và chạy

Dừng chương trình cũ, giải nén trong thư mục gốc dự án:

```bash
cd ~/phoxi_yolo_3d
unzip -o ~/Downloads/phoxi_roi_3d_update.zip
source .venv/bin/activate
python -m unittest discover -s tests -v
python scripts/07_run_live.py --config config/inspection.yaml --weights runs/segment/phoxi_objects-2/weights/best.pt
```

Gói cập nhật không chứa `config/project.yaml`, model, dataset hoặc dữ liệu hiệu chuẩn của bạn. `config/inspection.yaml` bổ sung cấu hình mới vào cấu hình dự án hiện có, bao gồm confidence 0.25. Các tham số camera, crop, đo 3D và đường dẫn khác tiếp tục lấy từ `project.yaml`.

Mặc định chương trình chụp đúng **một ảnh lúc khởi động**, xử lý rồi giữ nguyên. Nhấn `Space` hoặc `N` để chụp ảnh tiếp theo. Phím chỉ có tác dụng khi cửa sổ PhoXi YOLO 3D đang được chọn.

Nếu chưa có ROI, vẽ bằng chuột trái; chuột phải hoặc Enter để đóng đa giác; `S` lưu; `U` lùi một điểm; `C` xóa; Esc hủy. ROI nằm trong ảnh đã crop 800×800.

Bỏ hết vật khỏi thùng, giữ yên camera/thùng rồi nhấn `B`. Chương trình chụp 15 lần để tạo nền. Chờ thông báo lưu thành công và kiểm tra tỷ lệ 3D nền hợp lệ. Đặt vật vào, nhấn `Space` để chụp một ảnh mới. Sau `B`, ảnh kiểm tra đang giữ được đánh giá lại với nền mới; nó không tự trở thành ảnh mới của cảnh hiện tại.

## 3. Các chế độ chụp

| Phím | Chế độ | Hành vi |
|---|---|---|
| `1` | Manual | Giữ ảnh; `Space/N` mới chụp tiếp |
| `2` | Continuous | Chụp tiếp sau mỗi lần xử lý |
| `3` | Interval | Sau mỗi lần xử lý, nghỉ 2 giây rồi chụp tiếp |
| `4` | Trigger | Chờ lệnh TCP `TRIGGER`, một lệnh tạo một ảnh |
| `P` | Tạm dừng/tiếp tục | Dừng lịch chụp tự động và không nhận trigger; ảnh đang chụp sẽ hoàn tất |
| `Space/N` | Một lần | Chụp đúng một ảnh, kể cả khi đã tạm dừng |
| `S` | Lưu ảnh kiểm tra | Lưu khung hình đang xem và dữ liệu gốc; không chụp lại |
| `D` | Chi tiết | Hiện mask bị loại, lý do và số điểm 3D của các vật |
| `M` | Tô mask | Bật/tắt màu trong mask, vẫn giữ đường viền |
| `R` | ROI | Vẽ lại vùng nhận diện |
| `B` | Nền | Chụp 15 ảnh thùng hoàn toàn trống |
| `Q/Esc` | Thoát | Đóng chương trình |

Các lệnh dưới dùng đường dẫn model ở `config/project.yaml`; nếu model chưa ở đó, thêm `--weights runs/segment/phoxi_objects-2/weights/best.pt` như lệnh đầu tiên.

Chạy theo chu kỳ với thời gian nghỉ 5 giây:

```bash
python scripts/07_run_live.py --config config/inspection.yaml --mode interval --interval 5
```

Chạy liên tục không giao diện, dừng sau 20 ảnh:

```bash
python scripts/07_run_live.py --config config/inspection.yaml --mode continuous --headless --count 20
```

`--count 1` sẽ đóng chương trình sau một ảnh; nếu muốn giữ cửa sổ để xem, dùng chế độ `manual` và không đặt `--count`.

## 4. Chụp theo tín hiệu

Cửa sổ terminal thứ nhất:

```bash
cd ~/phoxi_yolo_3d
source .venv/bin/activate
python scripts/07_run_live.py --config config/inspection.yaml --mode trigger
```

Chế độ này **không chụp ảnh khi khởi động**. Nó chờ tín hiệu. Terminal thứ hai:

```bash
cd ~/phoxi_yolo_3d
source .venv/bin/activate
python scripts/12_send_trigger.py
```

Mỗi lần chạy lệnh trên gửi một yêu cầu. Máy nhận trả JSON `queued` với mã yêu cầu, sau đó `done` với ID khung hình, nhãn, phép đo, số vật đã xác nhận 3D và lý do loại. Nếu đang xử lý yêu cầu trước, trả `busy`; nếu tạm dừng hoặc ở chế độ khác, trả `not_ready`. Yêu cầu bị từ chối không được xếp hàng và không tự gửi lại. Nếu kết nối bị đứt sau khi đã nhận `queued`, lần chụp đã nhận vẫn có thể hoàn tất; không tự phát lại tín hiệu để tránh chụp hai lần.

Kiểm tra trạng thái mà không chụp:

```bash
python scripts/12_send_trigger.py --status
```

Mặc định TCP ở `127.0.0.1:8765`, dùng giữa các chương trình trên cùng máy. Phần mềm khác có thể gửi chuỗi ASCII `TRIGGER\n` rồi đọc từng dòng JSON. Để nhận từ một máy khác trong mạng, chọn địa chỉ LAN của máy PhoXi bằng `--trigger-host DIA_CHI_IP_MAY_PHOXI`, và truyền địa chỉ đó cho `12_send_trigger.py --host ...`.

Đây là **tín hiệu phần mềm TCP**. Chưa cấu hình chân trigger điện của scanner hay đầu vào PLC. PLC cần gửi được TCP theo giao thức này hoặc có chương trình cầu nối từ tín hiệu của PLC.

## 5. Phép kiểm tra 3D mới

Tại mỗi pixel có dữ liệu hợp lệ ở cả scan hiện tại và nền thùng trống:

```text
h = Z_nền - Z_hiện_tại
ngưỡng_pixel = max(5 mm, 3 × nhiễu nền tại pixel)
N_nổi = số điểm trong mask có h >= ngưỡng_pixel
P_mask = N_nổi / tổng số pixel của mask
```

Biên mask được co 3 pixel khi lấy bằng chứng để hạn chế lẫn điểm thành thùng; mẫu số vẫn là diện tích mask gốc. Pixel thiếu depth không tạo bằng chứng nổi cao hoặc bằng chứng nằm trên nền.

| Trường hợp | Kết quả |
|---|---|
| Ít nhất 30 điểm nổi và `P_mask >= 2%` | `SURFACE PASS`: có bằng chứng nổi cao |
| Chưa đủ điểm nổi nhưng ít nhất 50% mask có cặp điểm hiện tại–nền hợp lệ | `NEAR_BIN_SURFACE`: loại |
| Thiếu cặp điểm 3D, nền khuyết, hoặc khác kích thước nền | `SURFACE UNKNOWN`: giữ nhãn 2D, chưa xác nhận 3D |
| Chưa có nền tham chiếu | `SURFACE MISSING`: giữ nhãn 2D, chưa xác nhận chiều cao |

Ví dụ mask có 10.000 pixel: cần ít nhất 200 điểm nổi để đạt 2%. Nếu có 300 điểm đo hợp lệ và cả 300 đều nổi cao, vật có thể đạt kiểm tra nền dù phần lớn mask thiếu depth. Nếu chỉ có 14 điểm hoặc không có điểm nào, chương trình giữ nhãn 2D nhưng không khẳng định có tọa độ/chiều cao đáng tin.

Ngưỡng `2%`, `30 điểm`, `5 mm` là điểm khởi đầu để kiểm tra trên scene của bạn; chưa được hiệu chỉnh bằng point cloud thực từ ba ảnh. Điều chỉnh trong `config/inspection.yaml`:

```yaml
postprocess:
  surface:
    min_height_mm: 5.0
    min_raised_ratio_mask: 0.02
    min_raised_points: 30
    min_valid_points: 30
    min_valid_ratio: 0.50
```

Bộ đo tọa độ vẫn dùng `measurement.min_valid_points: 80` từ cấu hình dự án. `3D-confirmed` chỉ tăng khi **cả bộ đo và kiểm tra nền đều đạt**. Trong JSON, dùng `verified_3d` để phân biệt với `postprocess.valid` (có giữ nhãn 2D hay không). Vì vậy một nhãn được giữ không đồng nghĩa có phép đo 3D hợp lệ.

`h` ở đây là chênh lệch theo trục Z của camera; khi camera nhìn nghiêng, nó không phải khoảng cách vuông góc chính xác tới mặt đáy. Phải chụp lại nền nếu camera, thùng hoặc crop bị dịch chuyển.

Trên ảnh và trong JSON, chương trình trả thêm hai số theo mm khi đủ cặp điểm
`PointCloud` hiện tại–nền tại cùng pixel trong mask:

- `Hmax` / `postprocess.surface_height_max_mm`: giá trị `max(Z_nền - Z_hiện_tại)`
  trong các pixel hợp lệ sau khi co biên mask. Một điểm nhiễu có thể làm số này tăng mạnh.
- `H99` / `postprocess.surface_height_p99_mm`: phân vị 99 của cùng các chênh lệch,
  để đối chiếu khi `Hmax` là một điểm bất thường. Hai số có trong JSON cùng với
  `surface_valid_points` và `surface_status`; ảnh chỉ hiện chúng khi `SURFACE PASS`.

Đây là độ cao của các điểm bề mặt **nhìn thấy được** theo trục Z camera so với ảnh
thùng trống, không đo phần bị che hoặc thay thế được phép đo theo pháp tuyến mặt
đáy. Nếu không có nền, thiếu dữ liệu 3D hoặc lệch kích thước thì giá trị là `null`.
`Z` trên nhãn là trung vị Z trong mask. `D` là trung vị của
`sqrt(X² + Y² + Z²)` trên các điểm XYZ hợp lệ trong mask; mã nguồn tự tính từ
`PointCloud` PhoXi, không lấy nguyên một trị `D` từ PhoXi Control hay `DepthMap`.

## 6. Vùng đen và khối lập phương

Phần xem 3D mới hiển thị đúng `PointCloud[..., 2]` mà thuật toán đang dùng. Đen nghĩa là điểm XYZ không hợp lệ hoặc Z không dương; không phải màu thật của vật và không phải quy ước "vật thấp". Thanh trạng thái báo riêng:

- `ROI valid 3D`: tỷ lệ điểm camera trả về hợp lệ trong ROI.
- `in-range`: tỷ lệ còn lại sau giới hạn Z trong `measurement`.
- `background`: tỷ lệ nền có điểm 3D hợp lệ trong ROI.

Ảnh texture sáng rõ vẫn có thể thiếu dữ liệu 3D. Tài liệu PhoXi mô tả điểm không đo được có thể là `(0,0,0)`; bóng chiếu, thiếu tín hiệu và phơi sáng quá mức đều có thể làm mất điểm. Chưa thể kết luận nguyên nhân riêng của các khối lập phương chỉ bằng ảnh chụp màn hình.

Hãy lưu một cảnh thùng trống và một cảnh có khối lập phương bằng `S`. Các file NPZ giúp phân biệt điểm camera không trả về, điểm bị giới hạn Z loại, và vị trí thiếu nền đối chiếu. Không nội suy các vùng đen để biến chúng thành điểm đo thật. Nếu nền vẫn thiếu quá nhiều điểm, cần chỉnh điều kiện quét 3D rồi chụp lại nền; giảm ngưỡng đếm không tạo thêm dữ liệu.

Tài liệu hãng: https://docs.photoneo.com/docs/3d_sensors/1.17/software/PXC/PXC_manual.html

## 7. Nhãn Anibox trùng và khối trụ bị thiếu

Bộ lọc mới giữ mask có confidence cao hơn khi hai mask cùng lớp có IoU >= 0.45 hoặc diện tích giao >= 80% diện tích mask nhỏ hơn. Hai mask rời nhau không bị gộp chỉ vì cùng lớp. Hai vật thật che nhau mạnh vẫn cần kiểm tra; có thể tắt `postprocess.duplicates.enabled` để so sánh trên cùng scan.

Chưa có model `best.pt` và dữ liệu raw của các ảnh này trong gói mã nguồn nên không thể xác nhận bộ lọc đã sửa được chính cặp Anibox bạn gửi. Nếu YOLO tách một hộp thành hai mask gần như không giao nhau, cần thêm dữ liệu gán nhãn các tư thế/che khuất này để cải thiện mô hình.

Với khối trụ ảnh 3, dùng scan đã lưu để thử ngưỡng thấp hơn, tách lỗi confidence khỏi lỗi mô hình:

```bash
python scripts/08_run_offline.py DUONG_DAN_INSPECTION/scans/frame.npz --config DUONG_DAN_INSPECTION/replay.yaml --conf 0.10 --debug
```

Có thể thử thêm `--imgsz 640` để so sánh với kích thước đã dùng khi train. Đây là phép kiểm tra; hạ confidence có thể tăng nhận nhầm. Nếu vẫn thiếu khối trụ, bổ sung ảnh gốc có khối trụ tối/che khuất vào bộ train/val; giữ bộ test độc lập để đánh giá lại. Không tăng confidence toàn cục để sửa nhãn hộp trùng trong khi recall khối trụ đang yếu.

Tài liệu Ultralytics về `conf`, `iou`, `imgsz`: https://docs.ultralytics.com/modes/predict/

## 8. Dữ liệu lưu bằng S

Mỗi lần nhấn `S`, tạo một thư mục `results/live_.../inspection_.../` gồm:

| File/thư mục | Nội dung |
|---|---|
| `dashboard.png` | Màn hình kết quả đang xem |
| `images/frame.png` | Ảnh texture gốc đưa vào YOLO |
| `scans/frame.npz` | Texture, point cloud và các map camera có trả về |
| `masks.npz` | Mask của từng ID, gồm cả các mask bị loại |
| `result.json` | Nhãn, confidence, phép đo, tỷ lệ 3D, lý do giữ/loại |
| `reference/` | Bản sao ROI và nền đang dùng, nếu có |
| `replay.yaml` | Cấu hình để chạy lại đúng nền và tham số |

Terminal in sẵn lệnh chạy lại. Nếu di chuyển cả thư mục sang máy khác, cập nhật đường dẫn model và đường dẫn `reference/` trong `replay.yaml`. Nén cả thư mục `inspection_...` để gửi kiểm tra, thay vì chỉ gửi ảnh màn hình.

## 9. File mới và file sửa

| File | Thao tác/vị trí |
|---|---|
| `phoxi_yolo3d/acquisition.py` | Mới, cùng cấp `camera.py`: lịch chụp và nhận tín hiệu TCP |
| `phoxi_yolo3d/duplicates.py` | Mới, cùng cấp `detector.py`: lọc mask trùng |
| `phoxi_yolo3d/diagnostics.py` | Mới, cùng cấp `measurement.py`: thống kê dữ liệu 3D |
| `config/inspection.yaml` | Mới, cùng cấp `project.yaml`: cấu hình bổ sung |
| `scripts/12_send_trigger.py` | Mới, cùng cấp `07_run_live.py`: gửi tín hiệu chụp |
| `scripts/07_run_live.py` | Sửa: chế độ chụp, trạng thái và lưu dữ liệu |
| `scripts/08_run_offline.py` | Sửa: chạy lại và xem chi tiết |
| `phoxi_yolo3d/surface_reference.py` | Sửa: bằng chứng nổi cao và trạng thái chưa đủ dữ liệu |
| `phoxi_yolo3d/postprocess.py` | Sửa: chuyển trạng thái bộ lọc |
| `phoxi_yolo3d/types.py` | Sửa: lưu tỷ lệ/đếm điểm/trạng thái mới |
| `phoxi_yolo3d/pipeline.py` | Sửa: lọc mask trùng và tách kết quả |
| `phoxi_yolo3d/visualization.py` | Sửa: contour, trạng thái 2D/3D và bảng điều khiển |
| `tests/test_*.py` | Các ca kiểm tra không cần camera thật |

Gói có kèm `roi.py` và test ROI từ bản trước để cài được cả lên mã nguồn ban đầu. Không cần sửa `detector.py`, `camera.py`, `measurement.py` cho cập nhật này.

Đã kiểm tra 36 ca, gồm dữ liệu 3D thưa, độ cao theo mask, nhãn trùng, TCP và luồng manual với camera giả lập. Đã kiểm tra vẽ bằng OpenCV 4.14.0.94. Chưa chạy trên thiết bị PhoXi thật và chưa chạy model của bạn trên ảnh gốc mới.
