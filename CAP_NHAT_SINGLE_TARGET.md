# Cập nhật 0.2 — chọn một mục tiêu và đề xuất tư thế gắp trong hệ camera

Ngày đóng gói: 15/10/2026. Bản này ghép mã từ `phoxi_source_roi_3d.zip`,
`phoxi_roi_3d_update.zip`, `phoxi_graspnet_update.zip` đã được đối chiếu.
Không có trọng số YOLO, checkpoint GraspNet, scan camera thật hoặc hiệu chuẩn robot
trong các gói nguồn dùng để phát triển. Mã trên máy vận hành có thể có chỉnh sửa mới hơn.

## Chạy thử

Giải nén vào một thư mục mới. Dùng môi trường Python của dự án và chạy từ thư mục
có `pyproject.toml`:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/15_replay_picking.py examples/synthetic_inspection
```

Demo dùng dữ liệu **tổng hợp**, không cần camera, YOLO hay GPU. Kết quả nằm ở
`examples/synthetic_inspection/picking_replay/`: ảnh `target.png` và `picking.json`.
Trạng thái dự kiến là `PREVIEW_UNVALIDATED`, vì kích thước cơ khí để kiểm tra
tay kẹp chưa được cung cấp. Có hai vật trong dữ liệu; màn hình chính chỉ chú thích
vật được chọn. Thêm `--debug` để xem tất cả.

Để chạy thật, đối chiếu các giá trị camera, crop, đường dẫn `best.pt`, `imgsz`,
confidence và các thông số đang dùng với `config/project.yaml` của bản mới.
Giữ nguyên thứ tự năm lớp trong `classes.txt`. Không thay cấu hình máy đang chạy
bằng giá trị ví dụ khi chưa đối chiếu. Có thể trỏ `yolo.weights` đến file hiện có
bằng đường dẫn tuyệt đối, không cần chép lại trọng số.

```bash
python scripts/07_run_live.py --config config/picking.yaml
```

1. Để thùng trống. Bấm **R** để chọn ROI làm việc trong thùng.
2. Bấm **F** để chọn riêng vùng đáy phẳng nhìn thấy, tránh thành thùng. Đây là vùng
   để khớp mặt phẳng, không phải ROI phát hiện vật.
3. Bấm **B** để chụp 15 frame nền. Kiểm tra thông báo `FLOOR PLANE: READY`.
   Nếu khớp mặt phẳng thất bại, xem lý do; không tự chuyển sang đo theo Z trong chế độ này.
4. Đặt vật vào, bấm **Space/N** để chụp mới. **D** bật/tắt toàn bộ vật và thông tin
   loại bỏ. **S** lưu trọn gói bằng chứng. Sau thay đổi ROI/nền, cần chụp frame mới.
5. Chế độ chụp tự động/TCP vẫn dùng các phím 1–4; chưa có lệnh điều khiển robot.

`config/inspection.yaml` tiếp tục dùng luồng kiểm tra cũ, hiển thị tất cả vật.
Luồng chọn một vật chỉ bật với `picking.enabled: true`.

## Các thay đổi đã triển khai

| Phần | Hành vi mới |
|---|---|
| Cấu hình | Đưa các ngưỡng nền vào project.yaml; dùng `min_raised_ratio_mask: 0.02`, `min_raised_points: 30`. Khóa cũ/misspell và giá trị không hợp lệ ở các nhóm mới được báo lỗi. |
| Kiểm tra depth bằng chuột | Ánh xạ cả hai nửa Texture/Depth về đúng pixel và chặn tọa độ ngoài ảnh. |
| Auto-label | Kiểm tra số lớp, ID liên tục và đúng thứ tự trước khi truy cập tên lớp. |
| ROI | Điểm lõi nằm trong mask thay cho trung bình tọa độ khi kiểm tra tâm thuộc ROI. |
| Chiều cao | Khoảng cách có dấu vuông góc với mặt phẳng đáy; vẫn kiểm tra thay đổi so với nền trống để tránh coi thành thùng là vật mới. |
| Nền tham chiếu | Lưu camera, crop, hệ tọa độ, nguồn sáng và mã hiệu chuẩn; sai identity/cùng kích thước vẫn báo UNKNOWN. |
| Tầng sàng lọc | Thống kê 3D thô, ROI/nền, tỷ lệ dữ liệu hợp lệ; xếp theo tỷ lệ hợp lệ, H90 và confidence. Điểm xếp hạng không phải xác suất gắp thành công. |
| Tầng chi tiết | Xử lý từng ứng viên theo thứ tự, tối đa 3; chỉ đo đầy đủ cho mục tiêu cuối. Vật đầu thất bại thì thử hướng khác hoặc vật sau. |
| Làm sạch | Tách liên thông 3D bằng khoảng cách giữa điểm lân cận; giữ bề mặt chiếm ưu thế. Không cắt tùy ý phần gần camera nhất. |
| Điểm mốc gắp | Distance transform tìm lõi, sau đó chọn đúng một điểm 3D đã đo gần trung vị cục bộ. |
| Pháp tuyến | Đồng nhất dấu NormalMap hướng về camera, đối chiếu mặt phẳng cục bộ RANSAC/SVD; dùng SVD nếu NormalMap không đáng tin. |
| Hướng kẹp | OBB trên mặt tiếp tuyến; thử cạnh ngắn/cạnh dài và các hướng bổ sung với vật đối xứng. Vượt góc nghiêng thì loại, không ép góc. |
| Khẩu độ | Kiểm tra bề rộng đối xứng quanh điểm mốc, biên sai số và khe hở hai bên trước khi so với hành trình mở. |
| Khoảng trống | Thể tích ngón và lòng kẹp quét theo đường tiếp cận thẳng; giữ toàn bộ điểm cảnh, kể cả điểm bị loại khỏi phép đo vật. |
| Dữ liệu không quan sát | Lỗ depth, vùng bị che, thể tích ra ngoài ảnh hoặc phép chiếu không đạt sai số đều cho UNKNOWN. |
| Tuổi dữ liệu | Xuất thời điểm bắt đầu chụp, tuổi frame và hạn dùng; không chọn mục tiêu live từ frame quá hạn. Hình đang giữ cũng đổi trạng thái khi hết hạn. |
| Texture | CLAHE tùy chọn, mặc định tắt; cùng công thức cho thu ảnh và suy luận, không áp hai lần khi chạy lại. |
| Replay | Dùng chính mask đã lưu, không suy luận YOLO lại; tham chiếu tương đối, kiểm tra hash để phát hiện bộ dữ liệu bị trộn/sửa. |

Không tự gộp hai mask sát nhau và đồng phẳng. Chưa biết kích thước đầy đủ của vật
bị che nên không tuyên bố vật nguyên vẹn hoặc bề rộng thực đã được xác nhận.
`class_shapes` chỉ hỗ trợ chọn kiểu hình học và các góc cần thử. Với chai/trụ,
chế độ tiếp xúc nắp/thân và phân biệt đứng/nằm chưa được xác nhận bằng dữ liệu thật;
đầu ra ghi `cylinder_contact_mode: UNVERIFIED`.

## Đọc kết quả

- `measurement.center_xyz_mm`: trung vị từng trục để thống kê bề mặt thấy được.
- `picking.pose.surface_anchor_xyz_mm`: một điểm đo thật; pixel tương ứng là
  `surface_anchor_uv`. Đây chưa phải vị trí TCP hay tâm đóng ngón ở độ sâu kẹp.
- `rotation_camera`: ma trận 3×3, các **cột** lần lượt là x tiếp tuyến, y đóng/mở
  ngón, z tiếp cận vào vật. Ma trận ánh xạ vector từ hệ gắp hình học sang hệ camera.
  Đơn vị vị trí là mm. Không xuất Euler chưa xác định quy ước robot.
- `proposed_opening_mm` gồm bề rộng nhìn thấy tính đối xứng quanh điểm mốc,
  `width_uncertainty_mm` và hai lần `opening_gap_mm`.
- `robot_ready` luôn là **false**, kể cả khi phép kiểm tra tiếp cận đạt.
- `pipeline_timings_ms` tách chuẩn hóa, YOLO, các bộ lọc/thống kê, và phần chọn mục
  tiêu/đo chi tiết. `picking.timings_ms.rank` chỉ là bước xếp hạng, không phải toàn
  bộ tầng 1. Không có cam kết 1–2 ms; cần đo trên máy GTX 1650 và scan thật.

| Trạng thái | Ý nghĩa |
|---|---|
| `NO_TARGET` | Không đủ điều kiện hoặc mọi tư thế bị loại; xem reasons/ranking/attempts. |
| `PREVIEW_UNVALIDATED` | Có đề xuất hình học, nhưng khoảng trống chưa được xác minh. |
| `APPROACH_CHECKED_PROPOSAL` | Thể tích tiếp cận đã kiểm tra trong phần cảnh quan sát được với mô hình đã cấu hình. |
| `STALE_PROPOSAL` / `STALE_DISPLAY` | Frame hết hạn; không cung cấp mục tiêu live để dùng tiếp. |
| `RECAPTURE_REQUIRED` | Đã đổi ROI/nền trên frame cũ; cần chụp mới. |

Tuổi frame không chứng minh cảnh chưa chuyển động. Việc khóa chu kỳ robot,
đồng bộ vùng làm việc và xác nhận cảnh còn hiệu lực phải được nối ở bước tích hợp robot.

## Mô hình kiểm tra khoảng trống cần dữ liệu cơ khí

Trong `picking.clearance`, các kích thước chưa biết để `null` và
`geometry_verified: false`. Khi thiếu thông số, đầu ra là
`GRIPPER_GEOMETRY_UNCONFIGURED`; hệ thống vẫn cho xem đề xuất hình học.

Các hộp bao phải bao trùm cơ cấu thật tại khẩu độ đề xuất, gồm ngón, khớp/linkage
và lòng kẹp. Không điền số ước lượng để bật `geometry_verified`. Trong hệ gắp hình học:

- `finger_span_mm`: bề rộng ngón theo x; `finger_thickness_mm`: bề dày theo y.
- `finger_length_mm`: chiều dài ngón theo z; `insertion_depth_mm`: mức đầu ngón
  đi vào dưới mốc bề mặt theo +z, phải nhỏ hơn chiều dài ngón.
- `palm_span_mm`, `palm_width_mm`, `palm_depth_mm`: kích thước lòng kẹp theo x/y/z.
- `vertical_motion_envelope_mm`: phần bao thêm để tính đến độ dịch dọc z của cơ cấu.
- `approach_mm`: chiều dài tiếp cận thẳng từ phía -z; `safety_margin_mm`: nới hộp bao.

Phép kiểm tra dùng mọi điểm đo để tìm vật cản và kiểm tra tia nhìn xuyên qua hộp
bao tới nền quan sát được. Phép chiếu ảnh được ước lượng từ XYZ đã tổ chức và kiểm
tra sai số trên mọi điểm hợp lệ; nếu không đạt thì UNKNOWN. Phép chiếu này phục vụ
kiểm tra/hiển thị, không thay thế hiệu chuẩn camera hay hand-eye.

Chưa có xác nhận quỹ đạo đóng ngón với tiếp xúc, rút vật đang cầm, cổ tay/toàn bộ
tay máy và giới hạn khớp. Các trường `closing_status`, `retreat_status`, `arm_status`
ghi rõ phần chưa kiểm chứng. Điều kiện này không được nâng thành “an toàn tuyệt đối”.

## Tương thích GraspNet và dữ liệu cũ

`13_run_grasp_offline.py`, checkpoint/API và kết quả GraspNet giữ luồng riêng.
Backend mới trong live là `deterministic_surface_geometry`; không tự tải/chạy
GraspNet trên GPU. Hàm `geometry.to_graspnet_rotation()` chuyển **hướng** từ quy ước
mới sang GraspNet (+x approach, +y closing). Vị trí tham chiếu của hai backend khác
nhau; không dùng hàm này để suy ra TCP hoặc chuyển trực tiếp translation.

```bash
python scripts/15_replay_picking.py /duong/dan/inspection --config config/picking.yaml
```

Nếu bộ inspection cũ chưa có identity/mặt phẳng, chế độ chuẩn sẽ báo thiếu dữ liệu.
Để chỉ xem thử hình học trên dữ liệu cũ, dùng chế độ có tên rõ ràng:

```bash
python scripts/15_replay_picking.py /duong/dan/inspection --config config/picking_legacy_preview.yaml
```

Chế độ này dùng camera-Z và cho phép nền chưa có identity; chỉ dùng để phân tích
ngoại tuyến. `reference.identity_status` cho biết giới hạn đó. Với vận hành mới,
vẽ vùng đáy và chụp lại nền.

Khi thay camera/crop/vị trí thùng/hiệu chuẩn, tăng `camera.calibration_id` rồi chụp
lại nền. Mã không thể tự phát hiện camera bị dịch chuyển vật lý nếu cấu hình và mã
này đều không đổi.

## CLAHE và tập huấn luyện

Mặc định tắt để giữ dữ liệu đầu vào của model đang có. Khi bật, ảnh thu mới và ảnh
suy luận dùng cùng công thức. Ảnh PNG huấn luyện cũ không tự biến đổi theo cấu hình;
cần chuẩn bị lại dữ liệu cùng công thức và đánh giá trên cùng các cảnh đã gán nhãn.
Nếu replay không có texture_raw và công thức trước đó khác công thức mới, chương
trình báo lỗi thay vì tăng tương phản chồng nhiều lần.

## Dữ liệu cần cho lần tích hợp kế tiếp

1. **Mã nguồn trên máy hiện tại**: thư mục `phoxi_yolo3d`, `scripts`, `config`,
   `pyproject.toml` và requirements. Có thể bỏ môi trường ảo, cache, runs và ảnh lớn.
   Cần xác nhận có sửa gì sau ba gói nguồn nêu đầu tài liệu hay không.
2. **Các bộ inspection nguyên vẹn** lưu bằng S: thùng trống, hai hộp sát đồng phẳng,
   hộp bị che một phần, trụ/chai đứng và nằm, vật sát thành, cảnh thiếu/nhiễu depth.
   Giữ `scans/frame.npz`, `masks.npz`, `result.json`, `replay.yaml`, thư mục reference
   và manifest nếu có. 5–10 cảnh đa dạng là đủ cho vòng hiệu chỉnh đầu.
3. **Cơ khí và hiệu chuẩn**: xác nhận ngón Robotiq tiêu chuẩn hay ngón tùy biến;
   CAD/bản vẽ và kích thước thực, adapter, vị trí TCP, chiều sâu ôm mong muốn;
   kích thước từng lớp vật; camera cố định hay gắn cổ tay; model robot và phép biến
   đổi camera–robot (tên hệ trục, chiều biến đổi, đơn vị).
4. **Đánh giá YOLO/tốc độ**: cần `best.pt` đang dùng và đúng imgsz/confidence nếu
   muốn kiểm tra lại suy luận/CLAHE. Frozen-mask replay không cần trọng số.

Gói hiện tại đã kiểm thử bằng cảnh tổng hợp và camera/GUI giả lập. Chưa thử PhoXi
thật, chuyển động Robotiq, tải checkpoint GraspNet hoặc đo độ chính xác gắp thực tế.
