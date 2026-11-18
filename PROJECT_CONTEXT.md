# PhoXi_3D — hồ sơ bàn giao sang chat mới

Cập nhật theo cuộc trao đổi đến ngày **06/11/2026, múi giờ Asia/Saigon**. Tài liệu này ghi lịch sử và quyết định thiết kế. **Giá trị thực trên máy người dùng phải đọc lại từ mã, cấu hình và log được xuất cùng gói; không coi mọi đề xuất dưới đây là đã cài hoặc đã thử thành công.**

## 1. Đọc gì trước ở chat mới

1. Đọc tài liệu này, `HANDOVER_SNAPSHOT.json` và `HANDOVER_MANIFEST.json` nếu có.
2. Đọc mã nguồn trong ZIP mới do người dùng xuất trên Ubuntu, đặc biệt `config/project.yaml`, `config/picking_robot.yaml`, `config/continuous.yaml` và các tệp tham chiếu calibration/điểm dạy.
3. Đọc `events.jsonl`, `contract.json`, `configuration.json`, `scan_*/result.json` của phiên liên tục mới nhất; xem mã DRL trong `generated/` nếu có.
4. Phân biệt: đề xuất trong chat, giá trị hiện có trên đĩa, mã đã xuất và mã thực sự đã dán trên pendant. Một tệp `generated/*.drl` không chứng minh pendant đang chạy đúng nội dung đó.
5. Tóm tắt ngắn trạng thái đã hiểu trước khi sửa. Không yêu cầu người dùng kể lại toàn bộ nếu gói đã đủ thông tin.

README gốc được viết từ giai đoạn chỉ có proposal camera nên có thể còn nói chưa hiệu chỉnh/TCP. Các phần mở rộng đã có calibration, chuyển TCP sang BASE và điều phối robot. Đối chiếu thời điểm và mã đang chạy, không dùng một đoạn README cũ để kết luận dự án chưa kết nối robot.

## 2. Mục tiêu và kiến trúc đang dùng

Hệ thống gắp vật trong thùng bằng **PhoXi 3D Scanner M cố định bên ngoài**, YOLO instance segmentation và robot **Doosan M0609 + Robotiq 2F85**. Vật được chọn từ ảnh mới, robot gắp, nâng qua vành thùng, chuyển đến vị trí thả trên băng tải và trở về điểm chờ để lặp.

Nhánh đang vận hành thực tế là **Python/Tkinter trên Ubuntu + socket Ethernet + DRL trong Task Writer**. Chương trình PC xử lý camera, nhận dạng, hình học và cấp lệnh theo từng bước; Task Writer thực hiện MoveL và Modbus. ROS2/MoveIt từng được thảo luận trong dự án khác, **chưa phải bộ điều khiển của luồng gắp liên tục hiện tại**.

Không có bộ lập quỹ đạo tránh va chạm toàn bộ tay máy. Kiểm tra khoảng hở của đầu kẹp trong dữ liệu nhìn thấy không tương đương chứng minh toàn tuyến không va chạm.

## 3. Phần cứng, môi trường và mạng

| Thành phần | Thông tin đã biết |
|---|---|
| Máy chạy vision | Acer Nitro AN515-57, Ubuntu 22.04, Python trong `.venv` |
| Thư mục dự án | `/home/noridomi-05/phoxi-single-target` hoặc `~/phoxi-single-target` |
| Robot | Doosan M0609, bộ điều khiển dòng 2.9.4, DART hiển thị GV02090400/GF02090400 |
| Tay kẹp | Robotiq 2F85, Modbus TCP, IP từng cấu hình `192.168.137.12` |
| TCP trên pendant | Tên chính xác `2F 85` (có dấu cách), phải cho phép đổi qua config |
| Robot IP | `192.168.137.100` |
| PC IP dùng socket | `192.168.137.50`; địa chỉ `.51` trong các bản cũ đã đổi |
| NIC từng kiểm tra | `enp2s0`, route tới robot chọn src `.50` |
| Scanner | ID `JEJ-043`, IP từng đọc `192.168.0.2` |
| PhoXi | PhoXi Control 1.17.0 trên Ubuntu; firmware scanner trong ảnh 1.2.34 |
| Hệ tọa độ camera | `CameraSpace/PrimaryCamera`, đơn vị mm |
| Pose robot | BASE, XYZABC, mm/độ, Euler nội tại ZYZ |

Lỗi scanner `Non routable address` trước đây đã xử lý bằng thêm địa chỉ NIC phù hợp với mạng camera. Không cần đổi IP robot chỉ vì scanner nằm ở subnet khác. Không suy luận cấu hình mạng hiện tại từ một file ví dụ cũ.

## 4. Trạng thái thực tế: đã làm tới đâu

| Hạng mục | Bằng chứng và trạng thái |
|---|---|
| Eye-to-hand PhoXi → BASE | Đã thu dữ liệu và tính kết quả được đánh giá trong ngưỡng cấu hình |
| Preview pose sang robot | Đã nhận ACK qua LAN, PC `.50`, robot `.100` |
| Pregrasp | Đã đưa đủ XYZABC tới robot; người dùng xác nhận vị trí trên vật đúng |
| Hạ tới vị trí gắp | Người dùng hạ tổng cộng khoảng 50 mm từ pregrasp; xác nhận phù hợp |
| Kẹp và nâng riêng | Người dùng xác nhận kẹp và nhấc lên tốt |
| Tuyến chờ/thả | Đã dạy P_WAIT, P_DROP_UP, P_DROP và chạy thử tuyến thành công |
| Chu trình gắp-thả thử từng đoạn | Người dùng báo chạy hết, không báo lỗi |
| GUI gắp liên tục | Đã nhận dạng và **hoàn thành ít nhất 1 chu trình thực**; ảnh GUI hiển thị bộ đếm 1 |
| Lượt liên tục thứ hai | Dừng sau đóng kẹp: `CONT_NO_GRIP_CONTACT_CHECK_CLOSE_PR_AND_OBJECT_WIDTH` |
| Kiểm tra sau lỗi lượt hai | Người dùng tắt task, Jog tăng Z và mang được vật lên: lượt đó có giữ vật thực tế |
| PR đóng 255 + force 30 | Đã hướng dẫn chỉnh và kiểm tra logic bằng mô phỏng; **chưa có xác nhận thử thành công trên robot sau thay đổi này** |
| Tuổi ảnh 20 giây | Đã hướng dẫn; ảnh GUI sau đó hiển thị `3.6s / 20.0s`, cho thấy đã áp dụng ở phiên đó |
| Chuyển vùng 150 mm/s, 20°/s, acc 300/40 | Đã đưa lệnh chỉnh theo yêu cầu; chưa có số đo tốc độ thực hoặc xác nhận cấu hình cuối cùng |

Gói phần mềm liên tục mới nhất đã được kiểm thử **199 bài** gồm mô phỏng socket, điều kiện chặn chuyển động và Tk GUI trên màn hình ảo. Đây là kiểm thử phần mềm, không thay thế nghiệm thu trên robot. Một số cờ JSON được tạo trước đó còn `full_pick_cycle_verified_on_real_robot: false`; không tự sửa cờ thành true hoặc khẳng định mọi tình huống đã được xác minh chỉ từ báo cáo một lượt thành công.

## 5. Calibration đang dùng

Camera cố định, checkerboard gắn cứng vào cụm robot, nhập **pose mặt bích trong BASE**, không nhập pose TCP gắp thay cho mặt bích. Solver dùng quan hệ:

```text
T_base_flange[i] @ T_flange_board = T_base_camera @ T_camera_board[i]
```

Mỗi ảnh dùng **texture 2D để tìm góc và point cloud 3D để khôi phục hình học bảng**. GUI chỉ hiện ảnh texture không có nghĩa calibration chỉ dùng 2D. Phiên PhoXi được đọc trong bản nguồn có 7 × 9 góc trong, cạnh ô 20 mm. Không trộn với bộ RealSense cũ của người dùng có bảng và kích thước ô khác.

Kết quả từng kiểm tra: 17 mẫu fit, 5 mẫu validation độc lập. Fit RMS vị trí khoảng **0.504 mm**, max **0.951 mm**, RMS góc **0.153°**. Validation RMS vị trí khoảng **0.415 mm**, max **0.674 mm**. Mẫu V0003 từng sai gần 100 mm, đã bỏ chọn và bổ sung mẫu khác. Không dùng kết quả trước khi loại mẫu sai.

Tệp thường dùng: `config/phoxi_camera_to_base.json`; lấy đường dẫn thực từ `picking.robot_target.calibration_path`. Trạng thái cần đọc là `VERIFIED_WITHIN_CONFIGURED_LIMITS`, không phải cam kết độ chính xác tuyệt đối cho cả bài toán gắp.

Ma trận đã xuất (chỉ để nhận biết, dùng JSON gốc đầy đủ độ chính xác khi tính):

```text
T_base_camera ≈
[-0.035266  0.998710  0.036525  -341.359811]
[ 0.977618  0.042060 -0.206142   493.375931]
[-0.207413  0.028438 -0.977840  1025.257482]
[ 0         0         0           1       ]
```

T_base_camera không đổi chỉ vì đổi TCP nếu camera/base vẫn giữ nguyên. Đổi TCP vẫn cần cập nhật bù TCP–tâm tiếp xúc, kiểm tra hướng và dạy lại/đổi các pose cố định vốn được lưu theo TCP cũ. Khi camera/base thay đổi hình học phải hiệu chỉnh lại theo đúng nguyên nhân.

## 6. Cấu trúc mã và trách nhiệm

Đây là bản đồ của mã đã đọc. Gói xuất từ máy người dùng quyết định cấu trúc hiện hành nếu có khác biệt.

| Đường dẫn | Vai trò |
|---|---|
| `config/project.yaml` | Cấu hình nền; được deep-merge với YAML chỉ định |
| `config/picking.yaml` | Cấu hình run_live/picking ban đầu |
| `config/picking_robot.yaml` | Overlay pose TCP, calibration và tham số gắp đang dùng |
| `config/continuous.yaml` | IP/cổng, camera policy, số lượt, tốc độ, Modbus/PR, output |
| `config/doosan_taught_points.json` | TCP, XYZABC, joint pose, solution space của điểm cố định |
| `config/bin_roi.json`, `config/bin_floor_roi.json` | ROI vùng thùng và đáy |
| `data/calibration/bin_surface.npz` | Nền thùng trống; đường dẫn có thể thay trong config |
| `models/best.pt`, `config/classes.txt` | Model segmentation và thứ tự lớp |
| `scripts/07_run_live.py` | Giao diện live; chụp, chọn mục tiêu, lưu bằng S |
| `scripts/16_calibrate_gui.py` | GUI calibration: thiết lập, thu mẫu, tính, kiểm chứng |
| `scripts/17_prepare_robot_config.py` | Chuẩn bị picking_robot.yaml; bản cũ dùng chế độ tạo mới nên báo FileExistsError nếu đã có |
| `scripts/18_preview_robot.py` | Preview truyền pose, không chuyển động |
| `scripts/19_pregrasp_test.py` | Thử pregrasp từ ảnh lưu, server cổng 20003 |
| `scripts/20_pick_cycle_test.py` | Chu trình thử một lượt trước khi làm liên tục |
| `scripts/21_run_continuous.py` | Mở GUI gắp liên tục |
| `scripts/22_prepare_continuous.py` | Sinh hai đoạn DRL theo cấu hình hiện tại |
| `scripts/23_check_reference.py` | Chẩn đoán identity nền/ảnh từ dữ liệu lưu, không kết nối hardware |
| `phoxi_yolo3d/camera.py` | Wrapper PhoXi, trigger, Frame3D, crop đồng bộ texture/point cloud |
| `pipeline.py`, `detector.py` | Texture → YOLO mask → đo/lọc → chọn proposal |
| `postprocess.py`, `surface_reference.py`, `roi.py` | ROI, nền, mặt phẳng, loại nền/vật không phù hợp |
| `picking.py`, `target_geometry.py`, `geometry.py` | Xếp hạng, chọn bề mặt, pháp tuyến, trục kẹp, proposal camera |
| `clearance.py` | Kiểm tra hình học tiếp cận trong phần scene quan sát được |
| `robot_target.py` | Proposal camera → TCP GRASP/PREGRASP trong BASE |
| `calibration/` | Thu mẫu, khôi phục pose bảng 3D, solver SE(3), lưu phiên |
| `continuous/gui.py` | Start/Stop, Camera, xuất/copy DRL, ảnh, bộ đếm, log |
| `continuous/vision.py` | Nạp pipeline, mở camera đúng policy, chụp mới, đọc thông số thật, lưu scan |
| `continuous/reference.py` | Chụp nền trống có kiểm tra và backup; lưu lựa chọn camera |
| `continuous/contract.py` | Kiểm tra cấu hình, tạo hợp đồng/hash, xác minh job và pose |
| `continuous/protocol.py` | PC server, session/sequence/job ID, PREPARE/ARM/GO và cấp phép từng bước |
| `continuous/controller.py` | Sinh phần header DRL và ghép thân mã |
| `continuous/task_writer_body.drl` | Hàm robot: socket, kiểm tra TCP/pose, Modbus, MoveL, chu trình |
| `generated/task_writer_continuous_init.drl` | Toàn bộ khối mã trước MainSub, được sinh từ config |
| `generated/task_writer_continuous_cycle.drl` | Khối trong MainSub: `phoxi_run_once()` |
| `results/continuous/run_*/` | Contract/config snapshot, events, từng scan/result/dashboard |
| `tests/test_continuous*.py`, `test_reference_identity_fix.py` | Kiểm thử addon, socket và nền; các test khác kiểm tra core |

Các tài liệu theo giai đoạn: `ROBOT_PREVIEW_UPDATE.md`, `PREGRASP_TEST.md`, `PICK_CYCLE_TEST.md`, `CONTINUOUS_PICKING.md`. Không suy ra cấu hình thực từ giá trị mặc định của ZIP.

## 7. Luồng tính pose

1. Trigger scanner khi robot đang tại P_WAIT, cảnh đứng yên. Đọc texture và point cloud đồng bộ trong CameraSpace/PrimaryCamera; crop phải đồng bộ tất cả kênh.
2. Chuẩn hóa texture/CLAHE **nếu config bật**, YOLO tạo instance mask. Các lớp từng dùng: lọ nắp xanh, lọ nắp đỏ, khối trụ, khối lập phương, hộp Anibox. Đọc `classes.txt` thực tế để biết đúng tên/thứ tự.
3. Lọc theo ROI, dữ liệu 3D hợp lệ, nền/mặt phẳng, mask trùng; xếp hạng, dựng các tư thế kẹp và kiểm tra khoảng hở. Vật cần khe mở lớn hơn 75 mm bị loại.
4. Proposal gồm `surface_anchor_xyz_mm`, `rotation_camera`; đây là điểm neo **trên bề mặt**, không phải ngay TCP robot. Ma trận hướng proposal có cột x=tiếp tuyến, y=hướng đóng kẹp, z=hướng tiếp cận vào vật. Hướng bám độ nghiêng bề mặt, không cố định từ trên xuống.
5. Đổi sang BASE, cộng độ sâu gắp theo trục tiếp cận, bù TCP–tâm tiếp xúc và xuất Euler ZYZ.

Ký hiệu B=BASE, C=camera, S=proposal bề mặt, T=TCP, F=mặt bích; vector cột, mm:

```text
R_BS = R_BC @ R_CS
z_approach = R_BS[:, 2]
p_surface_B = R_BC @ p_surface_C + t_BC
p_contact_B = p_surface_B + grasp_depth_mm * z_approach
R_BT = R_BS @ R_proposal_tcp
d_tcp_contact = R_FT.T @ (p_contact_F - p_tcp_F)
p_tcp_B = p_contact_B - R_BT @ d_tcp_contact
p_pregrasp_B = p_tcp_B - pregrasp_distance_mm * z_approach
```

Không dùng trực tiếp góc kẹp 2D làm ABC, không bù hằng Z BASE thay cho vector bù quay theo TCP, không so góc Euler từng phần tử để kết luận sai hướng khi B gần ±180°.

## 8. TCP, tâm tiếp xúc và điểm dạy

TCP đã đọc trên pendant: `[0.809, 0.369, 185.271, 0, 0, 0]` so với mặt bích. Tâm tiếp xúc được ước lượng từ số đo flange tới vùng tiếp xúc **170 mm tại mở 75 mm**; dùng `[0.809, 0.369, 170]` trong config. Suy ra vector TCP → tâm tiếp xúc khoảng `[0, 0, -15.271]` trong trục TCP. Đây vẫn là số đo gần đúng, chưa có mô hình động học đầu ngón theo mọi độ đóng.

Vùng chạm vật là các thanh đen ở đầu ngón kẹp. Gắp vào bên hông gần mặt nắp/mặt trên, độ sâu mặc định 10 mm và phải điều chỉnh được. Pregrasp cách grasp 50 mm theo hướng tiếp cận.

Điểm cố định từng xác nhận (BASE, XYZABC, mm/độ, TCP `2F 85`, solution space 2):

| Điểm | XYZABC |
|---|---|
| P_WAIT | `[-405.390, -40.539, 200.260, 27.569, -179.000, 120.349]` |
| P_DROP_UP | `[-530.260, -264.910, 214.560, 34.620, -176.630, 106.240]` |
| P_DROP | `[-530.156, -264.862, 108.119, 34.622, -176.632, 106.242]` |

Các góc khớp tương ứng được lưu trong `doosan_taught_points.json`. Dùng tệp thực, không nhập lại từ bản làm tròn này.

**TCP Z chuyển vùng = 170 mm trong BASE**, do người dùng thấy pregrasp/start cũ vẫn thấp và có thể va vành thùng khi đi ngang. Phải nâng đến cao độ chuyển vùng trước khi chuyển ngang. Cao độ này được thử trong bố trí hiện tại, không phải chứng minh tránh mọi chướng ngại khi bố trí thay đổi.

## 9. Luồng liên tục và giao thức

GUI mới mở chưa kết nối camera/robot. Chỉ Start mới mở server `192.168.137.50:20005`; robot kết nối và READY tại P_WAIT rồi PC mới nạp/warm-up pipeline, mở scanner và chụp.

HELLO chứa hash cấu hình và TCP, WELCOME tạo session. Mỗi lượt có sequence và job ID riêng. Chuỗi: `READY → WAIT → chụp mới → PREPARE → ARM → GO`. Sau GO, mỗi bước robot gửi CHECK, PC gửi PERMIT, robot thực hiện và gửi REACHED. Không nhận permit đúng thì không tiếp tục bước mới.

Chu trình:

1. Mở kẹp về PR=30 tại P_WAIT.
2. MoveL tới trên mục tiêu ở TCP BASE Z=170, hướng lấy từ mục tiêu.
3. MoveL tới PREGRASP rồi GRASP theo hướng tiếp cận.
4. Đóng kẹp, xác nhận giữ vật.
5. Rút theo hướng tiếp cận về PREGRASP.
6. Nâng BASE Z tới 170, giữ XY/ABC của PREGRASP.
7. Chuyển về P_WAIT cùng vật, tới P_DROP_UP, hạ P_DROP.
8. Mở kẹp thả, rút lên P_DROP_UP, về P_WAIT.
9. DONE → PC mới tăng bộ đếm và ACK. Lượt tiếp theo phải có READY mới và ảnh mới.

Không phát lại GO/job khi lỗi, không tự nhả vật hoặc tự Resume. Stop trên GUI là ngừng cấp lượt tiếp theo, chờ hoàn tất lượt hiện tại; Task Stop trên pendant dùng để dừng robot. Đứt mạng không đảm bảo hủy ngay MoveL đã bắt đầu.

Các kiểm tra còn giữ: tên/offset TCP, solution space 2, vị trí và joint configuration tại điểm dạy, hash calibration/config/reference, tuổi ảnh, frame ID mới, capture sau READY, pose/đơn vị/ZYZ, hướng hạ gắp, giới hạn độ nghiêng, trạng thái kẹp và thứ tự bước. `robot_ready: false` trong proposal/preview vẫn được giữ; không đổi thành true để vượt qua kiểm tra.

## 10. Vận hành trên Ubuntu và pendant

Mở GUI liên tục:

```bash
cd ~/phoxi-single-target
source .venv/bin/activate
python scripts/21_run_continuous.py --config config/picking_robot.yaml
```

Trong Task Writer có hai Custom Code dạng Script:

- **Trước MainSub**: toàn bộ mã khởi tạo/định nghĩa hàm, lấy bằng nút Sao chép mã trước MainSub.
- **Trong MainSub**: chỉ `phoxi_run_once()`; đặt số vòng lặp MainSub bằng số lượt tối đa trên GUI khi dùng N dương.

Đứng tại P_WAIT, kẹp rỗng, bắt đầu GUI/server trước rồi chạy Task Writer Real Mode từ đầu. GUI 0 nghĩa là phía PC không đặt giới hạn lượt; số lần gọi hàm phía MainSub vẫn quyết định. Không tự coi 0 là vô hạn nếu MainSub không lặp.

Sau đổi TCP, pose/điểm dạy, calibration, tốc độ, Modbus/PR/force hoặc tuổi ảnh: **xuất/sao chép lại toàn bộ mã trước MainSub và Confirm**, không sửa rời header/hash. Đổi số lượt hoặc policy chiếu sáng không đổi hợp đồng robot trong mã đã đọc.

Sinh DRL bằng terminal nếu cần:

```bash
python scripts/22_prepare_continuous.py --config config/picking_robot.yaml
```

Live cũ dùng:

```bash
python scripts/07_run_live.py --config config/picking_robot.yaml
```

Run_live là nhánh chụp/preview, S lưu kết quả. Nhánh liên tục không cần bấm S/server riêng cho từng lượt. Không mở hai chương trình cùng giữ camera.

## 11. Camera, nền và lỗi đã xử lý

Profile người dùng đã chỉnh để nhận dạng tốt: **Texture Source=LED, Shutter Multiplier=1, Laser Power=2048, LED Power=2048**. GUI đã có ảnh đọc lại xác nhận đúng bốn giá trị này. Không tự đổi xuống LED 1048 để ép khớp nền cũ.

Nguồn `config` áp dụng bốn giá trị và đọc lại; `phoxicontrol` giữ thông số hiện có. Bản sửa đọc TextureSource/LEDPower thật ở mỗi frame để ghi identity đúng. Trong run_live cũ, nhãn thông số có thể vẫn lấy từ YAML lúc mở, dù người dùng đã chỉnh thật trong PhoXi Control; đây là khả năng gây nền sai metadata, không phải kết luận đã biết tất cả trường lệch trên máy.

GUI từng báo `REFERENCE_IDENTITY_UNVERIFIED`, dù 3D tốt và run_live chọn được vật. Bản sửa đã thêm log từng trường identity nền/ảnh, nút Lưu lựa chọn camera và Chụp lại nền — thùng đã trống. Khi chụp nền: dọn hết vật, robot tại P_WAIT, không chạy Task Writer, giữ camera/thùng/ROI. Công cụ chụp nhiều frame (mặc định 15), kiểm tra mặt phẳng/độ phủ, backup NPZ cũ rồi thay. Người dùng sau đó xác nhận đã nhận dạng được.

Không tắt `require_reference_identity` hoặc sửa nhãn NPZ cũ để bỏ qua lỗi. Chẩn đoán chỉ đọc:

```bash
python scripts/23_check_reference.py --config config/picking_robot.yaml
```

PhoXi từng báo không xác minh PrimaryCamera dù readback trả đúng CameraSpace/PrimaryCamera; bộ chọn chỉ đọc có thể báo không cho ghi. Đọc lại hai giá trị thực trước khi kết luận thất bại.

## 12. Tuổi ảnh, socket và tốc độ — tránh hiểu sai

Tuổi ảnh gốc 5 giây. GUI từng dựng preview ở khoảng 4,1/5 giây; sau truyền PREPARE và nhận ARM ảnh vượt ngưỡng, PC báo `CAPTURE_EXPIRED_BEFORE_GO`, đóng socket, pendant báo thêm `CONT_SOCKET_READ_FAILED`.

Bản sửa socket có `controller_revision=2`: pendant đọc theo khối `length=-1` và giữ bộ đệm TCP, thay vì một SDK call mỗi byte. PC ghi tuổi ảnh, thời gian PREPARE–ARM; kiểm tra hạn ngay trước GO sau ghi log; nếu hết hạn gửi END rõ lý do và không gửi GO. Khi cài bản này phải thay mã trước MainSub.

Người dùng yêu cầu tăng hạn, đã hướng dẫn `picking.max_frame_age_ms: 20000` trong **picking_robot.yaml**. Ảnh sau đó đã hiển thị 20 giây. Cảnh phải đứng yên từ lúc chụp đến bắt đầu gắp; tăng hạn không biến ảnh cũ thành ảnh cập nhật.

Tốc độ cấu hình ở **continuous.yaml → continuous.motion**, không phải sửa câu chữ popup. Mảng velocity `[linear_mm_s, angular_deg_s]` là tốc độ TCP/đổi hướng, không phải tốc độ ban đầu từng khớp. Gia tốc cũng có phần tịnh tiến và quay. Cấu hình ban đầu travel `[200,5]`, acc `[200,10]`; approach `[30,5]`, acc `[60,10]`.

Người dùng muốn mức gần 15% của mốc TCP 1 m/s. Đã hướng dẫn travel velocity `[150,20]`, travel acceleration `[300,40]`, giữ đoạn approach như đang dùng. Đây chỉ là tốc độ đặt, không bảo đảm tương đương chính xác cảm giác Jog Speed 15% ở mọi tư thế. Thanh Speed % pendant nhân giảm thêm tốc độ đã lập trình; ví dụ 150 mm/s với hệ số 15% chỉ còn mức danh nghĩa 22,5 mm/s trước các giới hạn khác. Chưa biết Speed % cuối cùng trên pendant; không tự kết luận đang 100%.

Không tăng vận tốc lên 4000 để chữa chuyển động chậm. Cần xem cả giới hạn quay, gia tốc, phần trăm Speed, đoạn travel/approach và việc đã dán lại DRL mới hay chưa.

## 13. Tay kẹp — vấn đề đang giải quyết

Các tên Modbus do người dùng cấu hình:

| Tín hiệu | Công dụng/giải mã trong mã hiện tại |
|---|---|
| `2F_PR_1` | Position request, 0–255 |
| `2F_SP_FO_1` | Word speed/force = `speed_raw * 256 + force_raw` |
| `2F_GS_1` | High byte: status, act/gto/sta/obj |
| `2F_FS_PRE_1` | High byte fault; low byte position request echo |
| `2F_PO_CU_1` | High byte actual finger position; low byte motor current raw |

Mở 75 mm đã đo tương ứng PR=30. Lệnh đóng đã thử trước đây là PR=85, word SP/FO=70. **85 là vị trí đóng, không phải lực, không phải mặc định đóng hết.** Word 70 là speed_raw=0 (tốc độ nhỏ nhất, không phải đứng yên), force_raw=70. Người dùng muốn giảm force_raw xuống 30; word tương ứng là 30 nếu giữ speed_raw=0. Nếu muốn cả speed_raw=30 và force_raw=30 thì word phải là 7710; đó không phải lựa chọn mới nhất đang được hướng dẫn. Raw force 30/70 không phải Newton và không đổi sang N bằng một phép tỉ lệ tùy ý.

Điều kiện giữ vật trong mã đã đọc:

```text
fault=0, act=1, sta=3, gto=1
request_echo == close_pr
gOBJ == 2
actual_position > open_pr + 1
```

Ba điều kiện cuối phải đạt ba lần đọc liên tiếp khi đóng (mỗi vòng chờ poll_seconds, mặc định 0,05 giây). Sau đó `PX_HOLDING=True`; tiếp tục kiểm tra giữ vật trước/sau các bước mang vật. Không có phép so lực thực với 70 hoặc 30 để quyết định nâng.

Lỗi `CONT_NO_GRIP_CONTACT_CHECK_CLOSE_PR_AND_OBJECT_WIDTH` xuất hiện khi echo đúng đích đóng, `gOBJ=3`, actual position gần đích trong ba lần đọc. `gOBJ=2` là dừng do tiếp xúc khi đang đóng; `gOBJ=3` là đạt vị trí yêu cầu, không xác nhận tiếp xúc. Người dùng đã Jog Z và nâng được vật sau lỗi, nên không được khẳng định chắc chắn vật đang rỗng/chưa được giữ. Có thể tiếp xúc nằm ngay gần PR=85; trạng thái end-position chưa thỏa tiêu chí software.

**Cách sửa mới nhất đã hướng dẫn (chưa có xác nhận chạy thực thành công):**

```yaml
continuous:
  gripper:
    open_pr: 30
    close_pr: 255
    speed_force_word: 30
```

Mục đích là yêu cầu đóng qua vị trí chạm vật, kẹp dừng theo cơ chế giới hạn lực/dòng nội bộ và báo tiếp xúc, thay vì chủ động dừng tại 85. **PR255 là đích vị trí, force vẫn 30.** Giữ xác nhận gOBJ=2; chưa viết fallback chấp nhận mọi gOBJ=3.

Mô phỏng đã kiểm tra: contact ở raw position 85 hoặc 120 khi yêu cầu 255 hoàn tất chu trình; đóng hụt tới 255 vẫn bị chặn nâng. Đây không xác nhận lực 30 đủ giữ mọi vật hoặc hình học ngón kẹp đúng với mọi độ đóng.

Bước tiếp theo: kiểm tra file và DRL đang dùng có thật sự close_pr=255/word30; thử một lượt với vật từng gây lỗi, ghi raw words/gOBJ/echo/actual position/current. Kiểm tra cả vật có kích thước khác nhau và trường hợp đóng hụt. Nếu vẫn giữ được vật mà gOBJ=3, cần dữ liệu phản hồi để thiết kế thêm xác nhận; **không sửa đơn giản `obj == 2` thành `obj in (2,3)`**, vì đóng rỗng cũng có thể cho obj=3.

## 14. Quy tắc làm việc và những gì chưa được xác minh

- Người dùng muốn hướng dẫn tiếng Việt rõ, lệnh terminal sao chép chạy được; không hỏi lại các thông tin đã có trong gói.
- Giữ cấu hình cục bộ khi giải nén addon; không ghi đè picking_robot.yaml, continuous.yaml, calibration hoặc điểm dạy bằng mặc định cũ. Có backup trước thay YAML/nền.
- Các addon từng là tệp untracked trong Git; `main` khớp `origin/main` không chứng minh toàn bộ bản chạy đã được commit. Không reset/checkout/stash lại theo suy đoán và làm mất phần đang chạy.
- Sau thay cấu hình thuộc hợp đồng robot phải xuất lại mã trước MainSub; không chỉ đổi PC rồi để pendant giữ hash cũ.
- Không thay kiến trúc sang ROS2/MoveIt nếu người dùng chưa yêu cầu.
- Chưa có đo lực N, mô hình bù tâm tiếp xúc theo toàn bộ độ đóng, tránh va chạm toàn tay, phát hiện giữ đúng instance độc lập hay nghiệm thu lâu dài nhiều chu kỳ.
- Chưa xác nhận tự hồi phục lỗi/đứt mạng, tự retry gắp hoặc tự bỏ qua vật lỗi. Hiện dừng và không replay job.
- Nếu cần sửa lỗi mới, ưu tiên log phiên thực và cấu hình hiện tại; phân biệt lỗi camera/reference, quá hạn ảnh, socket, geometry/TCP và Modbus/giữ vật.

## 15. Cách duy trì hồ sơ cho lần chuyển chat tiếp theo

Sau mỗi mốc thay đổi, bổ sung vào cuối tài liệu: ngày, mục tiêu vừa làm, tệp thay đổi, cấu hình thực, lệnh chạy, đã thử trên robot hay chỉ mô phỏng, kết quả và bước tiếp theo. Không xóa lịch sử lỗi chỉ vì đã có bản vá.

Trước mở chat mới: dừng chu trình, chạy `scripts/export_chat_context.py`, gửi ZIP xuất ra và prompt trong `START_NEW_CHAT.md`. Gói xuất mới có mã/cấu hình/log thật trên máy, đáng tin hơn những giá trị được chép trong tài liệu lịch sử này.
