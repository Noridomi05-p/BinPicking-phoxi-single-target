# Quy tắc gán nhãn cho vật xếp chồng

Mục tiêu là tạo nhãn **instance segmentation** nhất quán. Chất lượng nhãn thường ảnh
hưởng kết quả nhiều hơn việc đổi sang model lớn hơn.

## Quy tắc bắt buộc

1. Mỗi vật thể vật lý là một polygon riêng, kể cả hai vật cùng lớp đang chạm nhau.
2. Chỉ vẽ phần bề mặt thực sự nhìn thấy. Không đoán đường bao nằm dưới vật che phía trên.
3. Đường polygon bám theo biên vật; không lấy bóng, thành hộp, nền hoặc khe hở lớn.
4. Nếu một vật bị tách thành hai vùng nhìn thấy bởi vật che, vẫn gán chúng theo một quy
   tắc duy nhất cho toàn bộ dataset. Với công cụ hiện tại, nên tạo hai polygon cùng lớp;
   phép NMS/model sẽ học chúng như hai vùng quan sát. Nếu trường hợp này xuất hiện nhiều,
   cần nâng công cụ sang mask đa polygon cho một instance ở phiên bản sau.
5. Vật giống hệt nhau nhưng là hai cá thể khác nhau phải có hai mask.
6. Vật bị che quá nhiều và không thể xác định lớp một cách chắc chắn thì không gán.
7. Ảnh không có vật thuộc các lớp vẫn lưu file `.txt` rỗng bằng phím `S`.
8. Không đổi class ID giữa chừng. ID chính là số dòng trong `config/classes.txt`, bắt đầu
   từ 0.

## Cách xử lý chồng lấp

Tại một pixel chỉ có bề mặt gần camera nhất tạo ra depth. Vì vậy mask của vật phía trên
chiếm pixel đó; mask vật phía dưới chỉ bao phần còn nhìn thấy. Hai mask chuẩn không được
cùng chiếm một pixel trên ảnh nhãn hữu hình.

## Trước khi nhấn lưu

- Kiểm tra đúng tên lớp.
- Không bỏ sót vật rõ ràng.
- Không gộp hai vật cùng loại thành một polygon.
- Không để polygon cắt qua vùng vật khác.
- Phóng to/đặt nhiều điểm hơn tại đường biên cong hoặc nơi hai vật tiếp xúc.

## Bộ pilot

Sau khoảng 50–100 ảnh đầu tiên:

1. Chạy kiểm tra nhãn.
2. Train 50 epoch ở kích thước 640.
3. Xem các dự đoán sai để phát hiện quy tắc gán nhãn chưa nhất quán.
4. Sửa nhãn trước khi thu và gán hàng trăm ảnh tiếp theo.

Không nên hoàn thành toàn bộ dataset rồi mới chạy thử lần đầu.

