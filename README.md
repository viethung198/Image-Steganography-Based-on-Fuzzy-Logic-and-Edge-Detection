# 🛡️ Image Steganography Based on Fuzzy Logic and Edge Detection

Giấu tin mật (Data Hiding/Steganography) trên ảnh số kết hợp **Logic mờ (Fuzzy Logic)** để tối ưu dung lượng/vùng nhúng và **Phát hiện cạnh (Edge Detection)** để giấu tin vào vùng biên ít lộ thay đổi thị giác, tích hợp **YOLOv8** định vị vùng quan tâm, giao diện **PyQt6** và xác thực người dùng qua **MySQL**.

---

## 🏗️ Kiến trúc hệ thống & Workflow
1. **Authentication:** Đăng nhập/Đăng ký tài khoản kết nối Database MySQL (`UI/database.py`).
2. **Object Detection / Edge Analysis:** 
   - Dùng `YOLOv8` (`yolov8s.pt`) nhận diện vật thể hoặc phân tích biên ảnh qua `core/edge_detection.py` kết hợp **Fuzzy Logic** quyết định mật độ/vùng nhúng pixel tối ưu.
3. **Embedding / Extraction:** Nhúng payload vào không gian ảnh qua `core/stego_embed.py` và trích xuất qua `ExtractWindow.py`.
4. **Quality Evaluation:** Đánh giá độ suy giảm chất lượng ảnh (PSNR, MSE...) qua `core/core_metric.py`.

---

## ⚙️ Yêu cầu hệ thống
- **OS:** Windows 10 / 11
- **Python:** Version 3.12
- **Database:** MySQL Server / XAMPP (Port `3306`)

---

## 🚀 Hướng dẫn Clone & Chạy ngay lập tức

### Bước 1: Clone Repository
```bash
git clone [https://github.com/viethung198/Image-Steganography-Based-on-Fuzzy-Logic-and-Edge-Detection.git](https://github.com/viethung198/Image-Steganography-Based-on-Fuzzy-Logic-and-Edge-Detection.git)
cd Image-Steganography-Based-on-Fuzzy-Logic-and-Edge-Detection
Bước 2: Cài đặt thư viện Python
Bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
Bước 3: Cấu hình Database MySQL
Bật MySQL Server (XAMPP Control Panel hoặc MySQL service, đảm bảo port 3306).

Tạo database phù hợp với cấu hình trong file UI/database.py (Mặc định host='localhost', port=3306, user='root', password rỗng hoặc cấu hình tương ứng code của bạn).

Đảm bảo bảng người dùng (users hoặc tương đương) đã được khởi tạo để chức năng Login/Register ở app_controller.py hoạt động.

Bước 4: Chạy ứng dụng
Bash
py -u app_controller.py
📁 Cấu trúc thư mục dự án
Plaintext
├── UI/                     # Giao diện PyQt6 (.ui, .py) & quản lý DB (database.py)
├── core/                   # Lõi thuật toán: edge_detection, stego_embed, core_metric
├── assets/                 # Tài nguyên hình ảnh / icon giao diện
├── app_controller.py       # Controller trung tâm điều phối luồng GUI
├── main.py / main_edge...  # Entry points kiểm thử module lẻ
├── yolov8s.pt              # Trọng số mô hình YOLOv8
└── requirements.txt        # Danh sách thư viện phụ thuộc
