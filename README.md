# 🛡️ Image Steganography Based on Fuzzy Logic and Edge Detection

Đồ án giấu tin mật (Data Hiding/Steganography) trên ảnh số kết hợp **Logic mờ (Fuzzy Logic)** tối ưu dung lượng/vùng nhúng, **Phát hiện cạnh (Edge Detection)** giảm lộ thị giác, tích hợp **YOLOv8** định vị vùng quan tâm, giao diện **PyQt6** và xác thực qua **MySQL**.

---

## 🏗️ Kiến trúc hệ thống & Workflow
* **Authentication**: Đăng nhập/Đăng ký tài khoản kết nối Database MySQL (`UI/database.py`).
* **Object Detection & Edge Analysis**: Dùng `YOLOv8` (`yolov8s.pt`) định vị vật thể và `core/edge_detection.py` phân tích biên kết hợp **Fuzzy Logic** quyết định mật độ pixel nhúng.
* **Embedding / Extraction**: Nhúng payload qua `core/stego_embed.py` và trích xuất qua `ExtractWindow.py`.
* **Quality Evaluation**: Đánh giá PSNR, MSE qua `core/core_metric.py`.

---

## ⚙️ Yêu cầu hệ thống
* **OS**: Windows 10 / 11
* **Python**: Version 3.12
* **Database**: MySQL Server / XAMPP (Port `3306`)

---

## 🚀 Hướng dẫn Clone & Chạy ngay lập tức

### 📌 Bước 1: Clone Repository
```bash
git clone [https://github.com/viethung198/Image-Steganography-Based-on-Fuzzy-Logic-and-Edge-Detection.git](https://github.com/viethung198/Image-Steganography-Based-on-Fuzzy-Logic-and-Edge-Detection.git)
cd Image-Steganography-Based-on-Fuzzy-Logic-and-Edge-Detection
```
📌 Bước 2: Cài đặt thư viện Python
```Bash
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```
📌 Bước 3: Cấu hình Database MySQL
Bật MySQL Server / XAMPP Control Panel (Port 3306).

Cấu hình kết nối tương ứng trong UI/database.py.

Khởi tạo bảng người dùng (users) phục vụ Login/Register.

📌 Bước 4: Chạy ứng dụng
```Bash
py -u app_controller.py
```

📁 Cấu trúc thư mục dự án
Plaintext
├── UI/                     # Giao diện PyQt6 (.ui, .py) & quản lý DB (database.py)
├── core/                   # Lõi thuật toán: edge_detection, stego_embed, core_metric
├── assets/                 # Tài nguyên hình ảnh / icon giao diện
├── app_controller.py       # Controller trung tâm điều phối luồng GUI
├── main.py / main_edge...  # Entry points kiểm thử module lẻ
├── yolov8s.pt              # Trọng số mô hình YOLOv8
└── requirements.txt        # Danh sách thư viện phụ thuộc
