import cv2
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from UI.FormExtract_ui import Ui_ExtractForm

# Sử dụng lại các hàm phát hiện cạnh giống EmbedWindow
from core.edge_detection import (
    sobel_edge_optimized,
    canny_with_blur,
    hybrid_fuzzy_or_canny,
)

# ✅ THAY ĐỔI: Dùng hàm wrapper extract_message() thay vì gọi trực tiếp
from core.stego_embed import extract_message


class ExtractWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ExtractForm()
        self.ui.setupUi(self)

        # ===== Trạng thái =====
        self.selected_algorithm = None      # "sobel" / "canny" / "hybrid"
        self.stego_image_path = None
        self.x_bits = 3  # bits ở vùng edge (phải khớp lúc nhúng)
        self.y_bits = 1  # bits ở vùng non-edge (phải khớp lúc nhúng)

        # ===== Kết nối nút =====
        self.ui.btn_Sobel.clicked.connect(lambda: self.select_algorithm("sobel"))
        self.ui.btn_Canny.clicked.connect(lambda: self.select_algorithm("canny"))
        self.ui.btn_Hybrid.clicked.connect(lambda: self.select_algorithm("hybrid"))

        self.ui.btn_SelectStegoImage.clicked.connect(self.load_stego_image)
        self.ui.btn_Extract.clicked.connect(self.extract_message)

    # --- Chọn thuật toán ---
    def select_algorithm(self, algo: str):
        self.selected_algorithm = algo  # lưu dạng "sobel" / "canny" / "hybrid"
        QMessageBox.information(self, "Thuật toán", f"Đã chọn thuật toán: {algo.upper()}")

    # --- Chọn ảnh stego ---
    def load_stego_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh đã nhúng",
            "",
            "Images (*.png *.jpg *.bmp *.jpeg)",
        )
        if file_path:
            self.stego_image_path = file_path
            self.display_image(file_path, self.ui.label_Input)
            self.ui.label_Edge.setText("Thông điệp trích xuất")  # reset ô bên phải
            QMessageBox.information(self, "Ảnh nhúng", f"Đã chọn ảnh nhúng:\n{file_path}")

    # --- Hiển thị ảnh màu ---
    def display_image(self, file_path, label):
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.critical(self, "Lỗi", f"Không thể đọc ảnh:\n{file_path}")
            return

        label.setPixmap(
            pixmap.scaled(
                label.width(),
                label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # --- Trích xuất thông điệp và hiển thị trên label ---
    def extract_message(self):
        # 1. Kiểm tra dữ liệu đầu vào
        if not self.stego_image_path:
            QMessageBox.warning(self, "Thiếu dữ liệu", "Vui lòng chọn ảnh nhúng trước.")
            return

        if not self.selected_algorithm:
            QMessageBox.warning(
                self,
                "Thiếu thuật toán",
                "Vui lòng chọn thuật toán phát hiện cạnh (Sobel/Canny/Hybrid).",
            )
            return

        try:
            # 2. Đọc ảnh stego
            stego_img = cv2.imread(self.stego_image_path)
            if stego_img is None:
                QMessageBox.critical(self, "Lỗi", "Không thể đọc ảnh stego.")
                return

            gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)

            # 3. Tự tạo edge map từ ảnh nhúng dựa trên thuật toán đã chọn
            algo = self.selected_algorithm.lower()

            if algo == "sobel":
                edge_map = sobel_edge_optimized(gray)
            elif algo == "canny":
                edge_map = canny_with_blur(gray)
            elif algo == "hybrid":
                _, _, edge_map = hybrid_fuzzy_or_canny(gray)
            else:
                QMessageBox.warning(self, "Thuật toán", "Thuật toán không hợp lệ!")
                return

            # ✅ 4. GỌI HÀM WRAPPER - Tự động xử lý header + extract
            # Hàm này trả về STRING đã decode, không phải bits
            message = extract_message(
                stego_img,      # numpy array hoặc path
                edge_map,       # numpy array hoặc path
                x=self.x_bits,
                y=self.y_bits,
            )

            # 5. Hiển thị kết quả
            if message:
                self.ui.label_Edge.setWordWrap(True)
                self.ui.label_Edge.setText(message)

                QMessageBox.information(
                    self,
                    "Thành công",
                    f"Đã trích xuất thông điệp thành công!\n\n"
                    f"Độ dài: {len(message)} ký tự\n"
                    f"Nội dung đã hiển thị trong ô 'Thông điệp trích xuất'.",
                )
            else:
                self.ui.label_Edge.setText("")
                QMessageBox.warning(
                    self,
                    "Trích xuất rỗng",
                    "Không tìm thấy thông điệp nào trong ảnh (message rỗng).",
                )

        except ValueError as ve:
            # Lỗi có thể do header sai, edge map không khớp, x/y sai,...
            QMessageBox.critical(
                self,
                "Lỗi định dạng / tham số",
                f"Lỗi khi trích xuất:\n{str(ve)}\n\n"
                f"Đảm bảo:\n"
                f"- Ảnh này đã được nhúng tin nhắn bằng chương trình của bạn\n"
                f"- Thuật toán phát hiện cạnh ({algo.upper()}) trùng với lúc nhúng\n"
                f"- Ảnh không bị nén/chỉnh sửa sau khi nhúng (dùng PNG)",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Lỗi không xác định",
                f"Có lỗi xảy ra khi trích xuất:\n{str(e)}",
            )