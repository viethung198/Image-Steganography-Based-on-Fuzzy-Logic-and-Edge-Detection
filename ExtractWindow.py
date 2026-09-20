import cv2
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from UI.FormExtract_ui import Ui_ExtractForm
from core.stego_embed import extract_message
from PyQt6.QtGui import QPixmap, QTextCursor



class ExtractWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_ExtractForm()
        self.ui.setupUi(self)

        # ===== Trạng thái =====
        self.selected_algorithm = None  # Optional - để verify
        self.stego_image_path = None

        # ===== Kết nối nút =====
        self.ui.btn_Sobel.clicked.connect(lambda: self.select_algorithm("sobel"))
        self.ui.btn_Canny.clicked.connect(lambda: self.select_algorithm("canny"))
        self.ui.btn_Hybrid.clicked.connect(lambda: self.select_algorithm("hybrid"))

        self.ui.btn_SelectStegoImage.clicked.connect(self.load_stego_image)
        self.ui.btn_Extract.clicked.connect(self.extract_message)

    def select_algorithm(self, algo: str):
        """Chọn thuật toán để verify (không bắt buộc)"""
        self.selected_algorithm = algo

        # 💥 Reset vùng hiển thị message mỗi khi chọn thuật toán khác
        self.ui.text_Message.setPlainText("")
        
        QMessageBox.information(
            self, 
            "Thuật toán", 
            f"✅ Đã chọn: {algo.upper()}\n\n"
            f"💡 Nếu không khớp với ảnh, bạn sẽ được thông báo."
        )

    def load_stego_image(self):
        """Chọn ảnh stego"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh đã nhúng (Stego Image)",
            "",
            "Images (*.png *.jpg *.bmp *.jpeg)",
        )
        if file_path:
            self.stego_image_path = file_path
            self.display_image(file_path, self.ui.label_Input)
            self.ui.text_Message.setPlainText("Chờ trích xuất...")
            QMessageBox.information(
                self, 
                "✅ Đã chọn ảnh", 
                f"Ảnh stego:\n{file_path}\n\n"
                f"Nhấn 'Trích xuất' để lấy message."
            )

    def display_image(self, file_path, label):
        """Hiển thị ảnh lên label"""
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

    def extract_message(self):
        """Trích xuất message - TỰ ĐỘNG phát hiện algorithm từ header"""
        
        # 1. Kiểm tra ảnh
        if not self.stego_image_path:
            QMessageBox.warning(
                self, 
                "⚠️ Thiếu ảnh", 
                "Vui lòng chọn ảnh stego trước!"
            )
            return

        try:
            # 2. Đọc ảnh
            stego_img = cv2.imread(self.stego_image_path)
            if stego_img is None:
                QMessageBox.critical(self, "Lỗi", "Không thể đọc ảnh stego!")
                return

            # ✅ 3. GỌI HÀM TRÍCH XUẤT - Tự động detect algorithm
            # Truyền expected_algorithm nếu user đã chọn (để verify)
            message, detected_algo = extract_message(
                stego_img_path=stego_img,
                expected_algorithm=self.selected_algorithm  # None = auto-detect
            )

            # 4. Hiển thị kết quả
            if message:
                self.ui.text_Message.setPlainText(message)
                self.ui.text_Message.moveCursor(QTextCursor.MoveOperation.Start)

                # Thông báo thành công với thông tin algorithm
                algo_info = (
                    f"🔍 Algorithm phát hiện: {detected_algo.upper()}\n"
                )
                if self.selected_algorithm:
                    if self.selected_algorithm.lower() == detected_algo.lower():
                        algo_info += "✅ Khớp với lựa chọn của bạn!"
                    else:
                        algo_info += f"⚠️ Bạn chọn {self.selected_algorithm.upper()} nhưng ảnh dùng {detected_algo.upper()}"

                QMessageBox.information(
                    self,
                    "✅ Trích xuất thành công",
                    f"{algo_info}\n\n"
                    f"📝 Độ dài message: {len(message)} ký tự\n"
                    f"Message đã hiển thị trong ô bên phải."
                )
            else:
                self.ui.text_Message.setPlainText("(Không có message)")
                QMessageBox.warning(
                    self,
                    "⚠️ Message rỗng",
                    "Không tìm thấy message trong ảnh."
                )

        except ValueError as ve:
            # Lỗi từ extract_message (sai algorithm, header corrupt, v.v.)
            error_msg = str(ve)
            
            # Nếu lỗi về algorithm không khớp
            if "SAI THUẬT TOÁN" in error_msg:
                QMessageBox.critical(
                    self,
                    "❌ Sai thuật toán",
                    "Thuật toán bạn chọn KHÔNG trùng với thuật toán nhúng.\n"
                    "Vui lòng chọn đúng thuật toán nhúng."
    )

            else:
                QMessageBox.critical(
                    self,
                    "❌ Lỗi trích xuất",
                    f"{error_msg}\n\n"
                    f"Có thể do:\n"
                    f"• Ảnh không phải stego image\n"
                    f"• Ảnh bị chỉnh sửa/nén sau khi nhúng\n"
                    f"• Header bị hỏng"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "❌ Lỗi không xác định",
                f"Lỗi: {str(e)}"
            )