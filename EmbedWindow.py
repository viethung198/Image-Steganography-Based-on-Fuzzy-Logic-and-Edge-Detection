import cv2
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from UI.AnMa_ui import Ui_AnMaForm
from core.edge_detection import sobel_edge_optimized, canny_with_blur, hybrid_fuzzy_or_canny
from core.stego_embed import embed_message
import os
import shutil
from PyQt6.QtCore import Qt


class EmbedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_AnMaForm()
        self.ui.setupUi(self)

        # ===== State =====
        self.selected_method = None
        self.image_path = None
        self.edge_map = None
        self.stego_img_path = None

        # ===== Event connections =====
        self.ui.btn_Sobel.clicked.connect(lambda: self.select_method("sobel"))
        self.ui.btn_Canny.clicked.connect(lambda: self.select_method("canny"))
        self.ui.btn_Hybrid.clicked.connect(lambda: self.select_method("hybrid"))

        self.ui.btn_Select.clicked.connect(self.open_image)
        self.ui.btn_Embed.clicked.connect(self.embed_file)
        self.ui.btn_SaveImage.clicked.connect(self.save_stego_image)

    # ------------------------------
    def select_method(self, method):
        self.selected_method = method
        QMessageBox.information(
            self,
            "Thuật toán",
            f"✅ Đã chọn: {method.upper()}\n\n"
            "💡 Thuật toán sẽ được lưu vào HEADER của stego image."
        )

    # ------------------------------
    def open_image(self):
        img_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh gốc",
            "",
            "Image Files (*.png *.jpg *.bmp *.jpeg)"
        )
        if not img_path:
            return

        message = self.ui.txt_SecretMessage.text().strip()

        if not message:
            QMessageBox.warning(self, "⚠️ Thiếu message", "Vui lòng nhập thông điệp cần nhúng!")
            return

        if not self.selected_method:
            QMessageBox.warning(self, "⚠️ Thiếu thuật toán", "Bạn phải chọn thuật toán phát hiện cạnh!")
            return

        self.image_path = img_path
        self.display_image(img_path, self.ui.label_Input)

        # Detect edges trước
        self.detect_edges()

        # Kiểm tra lại edge_map
        if self.edge_map is None:
            QMessageBox.critical(self, "❌ Lỗi", "Không tạo được edge map!")
            return


    # ------------------------------
    def display_image(self, path, label):
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Lỗi", f"Không thể đọc ảnh: {path}")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)

        label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                label.width(),
                label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    # ------------------------------
    def detect_edges(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        try:
            if self.selected_method == "sobel":
                self.edge_map = sobel_edge_optimized(img)
            elif self.selected_method == "canny":
                self.edge_map = canny_with_blur(img)
            elif self.selected_method == "hybrid":
                _, _, self.edge_map = hybrid_fuzzy_or_canny(img)

            print(f"✅ Đã phát hiện cạnh bằng {self.selected_method.upper()}")

        except Exception as e:
            QMessageBox.critical(self, "❌ Lỗi phát hiện cạnh", str(e))

    # ------------------------------
    def embed_file(self):
        try:
            message = self.ui.txt_SecretMessage.text().strip()
            if not message:
                QMessageBox.warning(self, "⚠️ Lỗi", "Chưa nhập message!")
                return

            output_img = "stego_temp.png"

            # ⭐ NHÚNG THEO CORE MỚI — TRẢ VỀ OBJECT METRICS
            metrics = embed_message(
                cover_img_path=self.image_path,
                edge_map=self.edge_map,
                secret_message=message,
                output_path=output_img,
                algorithm=self.selected_method
            )

            self.stego_img_path = output_img
            self.display_image(output_img, self.ui.label_Output)

            # ⭐ HIỂN THỊ THÔNG SỐ ĐẦY ĐỦ
            QMessageBox.information(
                self,
                "✅ Nhúng thành công",
                f"📊 THÔNG SỐ EMBEDDING:\n"
                f"• PSNR: {metrics.psnr:.2f} dB\n"
                f"• SSIM: {metrics.ssim:.4f}\n"
                f"• MSE: {metrics.mse:.4f}\n"
                f"• Số bits nhúng: {metrics.embedded_bits}\n"
                f"• Dung lượng nhúng tối đa: {metrics.capacity_bits} bits\n"
                f"• % dung lượng nhúng sử dụng: {metrics.percent_used:.2f}%\n"
                f"• BPP: {metrics.bpp:.4f}\n"
                f"• Edge Pixels: {metrics.edge_pixels}\n"
                f"• Non-Edge Pixels: {metrics.non_edge_pixels}\n\n"
                f"🔐 Algorithm nhúng: {self.selected_method.upper()}\n"
            )

        except ValueError as e:
            QMessageBox.warning(self, "❌ Lỗi nhúng", str(e))
        except Exception as e:
            QMessageBox.critical(self, "❌ Lỗi", str(e))

    # ------------------------------
    def save_stego_image(self):
        if not self.stego_img_path or not os.path.exists(self.stego_img_path):
            QMessageBox.warning(self, "⚠️ Chưa có ảnh", "Chưa có ảnh stego để lưu.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu ảnh Stego",
            "stego_image.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )

        if save_path:
            try:
                shutil.copy(self.stego_img_path, save_path)

                QMessageBox.information(
                    self,
                    "✅ Lưu thành công",
                    f"Ảnh stego đã lưu tại:\n{save_path}\n\n"
                    f"🔐 Thuật toán nhúng: {self.selected_method.upper()} (lưu trong header)"
                )

            except Exception as e:
                QMessageBox.critical(self, "❌ Lỗi", str(e))
