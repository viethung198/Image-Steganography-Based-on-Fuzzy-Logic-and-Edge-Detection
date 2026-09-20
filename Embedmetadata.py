import cv2
import numpy as np
import json
import os
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtGui import QImage, QPixmap
from UI.AnMa_ui import Ui_AnMaForm
from core.edge_detection import sobel_edge_optimized, canny_with_blur, hybrid_fuzzy_or_canny
from core.stego_embed import embed_message, compute_metrics
import shutil
from PyQt6.QtCore import Qt


class EmbedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_AnMaForm()
        self.ui.setupUi(self)

        # ===== Biến lưu trạng thái =====
        self.selected_method = None      
        self.image_path = None           
        self.edge_map = None             
        self.stego_img_path = None
        self.edge_map_path = None
        
        # >>> ADD: thông số nhúng và edge params
        self.x_bits = 3  # bits ở vùng edge
        self.y_bits = 1  # bits ở vùng non-edge
        self.edge_params = {}  # ✅ LƯU parameters của edge detection

        # ===== Kết nối sự kiện =====
        self.ui.btn_Sobel.clicked.connect(lambda: self.select_method("sobel"))
        self.ui.btn_Canny.clicked.connect(lambda: self.select_method("canny"))
        self.ui.btn_Hybrid.clicked.connect(lambda: self.select_method("hybrid"))

        self.ui.btn_Select.clicked.connect(self.open_image)
        self.ui.btn_SaveImage.clicked.connect(self.save_stego_image)
        self.ui.btn_SaveEdge.clicked.connect(self.save_edge_map)

    # --------- Chọn thuật toán ----------
    def select_method(self, method):
        self.selected_method = method
        QMessageBox.information(self, "Thuật toán", f"Đã chọn thuật toán: {method.upper()}")

    # --------- Chọn ảnh cần ẩn tin ----------
    def open_image(self):
        img_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Image Files (*.png *.jpg *.bmp *.jpeg)"
        )
        if not img_path:
            return
        message = self.ui.txt_SecretMessage.text().strip()
        
        # Kiểm tra message
        if not message:
            QMessageBox.warning(self, "Thiếu thông điệp", "Vui lòng nhập thông điệp cần nhúng!")
            return

        if not self.selected_method:
            QMessageBox.warning(self, "Thiếu thuật toán", "Vui lòng chọn thuật toán phát hiện cạnh!")
            return

        self.image_path = img_path
        self.display_image(img_path, self.ui.label_Input)

        # Phát hiện cạnh
        self.detect_edges()

        # Tiến hành nhúng
        self.embed_file()

    # --------- Hiển thị ảnh lên QLabel ----------
    def display_image(self, path, label):
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Lỗi", f"Không thể đọc ảnh: {path}")
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.width(), 
            label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    # --------- ✅ SỬA: Phát hiện cạnh và LƯU parameters ----------
    def detect_edges(self):
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        
        try:
            if self.selected_method == "sobel":
                # Có thể thay đổi parameters này
                threshold = 40
                smooth = True
                self.edge_map = sobel_edge_optimized(img, threshold=threshold, smooth=smooth)
                
                # ✅ LƯU parameters
                self.edge_params = {
                    "threshold": threshold,
                    "smooth": smooth
                }
                
            elif self.selected_method == "canny":
                # Có thể thay đổi parameters này
                blur_ksize = (5, 5)
                sigma = 1.2
                th1 = 80
                th2 = 160
                self.edge_map = canny_with_blur(
                    img, 
                    blur_ksize=blur_ksize,
                    sigma=sigma,
                    th1=th1,
                    th2=th2
                )
                
                # ✅ LƯU parameters
                self.edge_params = {
                    "blur_ksize": list(blur_ksize),  # tuple -> list for JSON
                    "sigma": sigma,
                    "th1": th1,
                    "th2": th2
                }
                
            elif self.selected_method == "hybrid":
                # Parameters cho Canny bên trong Hybrid
                canny_params = {
                    "blur_ksize": (5, 5),
                    "sigma": 1.2,
                    "th1": 80,
                    "th2": 160
                }
                _, _, self.edge_map = hybrid_fuzzy_or_canny(img, canny_params)
                
                # ✅ LƯU parameters
                self.edge_params = {
                    "canny_params": {
                        "blur_ksize": list(canny_params["blur_ksize"]),
                        "sigma": canny_params["sigma"],
                        "th1": canny_params["th1"],
                        "th2": canny_params["th2"]
                    }
                }
            
            # Tự động lưu edge map vào thư mục tạm
            temp_edge_path = "temp_edge_map.png"
            cv2.imwrite(temp_edge_path, self.edge_map)
            self.edge_map_path = temp_edge_path
            
            print(f"[*] Đã phát hiện cạnh bằng {self.selected_method}")
            print(f"[*] Edge params: {self.edge_params}")
            print(f"[*] Edge map tạm thời: {temp_edge_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi phát hiện cạnh", str(e))

    # --------- ✅ SỬA: Nhúng và LƯU metadata ----------
    def embed_file(self):
        try:
            message = self.ui.txt_SecretMessage.text().strip()
            if not message:
                QMessageBox.warning(self, "Lỗi", "Bạn chưa nhập thông điệp để nhúng!")
                return

            output_img = "stego_temp.png"

            # Gọi hàm nhúng
            bits_used = embed_message(
                cover_img_or_path=self.image_path,
                edge_map=self.edge_map,
                secret_message=message,
                output_path=output_img,
                x=self.x_bits,
                y=self.y_bits
            )

            self.stego_img_path = output_img
            self.display_image(output_img, self.ui.label_Output)

            # Tính metrics
            psnr, payload, bpp = compute_metrics(self.image_path, output_img, bits_used)

            # ✅ THÊM: Tự động lưu metadata tạm thời
            self.save_metadata_temp()

            msg = (
                f"✅ Đã nhúng thông điệp thành công!\n\n"
                f"📊 Thông số đánh giá:\n"
                f"• PSNR = {psnr:.2f} dB\n"
                f"• Bits Embedded = {payload}\n"
                f"• BPP = {bpp:.4f}\n\n"
                f"⚠️ LƯU Ý: Metadata đã được tạo tạm thời.\n"
                f"Khi lưu ảnh stego, metadata sẽ được lưu cùng thư mục!"
            )
            QMessageBox.information(self, "Kết quả ẩn tin", msg)

        except ValueError as e:
            QMessageBox.warning(self, "Lỗi nhúng", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Lỗi không xác định", str(e))

    # --------- ✅ THÊM: Lưu metadata tạm thời ----------
    def save_metadata_temp(self):
        """Lưu metadata tạm thời (sẽ được copy khi user lưu stego image)"""
        if not self.stego_img_path:
            return
        
        metadata = {
            "algorithm": self.selected_method,
            "x_bits": self.x_bits,
            "y_bits": self.y_bits,
            "edge_params": self.edge_params,
            "original_image": os.path.basename(self.image_path) if self.image_path else None
        }
        
        # Lưu cùng tên với stego temp
        metadata_path = self.stego_img_path.replace(".png", "_metadata.json")
        
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"[*] Metadata tạm thời: {metadata_path}")
        except Exception as e:
            print(f"[!] Lỗi lưu metadata tạm: {e}")

    # --------- ✅ SỬA: Lưu ảnh stego + metadata ----------
    def save_stego_image(self):
        if not self.stego_img_path or not os.path.exists(self.stego_img_path):
            QMessageBox.warning(self, "Chưa có ảnh", "Chưa có ảnh ẩn mã để lưu.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu ảnh ẩn mã",
            "stego_image.png",
            "PNG Files (*.png)"  # ✅ CHỈ DÙNG PNG để tránh nén
        )
        if save_path:
            try:
                # Lưu ảnh stego
                shutil.copy(self.stego_img_path, save_path)
                
                # ✅ THÊM: Lưu metadata cùng thư mục
                metadata_source = self.stego_img_path.replace(".png", "_metadata.json")
                metadata_dest = save_path.replace(".png", "_metadata.json")
                
                if os.path.exists(metadata_source):
                    shutil.copy(metadata_source, metadata_dest)
                    
                    QMessageBox.information(
                        self,
                        "Thành công",
                        f"✅ Đã lưu thành công!\n\n"
                        f"📁 Ảnh stego: {save_path}\n"
                        f"📄 Metadata: {metadata_dest}\n\n"
                        f"⚠️ GIỮ CẢ 2 FILE NÀY để có thể trích xuất sau!"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Thiếu metadata",
                        f"Ảnh đã được lưu tại:\n{save_path}\n\n"
                        f"⚠️ NHƯNG không tìm thấy metadata!\n"
                        f"Bạn sẽ phải nhập THỦ CÔNG các thông số khi trích xuất."
                    )
                
            except Exception as e:
                QMessageBox.critical(self, "Lỗi lưu ảnh", str(e))

    # --------- 💾 Lưu edge map (tuỳ chọn) ----------
    def save_edge_map(self):
        """Lưu edge map riêng (không bắt buộc nếu đã có metadata)"""
        if self.edge_map is None:
            QMessageBox.warning(self, "Chưa có Edge Map", "Chưa phát hiện cạnh để lưu!")
            return

        suggested_name = "edge_map.png"
        if self.image_path:
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            suggested_name = f"{base_name}_edge_map.png"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu Edge Map (tuỳ chọn)",
            suggested_name,
            "PNG Files (*.png)"
        )
        if save_path:
            try:
                cv2.imwrite(save_path, self.edge_map)
                self.edge_map_path = save_path
                
                QMessageBox.information(
                    self, 
                    "Thành công", 
                    f"✅ Edge Map đã được lưu tại:\n{save_path}\n\n"
                    f"ℹ️ LƯU Ý: Nếu có metadata file, bạn KHÔNG CẦN edge map này.\n"
                    f"Edge map sẽ được tự động tái tạo từ metadata!"
                )
            except Exception as e:
                QMessageBox.critical(self, "Lỗi lưu Edge Map", str(e))