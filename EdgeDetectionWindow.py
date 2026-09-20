# EdgeDetectionWindow.py
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

import cv2
from core.edge_detection import sobel_edge_optimized, canny_with_blur, hybrid_fuzzy_or_canny
from UI.EdgeDetection_ui import Ui_EdgeDetectionForm


class EdgeDetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_EdgeDetectionForm()
        self.ui.setupUi(self)
        self.setWindowTitle("Phát hiện cạnh")

        # Thuộc tính lưu trạng thái
        self.selected_algorithm = None
        self.selected_image_path = None
        self.loaded_image_gray = None

        # --- Kết nối nút thuật toán ---
        self.ui.btn_Sobel.clicked.connect(lambda: self.select_algorithm("Sobel"))
        self.ui.btn_Canny.clicked.connect(lambda: self.select_algorithm("Canny"))
        self.ui.btn_Hybrid.clicked.connect(lambda: self.select_algorithm("Hybrid"))

        # --- Kết nối nút chọn ảnh ---
        self.ui.btn_Select.clicked.connect(self.choose_image)

    def select_algorithm(self, name):
        """Chọn thuật toán và đổi màu nút, nếu đã chọn ảnh thì chạy luôn"""
        self.selected_algorithm = name

        # Reset màu tất cả nút
        for btn in [self.ui.btn_Sobel, self.ui.btn_Canny, self.ui.btn_Hybrid]:
            btn.setStyleSheet("background-color: none; color: black;")

        # Màu nút được chọn
        if name == "Sobel":
            self.ui.btn_Sobel.setStyleSheet("background-color: orange; color: black;")
        elif name == "Canny":
            self.ui.btn_Canny.setStyleSheet("background-color: orange; color: black;")
        else:
            self.ui.btn_Hybrid.setStyleSheet("background-color: orange; color: black;")

        # Nếu đã có ảnh, áp dụng thuật toán luôn
        if self.loaded_image_gray is not None:
            self.apply_edge_detection()

    def choose_image(self):
        """Chọn ảnh và hiển thị ảnh gốc, nếu đã chọn thuật toán thì chạy luôn"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.jpg *.jpeg)")
        if not file_path:
            return

        self.selected_image_path = file_path

        # Hiển thị ảnh gốc
        pixmap = QPixmap(file_path).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio)
        self.ui.label_Input.setPixmap(pixmap)

        # Đọc ảnh xám
        img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        self.loaded_image_gray = img_gray

        # Nếu đã chọn thuật toán, áp dụng luôn
        if self.selected_algorithm:
            self.apply_edge_detection()

    def apply_edge_detection(self):
        """Áp dụng thuật toán đã chọn lên ảnh đã chọn và hiển thị kết quả"""
        if self.selected_algorithm is None:
            QMessageBox.information(self, "Thông báo", "Vui lòng chọn thuật toán trước")
            return
        if self.loaded_image_gray is None:
            QMessageBox.information(self, "Thông báo", "Vui lòng chọn ảnh trước")
            return

        img_gray = self.loaded_image_gray

        # Áp dụng thuật toán
        if self.selected_algorithm == "Sobel":
            edge_map = sobel_edge_optimized(img_gray)
        elif self.selected_algorithm == "Canny":
            edge_map = canny_with_blur(img_gray)
        else:  # Hybrid
            _, _, edge_map = hybrid_fuzzy_or_canny(img_gray)

        # Chuyển edge_map thành QPixmap hiển thị
        if len(edge_map.shape) == 3:
            edge_map = cv2.cvtColor(edge_map, cv2.COLOR_BGR2GRAY)
        h, w = edge_map.shape
        qimg = QImage(edge_map.data, w, h, w, QImage.Format.Format_Grayscale8)
        self.ui.label_Output.setPixmap(QPixmap.fromImage(qimg).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
