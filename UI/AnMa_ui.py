# Form implementation generated from reading ui file 'UI\AnMa.ui'
#
# Created by: PyQt6 UI code generator 6.10.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is run again.

from PyQt6 import QtCore, QtGui, QtWidgets

class Ui_AnMaForm(object):
    def setupUi(self, AnMaForm):
        AnMaForm.setObjectName("AnMaForm")
        AnMaForm.resize(766, 740)

        # ===== Tiêu đề =====
        self.label = QtWidgets.QLabel(parent=AnMaForm)
        self.label.setGeometry(QtCore.QRect(230, 10, 361, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")

        # ===== Nhóm thuật toán =====
        self.groupBox = QtWidgets.QGroupBox(parent=AnMaForm)
        self.groupBox.setGeometry(QtCore.QRect(60, 60, 661, 81))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.groupBox.setObjectName("groupBox")

        self.btn_Sobel = QtWidgets.QPushButton(parent=self.groupBox)
        self.btn_Sobel.setGeometry(QtCore.QRect(20, 20, 121, 51))
        self.btn_Sobel.setObjectName("btn_Sobel")

        self.btn_Canny = QtWidgets.QPushButton(parent=self.groupBox)
        self.btn_Canny.setGeometry(QtCore.QRect(270, 20, 131, 51))
        self.btn_Canny.setObjectName("btn_Canny")

        self.btn_Hybrid = QtWidgets.QPushButton(parent=self.groupBox)
        self.btn_Hybrid.setGeometry(QtCore.QRect(510, 20, 121, 51))
        self.btn_Hybrid.setObjectName("btn_Hybrid")

        # ===== Nhóm chính - Ẩn mã =====
        self.groupBox_2 = QtWidgets.QGroupBox(parent=AnMaForm)
        self.groupBox_2.setGeometry(QtCore.QRect(60, 150, 661, 480))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.groupBox_2.setObjectName("groupBox_2")

        # --- Label thông điệp bí mật ---
        self.label_Secret = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_Secret.setGeometry(QtCore.QRect(40, 20, 131, 31))
        self.label_Secret.setObjectName("label_Secret")
        self.label_Secret.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # --- QLineEdit nhập thông điệp ---
        self.txt_SecretMessage = QtWidgets.QLineEdit(parent=self.groupBox_2)
        self.txt_SecretMessage.setGeometry(QtCore.QRect(190, 20, 441, 31))
        self.txt_SecretMessage.setObjectName("txt_SecretMessage")
        self.txt_SecretMessage.setPlaceholderText("Nhập thông điệp bí mật...")

        # --- Ảnh gốc ---
        self.label_Input = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_Input.setGeometry(QtCore.QRect(20, 70, 300, 300))
        self.label_Input.setStyleSheet("border: 2px solid gray; background-color: white;")
        self.label_Input.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_Input.setObjectName("label_Input")

        # --- Ảnh sau khi ẩn mã ---
        self.label_Output = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_Output.setGeometry(QtCore.QRect(350, 70, 301, 300))
        self.label_Output.setStyleSheet("border: 2px solid gray; background-color: white;")
        self.label_Output.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_Output.setObjectName("label_Output")

        # --- 🖼️ Nút CHỌN ẢNH (bên trái) ---
        self.btn_Select = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.btn_Select.setGeometry(QtCore.QRect(90, 380, 161, 41))
        self.btn_Select.setObjectName("btn_Select")

        # --- 🔐 Nút NHÚNG DỮ LIỆU (bên phải) ---
        self.btn_Embed = QtWidgets.QPushButton(parent=self.groupBox_2)
        self.btn_Embed.setGeometry(QtCore.QRect(440, 380, 161, 41))
        self.btn_Embed.setObjectName("btn_Embed")


        # --- 💾 Nút LƯU ẢNH ẨN MÃ (nằm giữa toàn giao diện) ---
        self.btn_SaveImage = QtWidgets.QPushButton(parent=AnMaForm)
        self.btn_SaveImage.setGeometry(QtCore.QRect(280, 660, 201, 41))  # ✅ giữa toàn giao diện
        self.btn_SaveImage.setObjectName("btn_SaveImage")

        # ===== Dịch ngôn ngữ =====
        self.retranslateUi(AnMaForm)
        QtCore.QMetaObject.connectSlotsByName(AnMaForm)

    def retranslateUi(self, AnMaForm):
        _translate = QtCore.QCoreApplication.translate
        AnMaForm.setWindowTitle(_translate("AnMaForm", "Ứng dụng Ẩn mã"))
        self.label.setText(_translate("AnMaForm", "ỨNG DỤNG ẨN GIẤU THÔNG TIN"))
        self.groupBox.setTitle(_translate("AnMaForm", "Chọn thuật toán phát hiện cạnh"))
        self.btn_Sobel.setText(_translate("AnMaForm", "Sobel"))
        self.btn_Canny.setText(_translate("AnMaForm", "Canny"))
        self.btn_Hybrid.setText(_translate("AnMaForm", "Hybrid"))
        self.groupBox_2.setTitle(_translate("AnMaForm", "Ẩn mã dữ liệu vào ảnh"))
        self.label_Secret.setText(_translate("AnMaForm", "Thông điệp bí mật:"))
        self.label_Input.setText(_translate("AnMaForm", "Ảnh gốc"))
        self.label_Output.setText(_translate("AnMaForm", "Ảnh sau khi ẩn mã"))
        self.btn_Select.setText(_translate("AnMaForm", "🖼️ Chọn ảnh"))
        self.btn_Embed.setText(_translate("AnMaForm", "🔐 Nhúng dữ liệu"))
        self.btn_SaveImage.setText(_translate("AnMaForm", "💾 Lưu Ảnh Ẩn Mã"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_AnMaForm()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec())
