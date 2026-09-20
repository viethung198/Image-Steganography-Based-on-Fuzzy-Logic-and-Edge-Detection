# app_controller.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6.QtCore import Qt

# -------------------- Import UI --------------------
from UI.ui_formlogin import Ui_Form as Ui_LoginForm
from UI.ui_form_Register import Ui_Form as Ui_RegisterForm
from UI.ui_Menu_User import Ui_Form as Ui_MenuUser
from UI.EdgeDetection_ui import Ui_EdgeDetectionForm
from EdgeDetectionWindow import EdgeDetectionWindow
from EmbedWindow import EmbedWindow
from ExtractWindow import ExtractWindow

# -------------------- Import Database --------------------
from UI.database import check_user, add_user, get_connection

# -------------------- Helper --------------------
def check_username_exists(username):
    """Kiểm tra username đã tồn tại chưa"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

# -------------------- Login Window --------------------
class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_LoginForm()
        self.ui.setupUi(self)

        # Map widgets
        self.le_username = self.ui.txbAccount
        self.le_password = self.ui.txbPassword
        self.btn_login = self.ui.btnLogin
        self.btn_register = self.ui.btn_Register
        self.btn_exit = self.ui.btn_Exit

        # Connect signals
        self.btn_login.clicked.connect(self.on_login)
        self.btn_register.clicked.connect(self.open_register)
        self.btn_exit.clicked.connect(QApplication.quit)

    def on_login(self):
        username = self.le_username.text().strip()
        password = self.le_password.text()
        if not username or not password:
            QMessageBox.warning(self, "Thông báo", "Vui lòng nhập tài khoản và mật khẩu")
            return

        if check_user(username, password):
            self.menu_window = MenuWindow(username)
            self.menu_window.show()
            self.close()
        else:
            QMessageBox.warning(self, "Thông báo", "Tên tài khoản hoặc mật khẩu sai")

    def open_register(self):
        self.register_window = RegisterWindow()
        self.register_window.show()
        self.close()

# -------------------- Register Window --------------------
class RegisterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_RegisterForm()
        self.ui.setupUi(self)

        # Map widgets
        self.le_username = self.ui.txb_accountname
        self.le_password = self.ui.txb_Password1
        self.le_password_repeat = self.ui.txb_Password2
        self.btn_register = self.ui.btn_RegisterConfirm
        self.btn_exit = self.ui.btn_ExitRegister

        # Connect signals
        self.btn_register.clicked.connect(self.on_register)
        self.btn_exit.clicked.connect(self.back_to_login)

    def back_to_login(self):
        self.login_window = LoginWindow()
        self.login_window.show()
        self.close()

    def on_register(self):
        username = self.le_username.text().strip()
        password = self.le_password.text()
        password_repeat = self.le_password_repeat.text()

        if not username or not password or not password_repeat:
            QMessageBox.warning(self, "Thông báo", "Vui lòng điền đủ thông tin")
            return

        if password != password_repeat:
            QMessageBox.warning(self, "Thông báo", "Mật khẩu nhập lại không khớp")
            return

        if check_username_exists(username):
            QMessageBox.warning(self, "Thông báo", "Tài khoản đã tồn tại")
            return

        if add_user(username, password):
            QMessageBox.information(self, "Thành công", "Đăng ký thành công")
            self.back_to_login()
        else:
            QMessageBox.critical(self, "Lỗi", "Đăng ký thất bại, thử lại")

# -------------------- Menu Window --------------------
class MenuWindow(QMainWindow):
    def __init__(self, username):
        super().__init__()
        self.ui = Ui_MenuUser()
        self.ui.setupUi(self)

        # Hiển thị tên tài khoản
        self.ui.lb_TypeAccount.setText(f"Tài khoản: {username}")

        # Nút thoát
        self.ui.btn_exit.clicked.connect(QApplication.quit)

        # --- Nút Phát hiện cạnh ---
        self.ui.btn_ManageDocumentMenu.clicked.connect(self.open_edge_detection)

        # --- Nút Ẩn mã (btn_SearchingMenu) ---
        self.ui.btn_SearchingMenu.clicked.connect(self.open_embed_window)

        self.ui.btn_ExtractMenu.clicked.connect(self.open_extract_window)

    
    def open_extract_window(self):
        self.extract_window = ExtractWindow()
        self.extract_window.show()

    def open_edge_detection(self):
        self.edge_window = EdgeDetectionWindow()
        self.edge_window.show()

    def open_embed_window(self):
        self.embed_window = EmbedWindow()
        self.embed_window.show()

# -------------------- Main --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec())
