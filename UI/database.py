import mysql.connector

# Cấu hình kết nối MySQL
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # điền password nếu có
        database="myapp"  # tạo database user_db trước trong XAMPP
    )

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE,
            password VARCHAR(255)
        )
    """)
    conn.commit()
    conn.close()

def check_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def add_user(username, password):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()
