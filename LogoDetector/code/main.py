import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QLineEdit, QMessageBox, QStackedWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import json
from image_detector import perform_object_detection, apply_blur
import cv2

class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.processed_photos = 0  # Счетчик обработанных фотографий

class PhotoProcessingApp(QMainWindow):
    def __init__(self):
        super(PhotoProcessingApp, self).__init__()

        self.users = []  # Список зарегистрированных пользователей
        self.processed_image = None  # Переменная для хранения обработанного изображения

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Photo Processing App')
        self.setWindowState(Qt.WindowMaximized)  # Устанавливаем окно во весь экран при запуске

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.stacked_widget = QStackedWidget(self.central_widget)

        # Добавляем виджеты для окна входа, окна регистрации и основного окна
        self.login_widget = self.create_login_widget()
        self.register_widget = self.create_register_widget()
        self.main_widget = self.create_main_widget()

        self.stacked_widget.addWidget(self.login_widget)
        self.stacked_widget.addWidget(self.register_widget)
        self.stacked_widget.addWidget(self.main_widget)

        self.central_layout = QVBoxLayout()
        self.central_layout.addWidget(self.stacked_widget)
        self.central_widget.setLayout(self.central_layout)

        self.user = None  # Данные о текущем пользователе

        # Загружаем зарегистрированных пользователей из файла
        self.load_users()

        # Отображаем окно входа при запуске приложения
        self.stacked_widget.setCurrentWidget(self.login_widget)

    def create_login_widget(self):
        login_widget = QWidget(self)

        login_layout = QVBoxLayout()

        self.login_name_edit = QLineEdit(self)
        self.login_name_edit.setPlaceholderText('Enter your name')
        login_layout.addWidget(self.login_name_edit)

        self.login_email_edit = QLineEdit(self)
        self.login_email_edit.setPlaceholderText('Enter your email')
        login_layout.addWidget(self.login_email_edit)

        login_button = QPushButton('Login', self)
        login_button.clicked.connect(self.login_user)
        login_layout.addWidget(login_button)

        switch_to_register_button = QPushButton('Switch to Register', self)
        switch_to_register_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.register_widget))
        login_layout.addWidget(switch_to_register_button)

        login_widget.setLayout(login_layout)

        login_widget.setStyleSheet("""
            QLineEdit {
                font-size: 16px;
                padding: 10px;
            }
            QPushButton {
                font-size: 16px;
                padding: 10px;
            }
        """)

        return login_widget

    def create_register_widget(self):
        register_widget = QWidget(self)

        register_layout = QVBoxLayout()

        self.register_name_edit = QLineEdit(self)
        self.register_name_edit.setPlaceholderText('Enter your name')
        register_layout.addWidget(self.register_name_edit)

        self.register_email_edit = QLineEdit(self)
        self.register_email_edit.setPlaceholderText('Enter your email')
        register_layout.addWidget(self.register_email_edit)

        register_button = QPushButton('Register', self)
        register_button.clicked.connect(self.register_user)
        register_layout.addWidget(register_button)

        switch_to_login_button = QPushButton('Switch to Login', self)
        switch_to_login_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.login_widget))
        register_layout.addWidget(switch_to_login_button)

        register_widget.setLayout(register_layout)

        register_widget.setStyleSheet("""
            QLineEdit {
                font-size: 16px;
                padding: 10px;
            }
            QPushButton {
                font-size: 16px;
                padding: 10px;
            }
        """)

        return register_widget

    def create_main_widget(self):
        main_widget = QWidget(self)

        main_layout = QVBoxLayout()

        self.user_info_label = QLabel(self)
        main_layout.addWidget(self.user_info_label)

        self.label = QLabel(self)  # Добавляем QLabel для отображения обработанных изображений
        main_layout.addWidget(self.label)

        process_button = QPushButton('Process Photo', self)
        process_button.clicked.connect(self.process_photo)
        main_layout.addWidget(process_button)

        save_button = QPushButton('Save Photo', self)
        save_button.clicked.connect(self.save_photo)
        main_layout.addWidget(save_button)

        logout_button = QPushButton('Logout', self)
        logout_button.clicked.connect(self.logout_user)
        main_layout.addWidget(logout_button)

        main_widget.setLayout(main_layout)

        main_widget.setStyleSheet("""
            QLabel {
                font-size: 18px;
                padding: 10px;
            }
            QPushButton {
                font-size: 18px;
                padding: 10px;
            }
        """)

        return main_widget

    def show_register_window(self):
        self.stacked_widget.setCurrentWidget(self.register_widget)

    def show_login_window(self):
        self.stacked_widget.setCurrentWidget(self.login_widget)

    def show_main_window(self):
        self.stacked_widget.setCurrentWidget(self.main_widget)
        user_info = f'User: {self.user.name}, Email: {self.user.email}, Processed Photos: {self.user.processed_photos}'
        self.user_info_label.setText(user_info)

    def load_users(self):
        try:
            with open('users.json', 'r') as file:
                data = json.load(file)
                for user_data in data:
                    user = User(user_data['name'], user_data['email'])
                    user.processed_photos = user_data.get('processed_photos', 0)
                    self.users.append(user)
        except FileNotFoundError:
            pass  # Если файл не существует, просто проигнорируем

    def save_users(self):
        with open('users.json', 'w') as file:
            data = [{'name': user.name, 'email': user.email, 'processed_photos': user.processed_photos} for user in self.users]
            json.dump(data, file)

    def login_user(self):
        name = self.login_name_edit.text().strip()
        email = self.login_email_edit.text().strip()

        if not name or not email:
            QMessageBox.warning(self, 'Login Error', 'Please enter your name and email.')
            return

        for user in self.users:
            if user.name == name and user.email == email:
                self.user = user
                self.setWindowTitle(f'Photo Processing App - {self.user.name}')
                self.show_main_window()
                return

        QMessageBox.warning(self, 'Login Error', 'User not found. Please register.')

    def register_user(self):
        name = self.register_name_edit.text().strip()
        email = self.register_email_edit.text().strip()

        if name and email:
            self.user = User(name, email)
            self.setWindowTitle(f'Photo Processing App - {self.user.name}')
            self.users.append(self.user)
            self.save_users()  # Сохраняем данные о пользователе в файл
            self.show_main_window()
        else:
            QMessageBox.warning(self, 'Registration Error', 'Please enter your name and email.')

    def process_photo(self):
        if not self.user:
            QMessageBox.warning(self, 'User Not Logged In', 'Please log in or register before processing photos.')
            return

        # Получаем путь к изображению
        image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.jpeg *.bmp)')

        if image_path:
            self.last_image_path = image_path  # Сохраняем путь к последнему открытому изображению
            result, result_without_boxes = perform_object_detection(image_path, return_result_without_boxes=True)
            self.processed_image = result_without_boxes  # Сохраняем обработанное изображение без рамок
            self.show_result(result)

            # Увеличиваем счетчик обработанных фотографий для текущего пользователя
            self.user.processed_photos += 1
            self.save_users()  # Сохраняем обновленные данные в файл

            # Обновляем информацию в окне профиля пользователя
            self.show_main_window()

    def save_photo(self):
        if self.processed_image is None:
            QMessageBox.warning(self, 'No Image', 'There is no processed image to save.')
            return

        # Открываем диалоговое окно для выбора пути сохранения файла
        save_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if save_path:
            # Сохраняем изображение
            cv2.imwrite(save_path, self.processed_image)
            QMessageBox.information(self, 'Image Saved', 'The image has been saved successfully.')

    def show_result(self, result):
        # Задаем фиксированный размер для отображения изображений
        fixed_width = 800
        fixed_height = 600

        # Масштабируем изображение до заданного размера
        result = cv2.resize(result, (fixed_width, fixed_height))

        height, width, channel = result.shape
        bytes_per_line = 3 * width
        qImg = QImage(result.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)

        # Отображаем изображение в QLabel
        self.label.setPixmap(pixmap)
        self.label.setFixedSize(fixed_width, fixed_height)  # Устанавливаем фиксированный размер для QLabel
        self.label.setAlignment(Qt.AlignCenter)

    def logout_user(self):
        self.user = None
        self.setWindowTitle('Photo Processing App')
        self.show_login_window()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PhotoProcessingApp()
    window.show()
    sys.exit(app.exec_())
