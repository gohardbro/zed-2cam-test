import sys
import cv2
import numpy as np
import pyzed.sl as sl
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QSizePolicy, QSpinBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

# --- Depth 처리 함수--- clip_range 범위를 크게 할수록 넓게 표현
def process_depth_image(depth_np, clip_range=(0, 1000), colormap=cv2.COLORMAP_JET):
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    depth_np_clipped = np.clip(depth_np, clip_range[0], clip_range[1])
    depth_normalized = cv2.normalize(depth_np_clipped, None, 0, 255, cv2.NORM_MINMAX)
    depth_8u = depth_normalized.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8u, colormap)
    return depth_color

class ZEDCameraViewer:
    def __init__(self, clip_range=(0, 5000), colormap=cv2.COLORMAP_JET):
        self.camera = sl.Camera()
        self.init_params = sl.InitParameters()
        self.runtime = sl.RuntimeParameters()
        self.depth_mat = sl.Mat()
        self.running = False
        self.clip_range = clip_range
        self.colormap = colormap

    def set_resolution(self, resolution_text):
        resolutions = {
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA
        }
        self.init_params.camera_resolution = resolutions.get(resolution_text, sl.RESOLUTION.HD720)

    def set_fps(self, fps_text):
        self.init_params.camera_fps = int(fps_text)

    def set_clip_range(self, clip_min, clip_max):
        self.clip_range = (clip_min, clip_max)

    def start(self):
        if self.camera.open(self.init_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("카메라 열기 실패")
        self.running = True

    def stop(self):
        self.running = False
        self.camera.close()

    def get_depth_image(self):
        if not self.running:
            return None
        if self.camera.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
            depth_np = self.depth_mat.get_data()
            h, w = depth_np.shape[:2]
            if h == 0 or w == 0:
                return None
            return process_depth_image(depth_np, self.clip_range, self.colormap)
        return None

class DualZEDViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZED Dual Depth Viewer")
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.viewer1 = ZEDCameraViewer()
        self.viewer2 = ZEDCameraViewer()

        self.label1 = QLabel("Camera 1")
        self.label2 = QLabel("Camera 2")

        self.res_combo1 = QComboBox()
        self.res_combo2 = QComboBox()
        for res in ["HD1080", "HD720", "VGA"]:
            self.res_combo1.addItem(res)
            self.res_combo2.addItem(res)

        self.fps_combo1 = QComboBox()
        self.fps_combo2 = QComboBox()
        for fps in ["15", "30"]:
            self.fps_combo1.addItem(fps)
            self.fps_combo2.addItem(fps)

        self.clip_min_box = QSpinBox()
        self.clip_min_box.setRange(0, 10000)
        self.clip_min_box.setValue(0)
        self.clip_min_box.setPrefix("Min: ")

        self.clip_max_box = QSpinBox()
        self.clip_max_box.setRange(0, 10000)
        self.clip_max_box.setValue(1000)
        self.clip_max_box.setPrefix("Max: ")

        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["Both Cameras", "Camera 1 Only", "Camera 2 Only"])

        self.image_label1 = QLabel()
        self.image_label2 = QLabel()
        for lbl in [self.image_label1, self.image_label2]:
            lbl.setScaledContents(True)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setMinimumWidth(200)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_viewing)

        config_layout = QGridLayout()
        config_layout.addWidget(self.label1, 0, 0)
        config_layout.addWidget(self.res_combo1, 0, 1)
        config_layout.addWidget(self.fps_combo1, 0, 2)
        config_layout.addWidget(self.label2, 1, 0)
        config_layout.addWidget(self.res_combo2, 1, 1)
        config_layout.addWidget(self.fps_combo2, 1, 2)
        config_layout.addWidget(self.clip_min_box, 2, 0)
        config_layout.addWidget(self.clip_max_box, 2, 1)
        config_layout.addWidget(self.camera_selector, 2, 2)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_label1, stretch=1)
        image_layout.addWidget(self.image_label2, stretch=1)

        layout = QVBoxLayout()
        layout.addLayout(config_layout)
        layout.addWidget(self.start_button)
        layout.addLayout(image_layout)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_images)

    def start_viewing(self):
        clip_min = self.clip_min_box.value()
        clip_max = self.clip_max_box.value()
        self.viewer1.set_resolution(self.res_combo1.currentText())
        self.viewer2.set_resolution(self.res_combo2.currentText())
        self.viewer1.set_fps(self.fps_combo1.currentText())
        self.viewer2.set_fps(self.fps_combo2.currentText())
        self.viewer1.set_clip_range(clip_min, clip_max)
        self.viewer2.set_clip_range(clip_min, clip_max)

        selected_mode = self.camera_selector.currentText()

        if selected_mode in ["Both Cameras", "Camera 1 Only"]:
            self.viewer1.start()
        else:
            self.image_label1.clear()

        if selected_mode in ["Both Cameras", "Camera 2 Only"]:
            self.viewer2.start()
        else:
            self.image_label2.clear()

        self.timer.start(30)

    def update_images(self):
        selected_mode = self.camera_selector.currentText()

        if selected_mode in ["Both Cameras", "Camera 1 Only"]:
            img1 = self.viewer1.get_depth_image()
            if img1 is not None:
                self.image_label1.setPixmap(self.cv_to_pixmap(img1))

        if selected_mode in ["Both Cameras", "Camera 2 Only"]:
            img2 = self.viewer2.get_depth_image()
            if img2 is not None:
                self.image_label2.setPixmap(self.cv_to_pixmap(img2))

    def cv_to_pixmap(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DualZEDViewer()
    viewer.show()
    sys.exit(app.exec_())
