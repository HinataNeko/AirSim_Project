import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from DroneControlPanel import Ui_Form
from Drone import Drone


class MainApp(QWidget, Ui_Form):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setupUi(self)

        self.drone = Drone()

        self.connect_button.clicked.connect(self.connect_drone)  # 连接
        self.disconnect_button.clicked.connect(self.disconnect_drone)  # 断开
        self.reset_button.clicked.connect(self.drone.reset)  # 重置

        self.takeoff_button.clicked.connect(self.drone.take_off)
        self.land_button.clicked.connect(self.drone.land)

        self.startRecording_button.clicked.connect(self.drone.start_recording)
        self.stopRecording_button.clicked.connect(self.stop_recording)
        self.startNavigation_button.clicked.connect(self.start_navigation)
        self.stopNavigation_button.clicked.connect(self.stop_navigation)

        self.drone.connection_ready.connect(self.disconnect_drone)  # 断开
        self.drone.message_ready.connect(self.update_message_display)  # 连接消息信号
        self.drone.frame_ready.connect(self.update_video_stream)  # 连接视频帧信号
        self.drone.state_data_ready.connect(self.update_state_data)  # 连接状态数据信号

        self.set_wind_checkBox.stateChanged.connect(self.toggle_random_wind)  # 设置随机风
        self.wind_speed_spinBox.valueChanged.connect(self.update_wind_speed)

        # self.drone.main()

    # 重写 closeEvent 方法
    def closeEvent(self, event):
        self.drone.close()  # 关闭无人机连接
        event.accept()  # 接受关闭事件

    def connect_drone(self):
        self.drone.connect()
        self.indicator.setStyleSheet("background-color: green;")
        self.indicator_label.setText("已连接")

    def disconnect_drone(self, connection=False):
        if not connection:
            self.drone.close()
            self.reset_ui()
            self.indicator.setStyleSheet("background-color: gray;")
            self.indicator_label.setText("未连接")

    def update_message_display(self, message):
        # 将消息添加到界面组件中
        self.messagesTextEdit.append(message)

    def update_state_data(self, state_data):

        if not self.drone.is_connected:
            return

        def format_float(value):
            if isinstance(value, float):
                return "{:.3f}".format(value)
            else:
                return 'N/A'

        # 更新 UI 上的状态数据
        self.position_x_label.setText(format_float(state_data.get('position_x', 'N/A')))  # x轴位置
        self.position_y_label.setText(format_float(state_data.get('position_y', 'N/A')))  # y轴位置
        self.position_z_label.setText(format_float(state_data.get('position_z', 'N/A')))  # z轴位置

        self.linear_velocity_x_label.setText(format_float(state_data.get('linear_velocity_x', 'N/A')))  # x轴线速度
        self.linear_velocity_y_label.setText(format_float(state_data.get('linear_velocity_y', 'N/A')))  # y轴线速度
        self.linear_velocity_z_label.setText(format_float(state_data.get('linear_velocity_z', 'N/A')))  # z轴线速度

        self.angular_velocity_x_label.setText(format_float(state_data.get('angular_velocity_x', 'N/A')))  # x轴角速度
        self.angular_velocity_y_label.setText(format_float(state_data.get('angular_velocity_y', 'N/A')))  # y轴角速度
        self.angular_velocity_z_label.setText(format_float(state_data.get('angular_velocity_z', 'N/A')))  # z轴角速度

        self.eular_pitch_label.setText(format_float(state_data.get('eular_pitch', 'N/A')))  # 姿态角pitch
        self.eular_roll_label.setText(format_float(state_data.get('eular_roll', 'N/A')))  # 姿态角roll
        self.eular_yaw_label.setText(format_float(state_data.get('eular_yaw', 'N/A')))  # 姿态角yaw

    def update_video_stream(self, frame):
        if not self.drone.is_connected:
            return
        if self.drone.is_recording:
            self.recordingStatus_label.setText(f"Recording: {len(self.drone.images)}")
            self.recordingStatus_label.setStyleSheet("color: blue;")

        # 将OpenCV帧转换为QPixmap
        # rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_image = frame
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format)
        self.videoStreamLabel.setPixmap(pixmap.scaled(self.videoStreamLabel.size(), Qt.KeepAspectRatio))

    def stop_recording(self):
        self.drone.stop_recording()
        self.recordingStatus_label.setText("Not Recording")
        self.recordingStatus_label.setStyleSheet("color: red;")

    def start_navigation(self):
        self.drone.start_navigation()

    def stop_navigation(self):
        self.drone.stop_navigation()

    def toggle_random_wind(self, state):
        if state == Qt.Checked:
            wind_speed = self.wind_speed_spinBox.value()
            self.drone.toggle_random_wind(True, wind_speed)
        else:
            self.drone.toggle_random_wind(False)

    def update_wind_speed(self):
        wind_speed = self.wind_speed_spinBox.value()
        if self.set_wind_checkBox.checkState() == Qt.Checked:
            self.drone.toggle_wind(True, wind_speed=wind_speed)

    def reset_ui(self):
        # 重置 QTextEdit
        # self.messagesTextEdit.clear()

        # 重置 QLabel
        # self.pitch_label.setText('N/A')  # 俯仰角度，度数
        # self.roll_label.setText('N/A')  # 横滚角度，度数
        # self.yaw_label.setText('N/A')  # 偏航偏航，度数
        # self.vgx_label.setText('N/A')  # x轴速度
        # self.vgy_label.setText('N/A')  # y轴速度
        # self.vgz_label.setText('N/A')  # z轴速度
        # self.templ_label.setText('N/A')  # 主板最低温度，摄氏度
        # self.temph_label.setText('N/A')  # 主板最高温度，摄氏度
        # self.tof_label.setText('N/A')  # ToF距离，厘米
        # self.height_label.setText('N/A')  # 相对起飞点高度，厘米
        # self.battery_label.setText('N/A')  # 当前电量百分比
        # self.baro_label.setText('N/A')  # 气压计测量高度，米
        # self.time_label.setText('N/A')  # 电机运转时间，秒
        # self.agx_label.setText('N/A')  # x轴加速度
        # self.agy_label.setText('N/A')  # y轴加速度
        # self.agz_label.setText('N/A')  # z轴加速度

        self.videoStreamLabel.clear()
        self.videoStreamLabel.setText("No Video Stream")


def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)  # 自适应缩放
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
