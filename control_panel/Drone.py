import threading
import time
import cv2
import numpy as np
from pynput import keyboard
import datetime
import os
import sys
import random
import math
import airsim

from PyQt5.QtCore import pyqtSignal, QObject

sys.path.append("..")
from cnn_ncps_model import Model


class Drone(QObject):
    connection_ready = pyqtSignal(bool)  # 新增信号，用于发送连接状态
    message_ready = pyqtSignal(str)  # 新增信号，用于发送消息
    frame_ready = pyqtSignal(object)  # 新增信号，用于发送视频帧数据
    state_data_ready = pyqtSignal(dict)  # 新增信号，用于发送状态数据

    def __init__(self):
        super().__init__()
        self.active_keys = set()
        self.move_keys = {'w', 'a', 's', 'd',
                          keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right}
        self.control_keys = {'c',  # close
                             't',  # take off
                             'l',  # land
                             'r',  # start recording
                             'q',  # stop recording
                             'n',  # start navigation
                             'm',  # stop navigation
                             'p',  # reset
                             }
        self.valid_keys = self.move_keys | self.control_keys

        self.roll = self.pitch = self.thrust = self.yaw = 0.
        self.speed = 2.
        self.wind_angle = 0.
        self.wind_speed = 0.

        self.is_connected = False  # 可用于控制线程运行的标志
        self.is_flying = False  # 是否正在飞行
        self.is_navigating = False  # 是否正在导航（跟踪目标）
        self.navigation_start_sequence = True
        self.is_recording = False  # 是否正在记录数据
        self.images = []
        self.controls = []

        self.recv_thread = None
        self.state_thread = None
        self.video_thread = None

        self.model = Model(load_dataset=False)  # 导航控制模型
        self.model.load('../saved_model/CNN_CfC_model.pth')

    def connect(self):
        if self.is_connected:
            return  # 如果已经连接，则不重复执行以下操作

        self.client = airsim.MultirotorClient()  # connect to the AirSim simulator
        self.client.enableApiControl(True)  # 获取控制权
        self.client.armDisarm(True)  # 解锁
        self.client.simGetImage("0", airsim.ImageType.Scene)

        self.is_connected = True

        # 视频流线程
        if self.video_thread is None or not self.video_thread.is_alive():
            client1 = airsim.MultirotorClient()  # connect to the AirSim simulator
            client1.enableApiControl(True)  # 获取控制权
            client1.armDisarm(True)  # 解锁

            self.video_thread = threading.Thread(target=self._video_stream, args=(client1,))
            self.video_thread.start()

        # 状态接收线程
        if self.state_thread is None or not self.state_thread.is_alive():
            client2 = airsim.MultirotorClient()  # connect to the AirSim simulator
            client2.enableApiControl(True)  # 获取控制权
            client2.armDisarm(True)  # 解锁

            self.state_thread = threading.Thread(target=self._recv_state, args=(client2,))
            self.state_thread.start()

        # 键盘控制线程
        keyboard_thread = threading.Thread(target=self.start_keyboard_listener)
        keyboard_thread.start()

    def _recv_state(self, client):
        def parse_state_data(state_data):
            kinematics_state = state_data.kinematics_estimated
            position = kinematics_state.position  # 位置
            linear_velocity = kinematics_state.linear_velocity  # 线速度
            angular_velocity = kinematics_state.angular_velocity  # 角速度
            eularian_angles = airsim.to_eularian_angles(kinematics_state.orientation)  # 姿态角(pitch, roll, yaw)，单位rad

            state_dict = {
                'position_x': position.x_val,
                'position_y': position.y_val,
                'position_z': position.z_val,

                'linear_velocity_x': linear_velocity.x_val,
                'linear_velocity_y': linear_velocity.y_val,
                'linear_velocity_z': linear_velocity.z_val,

                'angular_velocity_x': angular_velocity.x_val,
                'angular_velocity_y': angular_velocity.y_val,
                'angular_velocity_z': angular_velocity.z_val,

                'eular_pitch': eularian_angles[0] * (180 / math.pi),
                'eular_roll': eularian_angles[1] * (180 / math.pi),
                'eular_yaw': eularian_angles[2] * (180 / math.pi),
            }
            return state_dict

        while self.is_connected:
            state = client.getMultirotorState()
            parsed_state_data = parse_state_data(state)
            if self.is_connected:
                self.state_data_ready.emit(parsed_state_data)  # 发射状态数据信号

    def _video_stream(self, client):
        while self.is_connected:
            response = client.simGetImage("0", airsim.ImageType.Scene)
            img_png = np.frombuffer(response, dtype=np.uint8)
            try:
                img_bgr = cv2.imdecode(img_png, cv2.IMREAD_COLOR)
            except:
                print(type(img_png))
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            if self.is_navigating:  # 正在导航
                if self.navigation_start_sequence:
                    controls = self.model.predict(img_rgb, start_sequence=True)
                    self.navigation_start_sequence = False
                else:
                    controls = self.model.predict(img_rgb, start_sequence=False)
                self.roll, self.pitch, self.thrust, self.yaw = [float(i) for i in controls]
                self.move()
            elif self.is_recording:  # 正在记录数据
                self.images.append(img_rgb)
                self.controls.append([self.roll, self.pitch, self.thrust, self.yaw])

            if self.is_connected:
                self.frame_ready.emit(img_rgb)  # 发射信号而不是显示帧

    def _on_press(self, key):
        key_to_check = key.char if hasattr(key, 'char') else key
        if key_to_check in self.valid_keys:
            if hasattr(key, 'char'):
                self.active_keys.add(key.char)
            else:
                self.active_keys.add(key)

            if key_to_check in self.move_keys:
                self.keyboard_move()
            if key_to_check in self.control_keys:
                self.keyboard_control()

    def _on_release(self, key):
        key_to_check = key.char if hasattr(key, 'char') else key
        if key_to_check in self.valid_keys:
            if hasattr(key, 'char'):
                self.active_keys.discard(key.char)
            else:
                self.active_keys.discard(key)

            if key_to_check in self.move_keys:
                self.keyboard_move()
            if key_to_check in self.control_keys:
                self.keyboard_control()

    def keyboard_move(self):
        self.roll = self.pitch = self.thrust = self.yaw = 0
        det = 1.

        if 'w' in self.active_keys:
            self.pitch = det  # 增加俯仰
        if 's' in self.active_keys:
            self.pitch = -det  # 减少俯仰
        if 'a' in self.active_keys:
            self.roll = -det  # 增加横滚
        if 'd' in self.active_keys:
            self.roll = det  # 减少横滚

        # 检查特殊按键
        if keyboard.Key.up in self.active_keys:
            self.thrust = det  # 增加油门
        if keyboard.Key.down in self.active_keys:
            self.thrust = -det  # 减少油门
        if keyboard.Key.left in self.active_keys:
            self.yaw = -det  # 增加偏航
        if keyboard.Key.right in self.active_keys:
            self.yaw = det  # 减少偏航

        if self.is_flying:
            self.move()

    def keyboard_control(self):
        if 'c' in self.active_keys:
            self.close()
            return
        elif 't' in self.active_keys:
            self.take_off()
            self.active_keys.discard('t')
            return
        elif 'l' in self.active_keys:
            self.land()
            self.active_keys.discard('l')
            return
        elif 'r' in self.active_keys:
            self.start_recording()
            return
        elif 'q' in self.active_keys:
            self.stop_recording()
            return
        elif 'n' in self.active_keys:
            self.start_navigation()
            return
        elif 'm' in self.active_keys:
            self.stop_navigation()
            return
        elif 'p' in self.active_keys:
            self.reset()
            return

    def start_keyboard_listener(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            while self.is_connected:
                listener.join(0.1)
            listener.stop()

    def move(self):
        # roll, pitch, thrust, yaw
        self.client.moveByVelocityBodyFrameAsync(
            vx=self.pitch * self.speed, vy=self.roll * self.speed, vz=-self.thrust * self.speed, duration=30,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw * 30.))

    def take_off(self):
        if not self.is_connected:
            return

        self.client.takeoffAsync()
        self.is_flying = True
        self.message_ready.emit('Took off!')

    def land(self):
        if not self.is_connected:
            return

        self.client.landAsync()  # 降落
        self.is_flying = False
        self.message_ready.emit("landed!")

    def reset(self):
        if not self.is_connected:
            return

        self.client.reset()
        self.client.enableApiControl(True)  # 获取控制权
        self.client.armDisarm(True)  # 解锁
        self.take_off()
        self.client.hoverAsync()

        start_position = airsim.Pose(airsim.Vector3r(0, 0, 0))
        self.client.simSetVehiclePose(start_position, ignore_collision=True)

    def start_recording(self):
        if not self.is_connected or self.is_recording:
            return
        self.images.clear()
        self.controls.clear()
        self.is_recording = True
        self.message_ready.emit("Start recording!")  # 发射信号

    def stop_recording(self):
        if not self.is_connected or not self.is_recording:
            return
        self.is_recording = False

        images_array = np.array(self.images, dtype=np.uint8)
        controls_array = np.array(self.controls, dtype=np.float32)

        # 保存到文件
        root_path = "./datasets/uav_recording"
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前日期和时间
        save_path = os.path.join(root_path, f"recording_{len(self.images)}_{current_time}.npz")
        np.savez(save_path, images=images_array, controls=controls_array)
        self.message_ready.emit(f"Recording saved to: {save_path}")  # 发射信号

        # 清空列表，以便于下次记录
        self.images.clear()
        self.controls.clear()

    def start_navigation(self):
        if not self.is_connected:
            return
        self.is_navigating = True
        self.navigation_start_sequence = True
        self.message_ready.emit("Start Navigation!")

    def stop_navigation(self):
        self.is_navigating = False
        self.message_ready.emit("Stop Navigation!")

    def toggle_wind(self, wind=True, wind_speed=None, wind_angle=None):
        if not self.is_connected:
            return
        if wind_speed is not None:
            self.wind_speed = wind_speed
        if wind_angle is not None:
            self.wind_angle = wind_angle
        if wind:
            wind_x = self.wind_speed * math.cos(self.wind_angle)  # x轴风速
            wind_y = self.wind_speed * math.sin(self.wind_angle)  # y轴风速
            wind = airsim.Vector3r(wind_x, wind_y, 0)  # z轴风速设置为0
            self.client.simSetWind(wind)
            # print(f"Setting wind to {wind_speed}m/s")
        else:
            # 重置风为 0
            wind = airsim.Vector3r(0, 0, 0)
            self.client.simSetWind(wind)
            # print("Resetting wind to 0")

    def toggle_random_wind(self, wind=True, wind_speed=10.):
        if not self.is_connected:
            return

        wind_angle = random.uniform(0, 2 * math.pi)
        self.toggle_wind(wind=wind, wind_speed=wind_speed, wind_angle=wind_angle)

    def close(self):
        if not self.is_connected:
            return

        # 确保无人机处于降落状态
        if self.is_flying:
            self.land()

        self.is_connected = False
        self.connection_ready.emit(False)

        # 结束其他后台线程
        if self.state_thread is not None:
            self.state_thread.join()
        if self.video_thread is not None:
            self.video_thread.join()

        self.client.armDisarm(False)  # 上锁
        self.client.enableApiControl(False)  # 释放控制权

        self.message_ready.emit("UAV connection closed.")

    def main(self):
        pass


if __name__ == "__main__":
    drone = Drone()
    drone.main()
