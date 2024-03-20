import threading
import socket
import cv2
import numpy as np
from pynput import keyboard
import datetime
import os
import airsim


class Drone:
    def __init__(self):
        # self.client = None

        self.active_keys = set()
        self.move_keys = {'w', 'a', 's', 'd',
                          keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right}
        self.control_keys = {'c',  # close
                             't',  # take off
                             'l',  # land
                             'p',  # reset
                             'v',  # auto navigation
                             }
        self.valid_keys = self.move_keys | self.control_keys

        self.camera_width = 320
        self.camera_height = 240
        self.roll = self.pitch = self.thrust = self.yaw = 0.
        self.speed = 2.

        self.is_connected = False  # 可用于控制线程运行的标志
        self.is_flying = False  # 是否正在飞行
        self.is_navigating = False  # 是否正在导航（跟踪目标）
        self.navigation_start_sequence = True
        self.is_recording = False  # 是否正在记录数据
        self.images = []
        self.controls = []

        self.video_thread = None

        self.client = airsim.MultirotorClient()  # connect to the AirSim simulator
        self.client.enableApiControl(True)  # 获取控制权
        self.client.armDisarm(True)  # 解锁

        # self.client.simSetDetectionFilterRadius("0", airsim.ImageType.Scene, 80 * 100)  # in [cm]
        self.client.simAddDetectionFilterMeshName("0", airsim.ImageType.Scene, "target")

        # self.model = Model()  # 导航控制模型
        # self.model.load()

    # 连接无人机
    def connect(self):
        if self.is_connected:
            return  # 如果已经连接，则不重复执行以下操作

        if self.video_thread is None or not self.video_thread.is_alive():
            client1 = airsim.MultirotorClient()  # connect to the AirSim simulator
            client1.enableApiControl(True)  # 获取控制权
            client1.armDisarm(True)  # 解锁

            self.video_thread = threading.Thread(target=self._video_stream, args=(client1,))
            self.video_thread.start()

        self.is_connected = True

    def _video_stream(self, client):
        cv2.destroyAllWindows()
        while self.is_connected:
            # 一次获取一张图片

            response = client.simGetImage(0, airsim.ImageType.Scene)
            img_png = np.frombuffer(response, dtype=np.uint8)
            try:
                img_bgr = cv2.imdecode(img_png, cv2.IMREAD_COLOR)
            except:
                print(type(img_png))
                continue
            # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # if self.is_recording:  # 正在记录数据
            #     self.images.append(img_rgb)
            #     self.controls.append([self.roll, self.pitch, self.thrust, self.yaw])

            cv2.imshow('Video', img_bgr)
            cv2.waitKey(1)

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
        elif 'p' in self.active_keys:
            self.reset()
            return
        elif 'v' in self.active_keys:
            self.auto_navigation()
            return

    def start_keyboard_listener(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            while self.is_connected:
                listener.join(0.1)
            listener.stop()

    def move(self):
        # roll, pitch, thrust, yaw
        self.client.moveByVelocityBodyFrameAsync(
            vx=self.pitch * self.speed, vy=self.roll * self.speed, vz=-self.thrust, duration=30,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw * 30.))

    def move_wait(self):
        self.client.moveByVelocityBodyFrameAsync(
            vx=self.pitch * self.speed, vy=self.roll * self.speed, vz=-self.thrust * self.speed, duration=0.05,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw * 30.)).join()

    def take_off(self):
        if not self.is_connected:
            return

        self.client.takeoffAsync()
        self.is_flying = True
        print('Took off')

    def land(self):
        if not self.is_connected:
            return

        self.client.landAsync()  # 降落
        self.is_flying = False
        print("landed")

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)  # 获取控制权
        self.client.armDisarm(True)  # 解锁
        self.take_off()
        self.client.hoverAsync()

    def start_recording(self):
        if not self.is_connected or self.is_recording:
            return
        self.images.clear()
        self.controls.clear()
        self.is_recording = True
        print("Start recording!")

    def stop_recording(self):
        if not self.is_connected or not self.is_recording:
            return
        self.is_recording = False

        images_array = np.array(self.images, dtype=np.uint8)
        controls_array = np.array(self.controls, dtype=np.float32)

        # 保存到文件
        root_path = "./datasets/uav_recording"
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前日期和时间
        save_path = os.path.join(root_path, f"recording_{current_time}_{len(self.images)}.npz")
        np.savez(save_path, images=images_array, controls=controls_array)
        print(f"Saved to: {save_path}")

        # 清空列表，以便于下次记录
        self.images.clear()
        self.controls.clear()

    def auto_navigation(self):
        target_position = self.client.simGetObjectPose("target").position
        is_completed = False
        stationary_count = 0
        self.start_recording()

        while True:
            # 检测目标是否在视野内
            self.client.simPause(True)
            detection = self.client.simGetDetections("0", airsim.ImageType.Scene)
            if len(detection) == 0:
                detection = self.client.simGetDetections("0", airsim.ImageType.Scene)
            if len(detection) > 0:
                min_vector2d = detection[0].box2D.min
                max_vector2d = detection[0].box2D.max
                x_center = (min_vector2d.x_val + max_vector2d.x_val) / 2.
                y_center = (min_vector2d.y_val + max_vector2d.y_val) / 2.
            else:
                print("No target detected")
                continue

            position = self.client.simGetVehiclePose().position
            distance = (position - target_position).get_length()
            self.roll = self.pitch = self.thrust = self.yaw = 0.

            if is_completed:
                self.roll = self.pitch = self.thrust = self.yaw = 0.
            else:
                if x_center < self.camera_width * 0.48:
                    self.yaw = -1.
                elif x_center > self.camera_width * 0.52:
                    self.yaw = 1.
                else:
                    self.yaw = 0.

                if y_center < self.camera_height * 0.45:
                    self.thrust = 1.
                elif y_center > self.camera_height * 0.55:
                    self.thrust = -1.
                else:
                    self.thrust = 0.

                if self.yaw != 0 or self.thrust != 0:
                    self.pitch = 0.8
                else:
                    self.pitch = 1.

            if distance <= 3.5:
                is_completed = True

            response = self.client.simGetImage("0", airsim.ImageType.Scene)
            img_png = np.frombuffer(response, dtype=np.uint8)
            try:
                img_bgr = cv2.imdecode(img_png, cv2.IMREAD_COLOR)
            except:
                raise Exception(f"Image error! Got type: {type(img_png)}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.images.append(img_rgb)
            self.controls.append([self.roll, self.pitch, self.thrust, self.yaw])

            if is_completed:
                stationary_count += 1
                if stationary_count >= 15:
                    print("Completed!")
                    break

            self.client.simPause(False)
            self.move_wait()

        self.stop_recording()
        self.client.simPause(False)

    def close(self):
        if not self.is_connected:
            return

        # 确保无人机处于降落状态
        if self.is_flying:
            self.land()

        self.is_connected = False

        self.client.armDisarm(False)  # 上锁
        self.client.enableApiControl(False)  # 释放控制权

        print("UAV connection closed.")
        exit()

    def main(self):
        print('Welcome to the Drone AirSim Control System!\r\n')

        self.connect()

        keyboard_thread = threading.Thread(target=self.start_keyboard_listener)
        keyboard_thread.start()


if __name__ == "__main__":
    drone = Drone()
    drone.main()
