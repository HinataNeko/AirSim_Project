import threading
import socket
import cv2
import numpy as np
from pynput import keyboard
import datetime
import os
import airsim
import time


# from model_ncps import Model


class Drone:
    def __init__(self):
        # self.client = None

        self.active_keys = set()
        self.move_keys = {'w', 'a', 's', 'd',
                          keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right}
        self.control_keys = {'c',  # close
                             't',  # take off
                             'l',  # land
                             'r',  # start recording
                             'q',  # stop recording
                             'p',  # reset
                             }
        self.valid_keys = self.move_keys | self.control_keys

        self.camera_width = 320
        self.camera_height = 240
        self.roll = self.pitch = self.thrust = self.yaw = 0.
        self.speed = 1.

        self.is_connected = False  # 可用于控制线程运行的标志
        self.is_flying = False  # 是否正在飞行
        self.navigation_start_sequence = True
        self.is_recording = False  # 是否正在记录数据

        self.video_thread = None

        self.client = airsim.MultirotorClient()  # connect to the AirSim simulator
        self.client.enableApiControl(True)  # 获取控制权
        self.client.armDisarm(True)  # 解锁

        # self.model = Model()  # 导航控制模型
        # self.model.load()

    # 连接无人机
    def connect(self):
        if self.is_connected:
            return  # 如果已经连接，则不重复执行以下操作

        if self.video_thread is None or not self.video_thread.is_alive():
            client1 = airsim.MultirotorClient()  # connect to the AirSim simulator
            client1.simAddDetectionFilterMeshName("0", airsim.ImageType.Scene, "target")

            self.video_thread = threading.Thread(target=self._video_stream, args=(client1,))
            self.video_thread.start()

        self.is_connected = True

    def _video_stream(self, client):
        cv2.destroyAllWindows()
        while self.is_connected:
            # 一次获取一张图片
            response = client.simGetImage("0", airsim.ImageType.Scene)
            img_png = np.frombuffer(response, dtype=np.uint8)
            try:
                img_bgr = cv2.imdecode(img_png, cv2.IMREAD_COLOR)
                img_bgr_copy = img_bgr.copy()
            except:
                print(type(img_png))
                continue
            # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            detection = client.simGetDetections("0", airsim.ImageType.Scene)
            if len(detection) == 0:
                detection = client.simGetDetections("0", airsim.ImageType.Scene)

            # 目标在视野内
            if len(detection) > 0:
                xywh = self.get_target_xywh(detection[0])

                # 在图像上绘制矩形
                x_min, y_min, x_max, y_max = self.xywh2xyxy(xywh)
                cv2.rectangle(img_bgr_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                if self.is_recording:  # 正在记录数据
                    # 保存到文件
                    yolo_root_path = "./datasets/yolo/all"
                    name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前日期和时间
                    image_save_path = os.path.join(yolo_root_path, "images", f"{name}.jpg")
                    label_save_path = os.path.join(yolo_root_path, "labels", f"{name}.txt")

                    cv2.imwrite(image_save_path, img_bgr)

                    # 将xywh数据写入文件
                    with open(label_save_path, "w") as f:
                        line = f"0 {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n"
                        f.write(line)
                    pass

            cv2.imshow('Video', img_bgr_copy)
            cv2.waitKey(10)

    def get_target_xywh(self, detection):
        min_vector2d = detection.box2D.min
        max_vector2d = detection.box2D.max
        x = (min_vector2d.x_val + max_vector2d.x_val) / 2.
        y = (min_vector2d.y_val + max_vector2d.y_val) / 2.
        w = max_vector2d.x_val - min_vector2d.x_val
        h = max_vector2d.y_val - min_vector2d.y_val

        x /= self.camera_width
        y /= self.camera_height
        w /= self.camera_width
        h /= self.camera_height
        return x, y, w, h

    def xywh2xyxy(self, xywh):
        x, y, w, h = xywh

        # 将比例转换回像素坐标
        x *= self.camera_width
        y *= self.camera_height
        w *= self.camera_width
        h *= self.camera_height

        # 计算矩形的左上角和右下角坐标
        x_min = int(x - w / 2)
        y_min = int(y - h / 2)
        x_max = int(x + w / 2)
        y_max = int(y + h / 2)

        return x_min, y_min, x_max, y_max

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
        self.is_recording = True
        print("Start recording!")

    def stop_recording(self):
        if not self.is_connected or not self.is_recording:
            return
        self.is_recording = False
        print("Stop recording!")

        # images_array = np.array(self.images, dtype=np.uint8)
        # controls_array = np.array(self.controls, dtype=np.float32)
        #
        # # 保存到文件
        # root_path = "./datasets/uav_recording"
        # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前日期和时间
        # save_path = os.path.join(root_path, f"recording_{len(self.images)}_{current_time}.npz")
        # np.savez(save_path, images=images_array, controls=controls_array)
        # print(f"Saved to: {save_path}")

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
