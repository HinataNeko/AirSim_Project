import threading
import cv2
import numpy as np
from pynput import keyboard
import datetime
import os
import time
import random
import airsim


class DroneEnvWrapper:
    def __init__(self):
        self.active_keys = set()
        self.move_keys = {'w', 'a', 's', 'd',
                          keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right}
        self.control_keys = {'c',  # close
                             't',  # take off
                             'l',  # land
                             'p',  # reset
                             }
        self.valid_keys = self.move_keys | self.control_keys

        self.camera_width = 320
        self.camera_height = 240
        self.roll = self.pitch = self.thrust = self.yaw = 0.
        self.speed = 2.
        self.time_step = 0.05

        self.is_connected = False  # 可用于控制线程运行的标志
        self.is_flying = False  # 是否正在飞行

        self.video_thread = None

        self.client = airsim.MultirotorClient()  # connect to the AirSim simulator
        self.client.enableApiControl(True)  # 获取控制权
        self.client.armDisarm(True)  # 解锁
        self.connect()

        # self.target_position = self.client.simGetObjectPose("target").position

        self.client.simAddDetectionFilterMeshName("0", airsim.ImageType.Scene, "target")

        # self.episode_reward = 0  # 一个episode获得的奖励
        # self.episode_distance_reward = 0
        # self.episode_detection_reward = 0

    # 连接无人机
    def connect(self):
        if self.is_connected:
            return  # 如果已经连接，则不重复执行以下操作

        # if self.video_thread is None or not self.video_thread.is_alive():
        #     client1 = airsim.MultirotorClient()  # connect to the AirSim simulator
        #
        #     self.video_thread = threading.Thread(target=self._video_stream, args=(client1,))
        #     self.video_thread.start()

        self.is_connected = True

    def _video_stream(self, client):
        cv2.destroyAllWindows()
        while self.is_connected:
            # 一次获取一张图片
            img_png = np.frombuffer(client.simGetImage("0", airsim.ImageType.Scene), dtype=np.uint8)
            try:
                img_bgr = cv2.imdecode(img_png, cv2.IMREAD_COLOR)
                # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                x, y, w, h = self.target_xywh

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

                # 在图像上绘制矩形
                cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            except:
                continue

            cv2.imshow('Camera', img_bgr)
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
        elif 'p' in self.active_keys:
            self.reset()
            return

    def start_keyboard_listener(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            while self.is_connected:
                listener.join(0.1)
            listener.stop()

    def move(self):
        self.client.moveByVelocityBodyFrameAsync(
            vx=self.pitch * self.speed, vy=self.roll * self.speed, vz=-self.thrust, duration=60,
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

    def step(self, action):
        # 距离奖励
        def get_distance_reward():
            distance_reward = (old_distance - self.distance) * 0.1 / self.time_step
            if distance_reward < 0:
                distance_reward *= 2
            distance_reward += self.target_xywh[2] * self.target_xywh[3]
            return distance_reward

        def get_detection_reward():
            # old_target_xywh = self.target_xywh
            # print(f"x_center: {x_center}\ty_center: {y_center}")

            # detection_reward = 0.
            # if old_target_xywh[0] < 0.48:
            #     detection_reward += self.target_xywh[0] - old_target_xywh[0]
            # elif old_target_xywh[0] > 0.52:
            #     detection_reward += old_target_xywh[0] - self.target_xywh[0]
            #
            # if old_target_xywh[1] < 0.48:
            #     detection_reward += self.target_xywh[1] - old_target_xywh[1]
            # elif old_target_xywh[1] > 0.58:
            #     detection_reward += old_target_xywh[1] - self.target_xywh[1]
            #
            # detection_reward = detection_reward * 4. / self.time_step
            detection_reward = 0.5 - abs(self.target_xywh[0] - 0.5) - abs(self.target_xywh[1] - 0.5)  # (-0.5, 0.5)
            detection_reward *= 0.4
            # print(detection_reward)

            return detection_reward

        # action: np.ndarray, 顺序(roll, pitch, thrust, yaw)
        roll, pitch, thrust, yaw = action.tolist()

        # 移动一个步长
        self.client.simPause(False)
        self.client.moveByVelocityBodyFrameAsync(
            vx=pitch * self.speed, vy=roll * self.speed, vz=-thrust * self.speed, duration=self.time_step,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw * 30.)).join()
        self.client.simSetObjectPose("target", airsim.Pose(self.target_position + self.target_pose_random_offset))
        self.client.simPause(True)

        # 更新agent和target位置
        self.target_pose = self.client.simGetObjectPose("target")
        self.target_position = self.target_pose.position
        self.target_orientation = self.target_pose.orientation
        self.position = self.client.simGetVehiclePose().position
        old_distance = self.distance
        self.distance = (self.position - self.target_position).get_length()
        detection = self.client.simGetDetections("0", airsim.ImageType.Scene)
        if len(detection) == 0:
            detection = self.client.simGetDetections("0", airsim.ImageType.Scene)

        reward = -0.1
        final_reward = 0.
        done = False

        # 目标在视野内
        if len(detection) > 0:
            self.target_xywh = self.get_target_xywh(detection[0])
            state = np.array(self.target_xywh)

            # 距离奖励
            distance_reward = get_distance_reward()
            detection_reward = get_detection_reward()
            reward += distance_reward + detection_reward

            self.episode_distance_reward += distance_reward
            self.episode_detection_reward += detection_reward

            # 结束
            if self.distance < 3.5:
                if detection_reward > 0:
                    final_reward += 100. * 5 * detection_reward
                # reward += 100.
                done = True
                print("Completed!")
        else:  # 目标在视野外
            final_reward += -50. if self.distance > 10 else -25.
            state = np.array([-1., -1., 0., 0.])
            done = True
            print("The target moved out of the camera's field of view")

        is_collided = self.client.simGetCollisionInfo().has_collided
        if is_collided:
            final_reward -= 1.
            done = True
            print("Collided!")

        # print(f"distance_reward: {distance_reward}\tdetection_reward: {detection_reward}\treward: {reward}")
        # print(state)
        reward += final_reward
        self.episode_reward += reward
        self.episode_final_reward += final_reward

        return state, reward, done

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)  # 获取控制权
        self.client.armDisarm(True)  # 解锁
        self.client.takeoffAsync()
        self.client.hoverAsync()

        # agent起始点随机偏移
        max_position_offset_x = 3.
        max_position_offset_y = 8.
        max_position_offset_z = 5.
        random_position = airsim.Pose(airsim.Vector3r(
            random.uniform(-max_position_offset_x, max_position_offset_x),
            random.uniform(-max_position_offset_y, max_position_offset_y),
            random.uniform(-max_position_offset_z, max_position_offset_z)))
        self.client.simSetVehiclePose(random_position, ignore_collision=True)

        # target起始点
        self.target_start_pose = airsim.Pose(position_val=airsim.Vector3r(9.4, 0., 0.),
                                             orientation_val=airsim.Quaternionr(0., 0., 0., 1.))
        self.client.simSetObjectPose("target", self.target_start_pose)
        max_target_position_offset = 0.01
        self.target_pose_random_offset = airsim.Vector3r(
            random.uniform(-max_target_position_offset, max_target_position_offset),
            random.uniform(-max_target_position_offset, max_target_position_offset),
            random.uniform(-max_target_position_offset, max_target_position_offset))

        self.client.simPause(True)

        self.episode_reward = 0.
        self.episode_distance_reward = 0.
        self.episode_detection_reward = 0.
        self.episode_final_reward = 0.

        self.target_pose = self.client.simGetObjectPose("target")
        self.target_position = self.target_pose.position
        self.target_orientation = self.target_pose.orientation
        self.position = self.client.simGetVehiclePose().position
        self.distance = (self.position - self.target_position).get_length()

        # 检测目标是否在视野内
        detection = self.client.simGetDetections("0", airsim.ImageType.Scene)
        if len(detection) == 0:
            detection = self.client.simGetDetections("0", airsim.ImageType.Scene)

        if len(detection) > 0:
            self.target_xywh = self.get_target_xywh(detection[0])
            state = np.array(self.target_xywh)
        else:
            state = None
            print("Reset Error")

        return state

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

        # keyboard_thread = threading.Thread(target=self.start_keyboard_listener)
        # keyboard_thread.start()


if __name__ == "__main__":
    drone = DroneEnvWrapper()
    drone.main()
