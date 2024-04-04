import threading
import cv2
import numpy as np
import sys
import os
import time
import math
import random
import airsim
import torch

sys.path.append("..")
from cnn_ncps_model import Model


class DroneEnvWrapper:
    def __init__(self, render=True):
        self.camera_width = 320
        self.camera_height = 240
        self.speed = 2.
        self.time_step = 0.05

        self.render = render
        self.is_connected = False  # 可用于控制线程运行的标志
        self.is_flying = False  # 是否正在飞行

        self.video_thread = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Model(load_dataset=False)  # 导航控制模型
        self.model.load()
        self.cnn_model = self.model.model.cnn
        self.cnn_model.eval()

        self.client = airsim.MultirotorClient()  # connect to the AirSim simulator
        self.client.enableApiControl(True)  # 获取控制权
        self.client.armDisarm(True)  # 解锁
        self.connect()

        self.client.simAddDetectionFilterMeshName("0", airsim.ImageType.Scene, "target")

    # 连接无人机
    def connect(self):
        if self.is_connected:
            return  # 如果已经连接，则不重复执行以下操作

        self.client.simGetImage("0", airsim.ImageType.Scene)
        if self.render:
            if self.video_thread is None or not self.video_thread.is_alive():
                client1 = airsim.MultirotorClient()  # connect to the AirSim simulator

                self.video_thread = threading.Thread(target=self._video_stream, args=(client1,))
                self.video_thread.start()

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
            distance_reward = (old_distance - self.distance) * 0.2 / self.time_step
            return distance_reward

        def get_detection_reward():
            detection_reward = - abs(self.target_xywh[0] - 0.5) - abs(self.target_xywh[1] - 0.5)  # (-1, 0)
            # detection_reward += self.target_xywh[2] * self.target_xywh[3]
            # detection_reward *= 0.5
            detection_reward += 0.1
            return 0 if detection_reward > 0 else detection_reward

        # action: np.ndarray, 顺序(roll, pitch, thrust, yaw)
        roll, pitch, thrust, yaw = action.tolist()

        # 移动一个步长
        self.client.simPause(False)
        self.client.moveByVelocityBodyFrameAsync(
            vx=pitch * self.speed, vy=roll * self.speed, vz=-thrust * self.speed, duration=self.time_step,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw * 30.)).join()
        self.client.simSetObjectPose("target", airsim.Pose(self.target_position + self.target_pose_random_offset))

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

        self.client.simPause(True)

        reward = -0.1
        final_reward = 0.
        done = False
        successful = False

        # 目标在视野内
        if len(detection) > 0:
            self.target_xywh = self.get_target_xywh(detection[0])

            # 距离奖励
            distance_reward = get_distance_reward()
            detection_reward = get_detection_reward()

            # 结束
            if self.distance < 3.5:
                final_reward += 50. * (detection_reward + 1)
                done = True
                successful = True
                print("Completed!")
                # distance_reward = -2 * distance_reward
                # detection_reward *= 2
                # reward += 1. * (detection_reward + 1)
                # self.stay_in_range_count += 1
                # if self.stay_in_range_count >= 40:  # 保持目标在指定距离与视野范围内
                #     final_reward += 50.
                #     done = True
                #     print("Completed!")
            else:
                pass
                # if self.stay_in_range_count > 0:
                #     distance_reward -= 10
                #     self.stay_in_range_count = 0

            reward += distance_reward + detection_reward
            self.episode_distance_reward += distance_reward
            self.episode_detection_reward += detection_reward
        else:  # 目标在视野外
            final_reward += -50. if self.distance > 8 else -25.
            done = True
            print("The target moved out of the camera's field of view")

        is_collided = self.client.simGetCollisionInfo().has_collided
        if is_collided:
            if self.distance < 3.5:  # 与目标发生碰撞
                final_reward -= 10
                print("Collided with target!")
            else:
                final_reward -= 25
                print("Collided with other objects!")
            done = True

        # print(f"distance_reward: {distance_reward}\tdetection_reward: {detection_reward}\treward: {reward}")
        reward += final_reward
        self.episode_final_reward += final_reward
        self.episode_reward += reward

        img_png = np.frombuffer(self.client.simGetImage("0", airsim.ImageType.Scene), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_png, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = (img_rgb.astype(np.float32) / 255.).transpose((2, 0, 1))
        with torch.no_grad():
            state = self.cnn_model(torch.tensor(img_float).unsqueeze(0).to(self.device))[0].cpu().numpy()

        return state, reward, done, successful

    def reset(self):
        self.client.simPause(False)
        self.client.reset()
        self.client.enableApiControl(True)  # 获取控制权
        self.client.armDisarm(True)  # 解锁
        self.client.takeoffAsync()
        self.client.hoverAsync()
        self.client.moveByVelocityBodyFrameAsync(vx=0, vy=0, vz=0, duration=0.02).join()

        # agent起始点随机偏移
        # 10~15m
        # max_position_offset_x = 3.
        # max_position_offset_y = 8.
        # max_position_offset_z = 5.

        # 10m
        # max_position_offset_x = 0.
        # max_position_offset_y = 8.
        # max_position_offset_z = 5.

        # 15m
        max_position_offset_x = 0.
        max_position_offset_y = 12.
        max_position_offset_z = 8.

        # 20m
        # max_position_offset_x = 0.
        # max_position_offset_y = 15.
        # max_position_offset_z = 10.

        # 25m
        # max_position_offset_x = 0.
        # max_position_offset_y = 18.
        # max_position_offset_z = 12.

        random_position = airsim.Pose(airsim.Vector3r(
            random.uniform(-max_position_offset_x, max_position_offset_x),
            random.uniform(-max_position_offset_y, max_position_offset_y),
            random.uniform(-max_position_offset_z, max_position_offset_z)))
        self.client.simSetVehiclePose(random_position, ignore_collision=True)

        # 设置目标距离与随机移动
        # self.target_pose = self.client.simGetObjectPose("target")
        self.target_start_pose = airsim.Pose(position_val=airsim.Vector3r(15, 0., 0.),
                                             orientation_val=airsim.Quaternionr(0., 0., 0., 1.))
        self.client.simSetObjectPose("target", self.target_start_pose)
        max_target_position_offset = 0.025
        self.target_pose_random_offset = airsim.Vector3r(
            random.uniform(-max_target_position_offset, max_target_position_offset),
            random.uniform(-max_target_position_offset, max_target_position_offset),
            random.uniform(-max_target_position_offset, max_target_position_offset))

        # 设置随机风
        wind_speed = random.uniform(0, 10)  # 风速
        wind_angle = random.uniform(0, 2 * math.pi)  # 风向
        wind_x = wind_speed * math.cos(wind_angle)  # x轴风速
        wind_y = wind_speed * math.sin(wind_angle)  # y轴风速
        wind = airsim.Vector3r(wind_x, wind_y, 0)  # z轴风速设置为0
        self.client.simSetWind(wind)

        # 重置初始奖励
        self.episode_reward = 0.
        self.episode_distance_reward = 0.
        self.episode_detection_reward = 0.
        self.episode_final_reward = 0.
        self.stay_in_range_count = 0

        # 重置初始位置
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
            img_png = np.frombuffer(self.client.simGetImage("0", airsim.ImageType.Scene), dtype=np.uint8)
            img_bgr = cv2.imdecode(img_png, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_float = (img_rgb.astype(np.float32) / 255.).transpose((2, 0, 1))
            with torch.no_grad():
                state = self.cnn_model(torch.tensor(img_float).unsqueeze(0).to(self.device))[0].cpu().numpy()
        else:
            state = None
            print("Reset Error")

        self.client.simPause(True)
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

        # self.connect()

        # keyboard_thread = threading.Thread(target=self.start_keyboard_listener)
        # keyboard_thread.start()


if __name__ == "__main__":
    drone = DroneEnvWrapper()
