import time
import os
import matplotlib.pyplot as plt
import numpy as np

from EnvWrapper_Image import DroneEnvWrapper
# from cnn_ncps_model import Model
# from evaluation.cnn_rnn_model import Model
# from evaluation.cnn_lstm_model import Model
# from evaluation.cnn_gru_model import Model
from snn.cnn_srnn_model import Model

if __name__ == '__main__':
    env_wrapper = DroneEnvWrapper(render=True, image_noise=False)
    model = Model(load_dataset=False)  # 导航控制模型
    model.load()

    n_total = 100
    n_successful = 0
    n_overtime = 0
    i = 0
    successful_speeds = []

    while i < n_total:
        t0 = time.time()
        step = 0
        episode_duration_history = []
        navigation_start_sequence = True
        state, start_position = env_wrapper.reset()

        for _ in range(800):  # max time step
            t1 = time.time()
            if navigation_start_sequence:
                action = model.predict(state, start_sequence=True)
                navigation_start_sequence = False
            else:
                action = model.predict(state, start_sequence=False)
            episode_duration_history.append(time.time() - t1)
            next_state, reward, done, successful, position = env_wrapper.step(action)

            state = next_state
            step += 1
            if done:
                if successful:
                    n_successful += 1
                    speed = (position - start_position).get_length() / (env_wrapper.time_step * step) * 2 / 3
                    successful_speeds.append(speed)
                    print(f"Successful! Speed: {speed} m/s")
                break

        # 异常episode
        if step < 15:
            continue

        if not done:
            n_overtime += 1
            print("Overtime!")

        i += 1
        print('\rEpisode: {} | '
              'Step: {} | '
              'Running Time: {:.2f} | '
              'Prediction Speed: {:.2f}ms per image'
              .format(i, step, time.time() - t0, 1000 * sum(episode_duration_history) / len(episode_duration_history))
              )

    print(f"Successful rate: {n_successful}/{n_total}={n_successful / n_total}，{n_overtime / n_total} overtime.")
    print(f"Average speed: {sum(successful_speeds) / len(successful_speeds)} m/s")
