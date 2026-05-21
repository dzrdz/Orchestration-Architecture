import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from collections import deque
import random


def generate_uniform_points(num_points, max_x, max_y):
    grid_size = int(num_points ** 0.5)
    cell_width = max_x / grid_size
    cell_height = max_y / grid_size
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < num_points:
                x = random.uniform(i * cell_width, (i + 1) * cell_width)
                y = random.uniform(j * cell_height, (j + 1) * cell_height)
                points.append([x, y])
    while len(points) < num_points:
        x = random.uniform(0, max_x)
        y = random.uniform(0, max_y)
        points.append([x, y])
    random.shuffle(points)
    return points


class CustomEnv(gym.Env):
    def __init__(self, num_ap=5, num_ue=10, tot_num=50, conf_threshold=0.6, debug=False):
        super(CustomEnv, self).__init__()
        self.num_ap = num_ap
        self.num_ue = num_ue
        self.tot_num = tot_num
        self.conf_threshold = conf_threshold
        self.debug = debug
        self.process_time = 2.395  # AP 处理时间
        self.aka_time = 30  # 认证时间

        # 生成AP和UE位置
        self.ap_positions = generate_uniform_points(num_ap, 200, 200)
        self.ue_positions = generate_uniform_points(tot_num, 200, 200)

        # 定义动作空间 (每个UE可选择AP或AKA)
        self.action_space = spaces.MultiDiscrete([num_ap + 1] * num_ue)

        # 观测空间 (状态信息: 置信度 + AP排队信息 + UE状态)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_ue * (num_ap * 3 + 2),), dtype=np.float32
        )

        self.reset()

    def reset(self):
        """重置环境"""
        self.total_delay = 0
        self.done = False
        self.current_batch = 0
        self.state = np.random.rand(*self.observation_space.shape).astype(np.float32)

        # 初始化UE和AP状态
        self.ue_states = [False] * self.tot_num  # 记录UE是否完成
        self.ap_queues = [[] for _ in range(self.num_ap)]  # AP的排队列表
        self.ap_processing = [[] for _ in range(self.num_ap)]  # AP当前处理中的任务

        return self.state

    def step(self, action):
        """执行一个批次的UE"""
        print("======== STEP START ========")
        start_idx = self.current_batch * self.num_ue
        end_idx = min(start_idx + self.num_ue, self.tot_num)
        batch_ue = list(range(start_idx, end_idx))

        # 先处理当前AP正在执行的任务
        for ap_index in range(self.num_ap):
            completed_tasks = self.ap_processing[ap_index][:2]  # 处理最多两个任务
            for ue_id in completed_tasks:
                self.ue_states[ue_id] = True
                self.total_delay += self.process_time
            self.ap_processing[ap_index] = []  # 清空已完成的任务

            # 从等待队列中取出新的任务
            while self.ap_queues[ap_index] and len(self.ap_processing[ap_index]) < 2:
                self.ap_processing[ap_index].append(self.ap_queues[ap_index].pop(0))

        # 处理新提交的UE请求
        for ue_id in batch_ue:
            ap_choice = action[ue_id - start_idx]
            if self.ue_states[ue_id]:  # 如果UE已完成，跳过
                continue

            if ap_choice == 0:
                # 选择AKA (认证)
                self.total_delay += self.aka_time
                self.ue_states[ue_id] = True
            else:
                ap_index = ap_choice - 1
                if len(self.ap_processing[ap_index]) < 2:  # AP最多并发2个任务
                    self.ap_processing[ap_index].append(ue_id)
                else:
                    self.ap_queues[ap_index].append(ue_id)  # 放入等待队列
                    self.total_delay += self.process_time  # 队列满了，增加等待时间

            if self.debug:
                print(f"UE {ue_id}: Action {ap_choice}, Total Delay: {self.total_delay}")
                for ap_index in range(self.num_ap):
                    print(f"AP {ap_index}: Processing {self.ap_processing[ap_index]}, Queue {self.ap_queues[ap_index]}")
                print("-----------------------------------")

        # 更新状态
        self.state = np.random.rand(*self.observation_space.shape).astype(np.float32)

        self.current_batch += 1
        if self.current_batch * self.num_ue >= self.tot_num:
            self.done = all(self.ue_states)
        reward = -self.total_delay  # 使用总时延作为奖励

        print("======== STEP END ========")
        return self.state, reward, self.done, {}

    def render(self, mode="human"):
        print(f"Total Delay: {self.total_delay}, Batch: {self.current_batch}")

    def close(self):
        pass


# 示例测试环境
env = CustomEnv(debug=True)
obs = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break
