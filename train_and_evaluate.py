import json
from functools import reduce
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import time
import sys

from tqdm import tqdm

from sc3 import Map, NodeB, UE, TestModel

from positions import ap_stations
from ue_positions import points as ue_positions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def generate_uniform_points(num_points, max_x, max_y):
    # 计算网格大小
    grid_size = int(num_points ** 0.5)

    # 计算每个网格的宽度和高度
    cell_width = max_x / grid_size
    cell_height = max_y / grid_size

    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) < num_points:
                # 在每个网格内随机生成一个点
                x = random.uniform(i * cell_width, (i + 1) * cell_width)
                y = random.uniform(j * cell_height, (j + 1) * cell_height)
                points.append([x, y])

    # 如果生成的点不足num_points，在整个区域内随机添加剩余的点
    while len(points) < num_points:
        x = random.uniform(0, max_x)
        y = random.uniform(0, max_y)
        points.append([x, y])

    # 打乱点的顺序，使其看起来更随机
    random.shuffle(points)

    return points


concurrent_num = 5  # AP并发数
process_time = 2.3953080177

from torch.utils.tensorboard import SummaryWriter



class AP_state():
    def __init__(self):
        # self.occupied_time = -1
        self.line = []
        self.concurrenct_num = concurrent_num

    def join_line(self, index):
        self.line.append(index)

    def clear_line(self):
        self.line = []

    def able_to_join(self):
        return len(self.line) < self.concurrenct_num


class UE_state():
    def __init__(self):
        self.aka = False
        # self.in_rff = False
        self.conf = {}
        self.rff_ap = []
        self.finished = False

    def choose_aka(self):
        self.aka = True
        self.finished = True

    def add_p(self, p):
        conf, ap = p
        if ap not in self.conf:
            self.conf[ap] = conf
        # print('conf', self.conf)

    def add_rff_ap(self, ap):
        if ap not in self.rff_ap:
            self.rff_ap.append(ap)

    def get_total_conf(self):
        if self.aka:
            return 1
        if not self.rff_ap:
            return 0
        # 1 - (1 - p1)(1 - p2)...(1 - pn)
        conf_list = [self.conf[ap] for ap in self.rff_ap]
        total_conf = 1 - reduce(lambda x, y: x * (1 - y), conf_list, 1)
        return total_conf


class Environment():
    def __init__(self, num_ap, num_ue, ue_positions=None, from_file=False, once = 50):

        self.num_ue = num_ue
        self.num_ap = num_ap
        self.once = once

        ap_position = ap_stations
        ue_position = generate_uniform_points(num_ue, 200, 200)
        if ue_positions:
            ue_position = ue_positions

        self.gnb_lists = []
        for index, position in enumerate(ap_position):
            self.gnb_lists.append(NodeB(position[0], position[1], f"TestModel{index}"))
        # self.gnb_lists = list(map(lambda x: NodeB(x[0], x[1]), ap_position))

        self.ue_lists = []
        for index, position in enumerate(ue_position):
            self.ue_lists.append(UE(position[0], position[1], ue_type=index))

        self.ue_tot_states = list(map(lambda x: UE_state(), range(self.num_ue)))
        self.ap_states = list(map(lambda x: AP_state(), range(self.num_ap)))
        self.ue_states = self.ue_tot_states[:self.once]

        if from_file:
            self.import_p_info_file()

        # self.load_nearest_ap()

    def reset_points(self):
        self.ue_tot_states = self.ue_tot_states[self.once:]
        self.ue_states = self.ue_tot_states[:self.once]
    def get_p(self, scmap):
        node_json_list = [[i] * self.num_ue for i in range(self.num_ap)]
        aka_json = [0] * self.num_ue
        ue_idx = list(range(self.num_ue))

        for index in range(self.num_ap):
            _, p_dict = scmap.get_tot_time(node_json_list[index], aka_json, ue_idx)
            for i in range(self.num_ue):
                self.ue_tot_states[i].add_p(p_dict[i])
        #print([ue.conf for ue in self.ue_states])

    def get_state(self):
        p_info = []
        for i in range(len(self.ue_states)):
            for j in range(self.num_ap):
                p_info.append(self.ue_states[i].conf[j])
            if self.ue_states[i].finished:
                p_info.append(1)
            else:
                p_info.append(self.ue_states[i].get_total_conf())

        state = torch.tensor(p_info, dtype=torch.float32).to(device)

        return state

    def export_p_info(self):
        return [ue.conf for ue in self.ue_states]

    def export_p_info_file(self, filename='p_info.py'):
        p_info = [ue.conf for ue in self.ue_states]
        with open('p_info.py', 'w') as f:
            print('p_info = ', end='', file=f)
            print(p_info, file=f)

    def import_p_info(self, p_info):
        for index, ue in enumerate(self.ue_states):
            ue.conf = p_info[index]

    def import_p_info_file(self):
        import p_info_2 as p_info
        self.import_p_info(p_info.p_info)

    # def load_nearest_ap(self):
    #     current_nearest_ap = nearest_ap[self.num_ue]
    #     for index, ue in enumerate(self.ue_states):
    #         ue.nearest_ap = current_nearest_ap[index]


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc1.weight.data.normal_(0, 0.1)

        self.fc2 = nn.Linear(256, 256)
        self.fc2.weight.data.normal_(0, 0.1)

        self.fc3 = nn.Linear(256, output_size)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def remove_indices(original_list, indices_to_remove):
    # 删除列表中指定index的元素
    return [item for index, item in enumerate(original_list) if index not in indices_to_remove]


# Hyperparameters
alpha = 0.00005  # learning rate
gamma = 0.9  # discount factor
epsilon = 1.0  # exploration rate
epsilon_decay = 0.99995
epsilon_threshold = 0.2
transitions = 250  # training episodes
EP_MODE = True
buffer_capacity = 10000
batch_size = 64
target_update = 10  # update target network every 10 episodes
reward_scaling = 1e-2

if __name__ == '__main__':
    total_delays = []
    losses = []
    episode_rewards = []
    epsilons = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 命令行参数设置
    if len(sys.argv) < 4:
        print('Usage: python q_learning_v5.py num_ap num_ue threshold')
        print('Using Default Value AP*5 UE*10 threshold=0.5...')
        time.sleep(5)
        num_ap = 5
        num_ue = 50
        conf_threshold = 0.6
    else:
        num_ap = int(sys.argv[1])
        num_ue = int(sys.argv[2])
        conf_threshold = float(sys.argv[3])

    env = Environment(
        num_ap=num_ap,
        num_ue=num_ue,
        # ue_positions=ue_positions,
        # from_file='p_info.py'
    )
    scmap = Map(
        gnb_points=env.gnb_lists,
        ue_points=env.ue_lists,
        # model=TestModel(),
        aka_time=30
    )
    env.get_p(scmap)
    # env.import_p_info_file()

    # input_size = num_ue * (num_ap + 1) + num_ap
    input_size = num_ue * (num_ap + 1)
    output_size = num_ue * (num_ap + 1)

    policy_net = QNetwork(input_size, output_size).to(device)
    target_net = QNetwork(input_size, output_size).to(device)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs!")
    #     policy_net = nn.DataParallel(policy_net)
    #     target_net = nn.DataParallel(target_net)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
    criterion = nn.MSELoss()

    replay_buffer = ReplayBuffer(buffer_capacity)

    episode = 0
    total_delay = 0

    for transition in range(1000000):  # Large number to allow for episode-based termination

        # print(transition)
        start_time = time.time()
        for ap in env.ap_states:
            ap.clear_line()

        if all([ue.finished for ue in env.ue_states]):
            # 完成一个episode
            print('Finished ------------------------------------', transition, total_delay)

            episode += 1
            if EP_MODE and episode >= transitions:
                break

            temp_conf_list = env.export_p_info()
            env = Environment(num_ap=num_ap, num_ue=num_ue, ue_positions=ue_positions)
            env.import_p_info(temp_conf_list)

            total_delays.append(total_delay)


            epsilon = max(epsilon * epsilon_decay, epsilon_threshold)
            # epsilons.append(epsilon)

            total_delay = 0

            # Update target network
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        state = env.get_state()

        if random.uniform(0, 1) < epsilon:
            actions = [random.randint(0, num_ap) for _ in range(num_ue)]
        else:
            with torch.no_grad():
                act_tensor = policy_net(state).view(num_ue, num_ap + 1)
                actions = torch.argmax(act_tensor, dim=1).tolist()

        #print('actions', actions)

        # total_delay = 0
        rff_ue = []
        aka_ue = []

        this_transition_finish = []

        for i in range(num_ue):
            if env.ue_states[i].get_total_conf() > conf_threshold:
                env.ue_states[i].finished = True

            if env.ue_states[i].finished:
                continue

            if actions[i] == 0:
                env.ue_states[i].choose_aka()
                aka_ue.append(i)
                total_delay += 30

            else:
                # PLS-Au
                # print(f"UE {i} choose AP {actions[i] - 1}")
                ap = actions[i] - 1
                if env.ap_states[ap].able_to_join():
                    rff_ue.append({i: ap})
                    # env.ap_states[ap].occupy(transition)
                    env.ap_states[ap].join_line(i)
                    env.ue_states[i].add_rff_ap(ap)
                    total_delay += process_time
                else:
                    total_delay += process_time

        # node_json =
        # aka_json = [0]

        # tot, _ = scmap.get_tot_time(node_json, aka_json, ue_idx)

        # total_delay += tot
        # # print('tot', tot)

        # reward = -total_delay * 1e-1
        reward = -total_delay * reward_scaling
        # episode_rewards.append(reward)

        next_state = env.get_state()
        done = all([ue.finished for ue in env.ue_states])

        # Store transition in replay buffer
        replay_buffer.push(state, actions, reward, next_state, done)

        # Training
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.stack(states)
            actions = torch.tensor(actions, dtype=torch.long).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_states = torch.stack(next_states)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            # Reshape actions to match the output of the policy network
            actions = actions.view(batch_size, num_ue)

            # Compute current Q values
            current_q_values = policy_net(states).view(batch_size, num_ue, num_ap + 1)
            current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

            # Compute next Q values
            with torch.no_grad():
                next_q_values = target_net(next_states).view(batch_size, num_ue, num_ap + 1)
                max_next_q_values, _ = next_q_values.max(dim=2)
                target_q_values = rewards.unsqueeze(1) + gamma * max_next_q_values * (1 - dones.unsqueeze(1))

            # Compute loss
            loss = criterion(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        end_time = time.time()
        #print(end_time - start_time)
        # print('-----------')

    # env.export_p_info_file('p_info.py')


    # 可视化
    print('finish times when training:', len(total_delays))
    print('evaluate start')
    ave_delays = []
    for ue_tot in range(num_ue, 40000 + num_ue, num_ue):
        delays = 0

        for nothing in range(1000):  # 重复1000次取平均值

            env = Environment(
                num_ap=num_ap,
                num_ue=ue_tot,
                # ue_positions=ue_positions,
                # from_file='p_info.py'
            )
            scmap = Map(
                gnb_points=env.gnb_lists,
                ue_points=env.ue_lists,
                # model=TestModel(),
                aka_time=30
            )
            env.get_p(scmap)


            total_delay_pro = 0
            for j in range(ue_tot // num_ue):


                done = all([ue.finished for ue in env.ue_states])
                total_delay = 0
                times = 0

                while not done:
                    for ap in env.ap_states:
                        ap.clear_line()

                    state = env.get_state()
                    start_time = time.time()
                    if random.uniform(0, 1) < epsilon:
                        actions = [random.randint(0, num_ap) for _ in range(num_ue)]
                    else:
                        with torch.no_grad():
                            act_tensor = policy_net(state).view(num_ue, num_ap + 1)
                            actions = torch.argmax(act_tensor, dim=1).tolist()

                    rff_ue = []
                    aka_ue = []

                    this_transition_finish = []

                    for i in range(num_ue):
                        if env.ue_states[i].get_total_conf() > conf_threshold:
                            env.ue_states[i].finished = True

                        if env.ue_states[i].finished:
                            continue

                        if actions[i] == 0:
                            env.ue_states[i].choose_aka()
                            aka_ue.append(i)
                            total_delay += 30

                        else:
                            # PLS-Au
                            # print(f"UE {i} choose AP {actions[i] - 1}")
                            ap = actions[i] - 1
                            if env.ap_states[ap].able_to_join():
                                rff_ue.append({i: ap})
                                # env.ap_states[ap].occupy(transition)
                                env.ap_states[ap].join_line(i)
                                env.ue_states[i].add_rff_ap(ap)
                                total_delay += process_time
                            else:
                                total_delay += process_time
                    reward = -total_delay * reward_scaling
                    # episode_rewards.append(reward)

                    next_state = env.get_state()
                    done = all([ue.finished for ue in env.ue_states])

                    # Store transition in replay buffer
                    replay_buffer.push(state, actions, reward, next_state, done)

                    # Training
                    if len(replay_buffer) >= batch_size:
                        batch = replay_buffer.sample(batch_size)
                        states, actions, rewards, next_states, dones = zip(*batch)

                        states = torch.stack(states)
                        actions = torch.tensor(actions, dtype=torch.long).to(device)
                        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                        next_states = torch.stack(next_states)
                        dones = torch.tensor(dones, dtype=torch.float32).to(device)

                        # Reshape actions to match the output of the policy network
                        actions = actions.view(batch_size, num_ue)

                        # Compute current Q values
                        current_q_values = policy_net(states).view(batch_size, num_ue, num_ap + 1)
                        current_q_values = current_q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

                        # Compute next Q values
                        with torch.no_grad():
                            next_q_values = target_net(next_states).view(batch_size, num_ue, num_ap + 1)
                            max_next_q_values, _ = next_q_values.max(dim=2)
                            target_q_values = rewards.unsqueeze(1) + gamma * max_next_q_values * (
                                        1 - dones.unsqueeze(1))

                        # Compute loss
                        loss = criterion(current_q_values, target_q_values)
                        losses.append(loss.item())

                        # Optimize the model
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    end_time = time.time()
                    print(end_time - start_time)
                if (ue_tot // num_ue) > 1:
                    env.reset_points()
                total_delay_pro += total_delay
            delays += total_delay_pro
        ave_delay = delays / 1000
        ave_delays.append(delays)
        print("ue_numbers: " + str(ue_tot))
        print("ave_delay: " + str(ave_delay))

    with open("output.txt", "w") as f:
        f.writelines(json.dumps(ave_delays))




    torch.save(policy_net.state_dict(), "model_weights_v8_AP{}_UE{}_v2.pth".format(num_ap, num_ue))

    plt.plot(total_delays)
    plt.xlabel('Episode')
    plt.ylabel(f'Total Delay when {num_ue} UE')
    current_time = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'./plt/total_delay_{num_ue}_{conf_threshold}_{current_time}.png')

