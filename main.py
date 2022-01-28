"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import json

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from matplotlib import pyplot as plt

from env.car_env2 import env as path_planning_env
from torchvision.models import resnet18

from util import soft_update

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Hyper Parameters
BATCH_SIZE = 1024
LR = 1e-4  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 10000
GREEDY_READY_EPI = 150


if False:
    env0 = gym.make('CartPole-v0')
    env0 = env0.unwrapped
    N_ACTIONS = env0.action_space.n
    N_STATES = env0.observation_space.shape[0]
    ENV_A_SHAPE = 0 if isinstance(env0.action_space.sample(),
                                  int) else env0.action_space.sample().shape  # to confirm the shape
    env = env0
else:
    env1 = path_planning_env()
    N_ACTIONS = env1.n_action
    N_STATES = env1.n_state
    ENV_A_SHAPE = env1.n_action_shape
    env = env1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(in_features=18432, out_features=200),
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=18432, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=200),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=200),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=200, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=28),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight.data, a=10, mode='fan_in',)  # normal: mean=0, std=1
                nn.init.normal_(m.weight.data, std=0.09)  # normal: mean=0, std=1
            elif isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight.data, std=0.09)  # normal: mean=0, std=1


    def forward(self, x):
        conv_out = self.conv1(x)
        fc_out = self.fc(conv_out)
        res_out = fc_out.reshape(-1, 7, 4)
        # action = torch.softmax(res_out, -1)
        return res_out


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 7 + N_ACTIONS))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        soft_update(self.target_net, self.eval_net, 1.)

    def choose_action(self, x, is_test=False, episode=GREEDY_READY_EPI):
        # show_state(x)
        # where_x = np.where(x>0)
        # where_x1 = where_x[1]
        # where_x2 = where_x[2]
        s = torch.unsqueeze(torch.FloatTensor(x), 0).cuda()
        # input only one sample
        episode = GREEDY_READY_EPI if episode > GREEDY_READY_EPI else episode
        epi = EPSILON * (episode / GREEDY_READY_EPI)
        if np.random.uniform() < epi or is_test:  # greedy
            action_74 = self.eval_net.forward(s)
            action = torch.max(action_74, dim=2)[1].cpu().numpy()
            # action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(low=0, high=4, size=(N_ACTIONS,))
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s.reshape(-1), a.reshape(-1), r, s_.reshape(-1)))
        # show_state(transition[:N_STATES].reshape((8,36,64)))
        # show_state(s)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        self.eval_net.train()
        # print('learn')
        # target parameter update
        # if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        soft_update(self.target_net, self.eval_net, 0.1)
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES].reshape(-1, 8, 36, 64)).cuda()
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + N_ACTIONS].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS:N_STATES + N_ACTIONS + 7]).cuda()
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:].reshape(-1, 8, 36, 64)).cuda()

        b_a_7 = b_a.reshape(-1, 7)
        b_a__1 = b_a_7.reshape(-1, 1)  # 转成一个长的0-3的向量

        # q_eval w.r.t the action in experience
        q_eval_b_28 = self.eval_net(b_s)  # shape (batch, 1)
        q_eval_b__4 = q_eval_b_28.reshape(-1, 4)
        q_eval_b__1 = q_eval_b__4.gather(1, b_a__1)
        q_eval = q_eval_b__1.reshape(-1, 7)
        # print(q_eval, b_r)
        q_next = self.target_net(b_s_).detach().reshape(-1, 7, 4)  # detach from graph, don't backpropagate
        # q_next_test = self.eval_net(b_s_).detach().reshape(-1, 7, 4)  # detach from graph, don't backpropagate
        q_next_max = torch.max(q_next, dim=2)[0]
        q_target = b_r + GAMMA * q_next_max  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # plt.switch_backend('agg')
        # show_state(b_s.cpu().numpy()[0])
        # plt.switch_backend('agg')
        # show_state(b_s_.cpu().numpy()[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, name=''):
        torch.save(self.eval_net, F'model/eval_net_{name}.pkl')
        torch.save(self.target_net, F'model/target_net_{name}.pkl')

    def restore_net(self, name=''):
        self.eval_net = torch.load(F'model/eval_net_{name}.pkl')
        self.target_net = torch.load(F'model/target_net_{name}.pkl')




def test(iter_num):
    dqn = DQN()
    if iter_num:
        print(F'restore_net {iter_num}')
        dqn.restore_net(str(iter_num))
    env = path_planning_env(show=True)
    s = env.reset()
    for i_step in range(50):
        env.render()
        a = dqn.choose_action(s, is_test=True)
        # action = np.random.randint(low=0, high=4, size=(N_ACTIONS,))
        # a = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        # take action
        s_, r, done, info = env.step(a)

        # dqn.store_transition(s, a, r, s_)


def random_test():
    net = Net().cuda()
    net = torch.load(F'model/eval_net_{1000}.pkl')

    for i in range(10):
        input_tensor = torch.rand((1, 8, 36, 64)).cuda()
        test_out = net.forward(input_tensor)
        # print(input_tensor)
        print(test_out)


def show_state(state):
    state = np.transpose(state, (1, 2, 0))
    # shape = state.shape
    # print(shape)
    # state = state.reshape(-1).reshape((36,64,-1))
    show_data = np.zeros((36, 64, 3), dtype=np.uint8)

    show_data[:, :, 0] = state[:, :, 7]
    show_data[:, :, 1] = np.sum(state[:, :, 0:7], axis=2)
    show_data[:, :, 2] = 1
    # plt.imshow(np.sum(self.get_render_state(), axis=2), cmap='gray')
    plt.imshow(show_data)

    # plt.pause(0.1)


def train(save_freq=1000):
    print('start')
    dqn = DQN()
    # dqn.restore_net(str(287000))
    # dqn.learn()
    print('\nCollecting experience....')
    ep_rs = []
    for i_episode in range(4000000):
        s = env.reset()
        ep_r = 0
        if i_episode % save_freq == 0:
            dqn.save(name=str(i_episode))
            with open('ep_rs.json', 'w') as f:
                f.write(json.dumps(ep_rs))

        for i_step in range(40000):
            # if i_episode > 10:
            #     env.render()
            a = dqn.choose_action(s, episode=i_episode)
            # take action
            s_, r, done, info = env.step(a)

            dqn.store_transition(s, a, r, s_)

            ep_r += np.sum(r)

            if done:
                if dqn.memory_counter > MEMORY_CAPACITY:
                    print(F'Ep: {i_episode} step:{i_step} avg_r:{ep_r / (i_step + 1)} learn')
                else:
                    print(F'Ep: {i_episode} step:{i_step} avg_r:{ep_r / (i_step + 1)}')
                ep_rs.append(ep_r)

            if done:
                break
            s = s_
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()


if __name__ == '__main__':
    # train()
    # test('')
    test(68000)
    # test(330000)
    # test(20)
    # test(207000)
    # random_test()
