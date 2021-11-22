"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from env import env as path_planning_env
from torchvision.models import resnet18

os.environ['CUDA_VISIABLE_DEVICE'] = '7'

# Hyper Parameters
BATCH_SIZE = 128
LR = 1e-5  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.7  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
GREEDY_READY_EPI = 100

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
        self.conv1 = nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=2304, out_features=512, bias=True)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=128, bias=True)
        self.relu5 = nn.Sigmoid()
        self.fc3 = nn.Linear(in_features=128, out_features=28, bias=True)

    def forward(self, x):
        resnet_out = self.resnet(x).reshape(-1, 7, 4)
        action = torch.softmax(resnet_out, -1)
        return action





class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 7 + N_ACTIONS))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, is_test=False, episode=GREEDY_READY_EPI):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).cuda()
        # input only one sample
        episode = GREEDY_READY_EPI if episode > GREEDY_READY_EPI else episode
        epi = EPSILON * (episode / GREEDY_READY_EPI) ** 2
        if np.random.uniform() < epi or is_test:  # greedy
            action_74 = self.eval_net.forward(x)
            action = torch.max(action_74, dim=2)[1].cpu().numpy()
            # action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(low=0, high=4, size=(N_ACTIONS,))
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s.reshape(-1), a.reshape(-1), r, s_.reshape(-1)))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        self.eval_net.train()
        print('learn')
        # target parameter update
        # if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        soft_update(self.target_net, self.eval_net, 0.04)
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES].reshape(-1, 8, 108, 192)).cuda()
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + N_ACTIONS].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS:N_STATES + N_ACTIONS + 7]).cuda()
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:].reshape(-1, 8, 108, 192)).cuda()

        b_a_7 = b_a.reshape(-1, 7)
        b_a__1 = b_a_7.reshape(-1, 1)  # 转成一个长的0-3的向量

        # q_eval w.r.t the action in experience
        q_eval_b_28 = self.eval_net(b_s)  # shape (batch, 1)
        q_eval_b__4 = q_eval_b_28.reshape(-1, 4)
        q_eval_b__1 = q_eval_b__4.gather(1, b_a__1)
        q_eval = q_eval_b__1.reshape(-1, 7)

        q_next = self.target_net(b_s_).detach().reshape(-1, 7, 4)  # detach from graph, don't backpropagate
        q_next_max = torch.max(q_next, dim=2)[0]
        q_target = b_r + GAMMA * q_next_max  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, name=''):
        torch.save(self.eval_net, F'model/eval_net_{name}.pkl')
        torch.save(self.target_net, F'model/target_net_{name}.pkl')

    def restore_net(self, name=''):
        self.eval_net = torch.load(F'model/eval_net_{name}.pkl')
        self.target_net = torch.load(F'model/target_net_{name}.pkl')


def test():
    dqn = DQN()
    dqn.restore_net('900')
    s = env.reset()
    for i_step in range(50):
        env.render()
        a = dqn.choose_action(s, is_test=True)

        # take action
        s_, r, done, info = env.step(a)

        # dqn.store_transition(s, a, r, s_)


def train():
    print('start')
    dqn = DQN()
    # dqn.learn()
    print('\nCollecting experience....')
    for i_episode in range(4000000):
        s = env.reset()
        ep_r = 0
        if i_episode % 100 == 0 and i_episode > 0:
            dqn.save(name=str(i_episode))

        for i_step in range(100):
            # if i_episode > 10:
            #     env.render()
            a = dqn.choose_action(s, episode=i_episode)
            # take action
            s_, r, done, info = env.step(a)

            dqn.store_transition(s, a, r, s_)

            ep_r += np.sum(r)

            if done or True:
                print(F'Ep: {i_episode} step:{i_step} avg_r:{ep_r / (i_step + 1)}')

            if done:
                break
            s = s_
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()


if __name__ == '__main__':
    # train()
    test()
    # action_net = Net()
    # x = torch.randn(2, 8, 108, 192)
    # action = action_net(x)
    # print(action)
    # asdf1111
