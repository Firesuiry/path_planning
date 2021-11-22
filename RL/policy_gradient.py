from RL.base_agent import BaseRlAgent
from net import ConvFCNet
import numpy as np
import torch
from torch import nn

# Hyper Parameters
from util import soft_update

BATCH_SIZE = 128
LR = 1e-5  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.7  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
GREEDY_READY_EPI = 100


class PolicyGradient(BaseRlAgent):
    def __init__(self, n_states, n_actions, env_a_shape):
        self.n_states = n_states
        self.n_actions = n_actions
        self.env_a_shape = env_a_shape

        self.eval_net, self.target_net = ConvFCNet().cuda(), ConvFCNet().cuda()
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, n_states * 2 + 7 + n_actions))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, is_test=False, episode=GREEDY_READY_EPI):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).cuda()
        # input only one sample
        episode = GREEDY_READY_EPI if episode > GREEDY_READY_EPI else episode
        epi = EPSILON * (episode / GREEDY_READY_EPI) ** 2
        if is_test:
            action_74 = self.eval_net.forward(x)
            action = torch.max(action_74, dim=2)[1].cpu().numpy()
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  # return the argmax index
        elif np.random.uniform() < epi:  # greedy
            action_74 = self.eval_net.forward(x)
            action_74_cpu = action_74.cpu().numpy()
            action = np.zeros(7)
            for i in range(7):
                action[i] = np.random.choice(4, p=action_74_cpu[i])
            action = action[0] if self.env_a_shape == 0 else action.reshape(self.env_a_shape)  # return the argmax index
        else:  # random
            action = np.random.randint(low=0, high=4, size=(self.n_actions,))
            action = action if self.env_a_shape == 0 else action.reshape(self.env_a_shape)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s.reshape(-1), a.reshape(-1), r, s_.reshape(-1)))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def generate_transition(self):
        index = self.memory_counter - 1
        while index > 0:
            r = self.memory[index][self.n_states + self.n_actions:self.n_states + self.n_actions + 7]
            index -= 1
            self.memory[index][self.n_states + self.n_actions:self.n_states + self.n_actions + 7] += GAMMA * r
        self.memory[:, self.n_states + self.n_actions:self.n_states + self.n_actions + 7] -= \
            np.mean(self.memory[:, self.n_states + self.n_actions:self.n_states + self.n_actions + 7], axis=0)
        self.memory[:, self.n_states + self.n_actions:self.n_states + self.n_actions + 7] /= \
            (np.std(self.memory[:, self.n_states + self.n_actions:self.n_states + self.n_actions + 7], axis=0) + 1e-4)
        # 使均值为0 方差为1

    def learn(self):
        self.eval_net.train()
        self.generate_transition()
        print('learn')
        # target parameter update
        # if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        # soft_update(self.target_net, self.eval_net, 0.04)
        self.learn_step_counter += 1

        # sample batch transitions
        # sample_index = np.arange(self.memory_counter)
        b_memory = self.memory
        b_s = torch.FloatTensor(b_memory[:, :self.n_states].reshape(-1, 8, 108, 192)).cuda()
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + self.n_actions].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, self.n_states + self.n_actions:self.n_states + self.n_actions + 7]).cuda()
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:].reshape(-1, 8, 108, 192)).cuda()

        '''
        s b 8, 108, 192
        a b 7 1
        r b 7 1
        '''

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
