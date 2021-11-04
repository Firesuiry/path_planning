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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env import env as path_planning_env

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
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
        # self.fc1 = nn.Linear(N_STATES, 50)
        # self.fc1.weight.data.normal_(0, 0.1)  # initialization
        # self.out = nn.Linear(50, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)  # initialization

        self.net = nn.Sequential(
            # nn.MaxPool2d(10, 10),
            # input 192*108*8
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 96*54*16
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 48*27*16
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 16*9*8

        )
        self.net2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2496, 56),
            nn.Sigmoid(),
            nn.Linear(56, 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.net(x)
        action = self.net2(conv_out)
        return action


def soft_update(target, source, x):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - x) * target_param.data + x * source_param.data)


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 1 + N_ACTIONS))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, is_test=False):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON or is_test:  # greedy
            action = self.eval_net.forward(x).detach().numpy()
            # action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.random(N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s.reshape(-1), a.reshape(-1), [r], s_.reshape(-1)))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        # if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        soft_update(self.target_net, self.eval_net, 0.01)
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES].reshape(-1, 8, 108, 192))
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + N_ACTIONS].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES + N_ACTIONS:N_STATES + N_ACTIONS + 1])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:].reshape(-1, 8, 108, 192))

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
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
    dqn.restore_net('3950')
    s = env.reset()
    for i_step in range(50):
        env.render()
        a = dqn.choose_action(s, is_test=True)

        # take action
        s_, r, done, info = env.step(a)

        dqn.store_transition(s, a, r, s_)


def train():
    dqn = DQN()
    print('\nCollecting experience...')
    for i_episode in range(4000):
        s = env.reset()
        ep_r = 0
        if i_episode % 50 == 0:
            dqn.save(name=str(i_episode))

        for i_step in range(50):
            # if i_episode > 10:
            #     env.render()
            a = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            dqn.store_transition(s, a, r, s_)

            ep_r += r

            if done or True:
                print(F'Ep: {i_episode} step:{i_step}')

            if done:
                break
            s = s_
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()


if __name__ == '__main__':
    # train()
    test()