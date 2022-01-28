import copy
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import logging

## 地图大小
from env.EnvBase import Env

map_height = 36
map_width = 64

## 障碍物中心位置
obstacle_left_top = [[7, 13], [20, 23], [20, 40], [20, 50], [27, 13], [7, 23], [11, 32]]

## 障碍物的边长
obstacle_height = 5
obstacle_width = 5

## 小车的起点
car_init = [[3, 2], [6, 2], [9, 2], [12, 2], [15, 2], [18, 2], [21, 2]]

## 小车的边长
car_height = 2
car_width = 3

## 终点线
end_line = 60

ACTION_NAME = ['stop', 'go', 'up', 'down']


def is_knock(center, car_center, obstacle_center, car_id):
    ## 小车边框
    b1_x1, b1_y1, b1_x2, b1_y2 = center[0] - car_height // 2, center[1] - car_width // 2, \
                                 center[0] + car_height // 2, center[1] + car_width // 2

    ## 判断是否会和其他小车碰撞
    for i, other_car_center in enumerate(car_center):
        if i == car_id:
            pass
        else:
            ## 其他小车边框
            b2_x1, b2_y1, b2_x2, b2_y2 = other_car_center[0] - car_height // 2, other_car_center[1] - car_width // 2, \
                                         other_car_center[0] + car_height // 2, other_car_center[1] + car_width // 2

            inter_rect_x1 = max(b1_x1, b2_x1)
            inter_rect_y1 = max(b1_y1, b2_y1)
            inter_rect_x2 = min(b1_x2, b2_x2)
            inter_rect_y2 = min(b1_y2, b2_y2)

            inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                         max(inter_rect_y2 - inter_rect_y1 + 1, 0)

            b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
            b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

            iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

            if iou > 0:
                return True

    ## 判断是否会和障碍物碰撞
    for other_obstacle_center in obstacle_center:

        ## 障碍物边框
        b2_x1, b2_y1, b2_x2, b2_y2 = other_obstacle_center[0] - obstacle_height // 2, other_obstacle_center[
            1] - obstacle_width // 2, \
                                     other_obstacle_center[0] + obstacle_height // 2, other_obstacle_center[
                                         1] + obstacle_width // 2

        inter_rect_x1 = max(b1_x1, b2_x1)
        inter_rect_y1 = max(b1_y1, b2_y1)
        inter_rect_x2 = min(b1_x2, b2_x2)
        inter_rect_y2 = min(b1_y2, b2_y2)

        inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, 0) * \
                     max(inter_rect_y2 - inter_rect_y1 + 1, 0)

        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        if iou > 0:
            return True

    return False


def plot_obstacle(map, obstacle_center):
    for center_x, center_y in obstacle_center:
        cv2.rectangle(map, (center_y - obstacle_width // 2, center_x - obstacle_height // 2),
                      (center_y + obstacle_width // 2, center_x + obstacle_height // 2), (255, 255, 255), -1)
    return map


def init_space(obstacle_center, car_init):
    space = np.zeros((8, map_height, map_width))
    space[-1, ...] = plot_obstacle(space[-1, ...], obstacle_center)
    for i, (center_x, center_y) in enumerate(car_init):
        cv2.rectangle(space[i, ...], (center_y - car_width // 2, center_x - car_height // 2),
                      (center_y + car_width // 2, center_x + car_height // 2), (255, 255, 255), -1)
    space[np.where(space != 0)] = 1
    return space


def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m" % '   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent * 100) + end_str
    print(bar, end='', flush=True)


class env(Env):
    n_action = 7
    n_action_shape = (7,)
    # n_state_shape = (8, map_width, map_height)
    n_state_shape = (36, 64, 8)
    n_state = n_state_shape[0] * n_state_shape[1] * n_state_shape[2]

    def __init__(self, obs_shape=n_state_shape, action_shape=(28,), action_low=-1, action_high=1, show=False,
                 chw=True):
        super(env, self).__init__(obs_shape, action_shape, action_low, action_high)

        obstacle_left_top_init = obstacle_left_top
        car_left_top_init = car_init
        self.step_num = 0

        ## 状态
        self.state_space = np.zeros(self.n_state_shape)
        self.obstacle_left_top_init = obstacle_left_top_init
        self.car_left_top_init = car_left_top_init
        # self.state_space = init_space(obstacle_left_top_init, car_left_top_init)

        ## 障碍物left top点位置
        self.obstacle_left_top = copy.deepcopy(obstacle_left_top_init)

        ## 小车left top点位置
        self.car_left_top = copy.deepcopy(car_left_top_init)

        ## 小车单步奖励
        self.reward = np.zeros(7)

        ## 小车累计奖励
        self.reward_sum = np.zeros(7)

        ## 系统奖励
        self.reward_system = 0

        # 展示相关
        self.fig = None
        self.ax = None
        self.show = show
        self.chw = chw

    def get_state(self, hwc=False):
        self.state_space[:] = 0
        for i in range(7):
            self.state_space[self.car_left_top[i][0]:self.car_left_top[i][0] + car_height,
            self.car_left_top[i][1]:self.car_left_top[i][1] + car_width, i] = 1

        for i in range(len(self.obstacle_left_top)):
            self.state_space[self.obstacle_left_top[i][0]:self.obstacle_left_top[i][0] + obstacle_height,
            self.obstacle_left_top[i][1]:self.obstacle_left_top[i][1] + obstacle_width, 7] = 1
        if self.chw and not hwc:
            return self.state_space.transpose((2, 0, 1))
        return self.state_space

    def get_render_state(self):
        return self.get_state(hwc=True)

    def reset(self):
        if self.show:
            print('env reset')
        # print('小车位置', self.car_left_top, self.car_center_init)
        ## 障碍物left top点位置
        self.obstacle_left_top = copy.deepcopy(self.obstacle_left_top_init)

        ## 小车left top点位置
        self.car_left_top = copy.deepcopy(self.car_left_top_init)

        # print('小车位置', self.car_center, self.car_center_init)

        self.reward = np.zeros(7)

        self.reward_sum = np.zeros(7)

        self.reward_system = 0

        self.step_num = 0

        return self.get_state()

    def is_stop(self):
        ok_car = np.where(np.array(self.car_left_top)[:, 1] >= end_line)[0]
        ok_car = ok_car.shape[0]
        if ok_car == 7:
            return True
        else:
            return False

    def knock_check(self, new_car_pos, car_id):
        # car knock check
        for i in range(len(self.car_left_top)):
            if i == car_id:
                continue
            if abs(self.car_left_top[i][0] - new_car_pos[0]) < car_height:
                if abs(self.car_left_top[i][1] - new_car_pos[1]) < car_width:
                    logging.debug(f'{car_id} {new_car_pos} knock with car {i} {self.car_left_top[i]}')
                    return True
        # obstacle knock check
        for i in range(len(self.obstacle_left_top)):
            obs_left_top = self.obstacle_left_top[i]

            if new_car_pos[0] < obs_left_top[0] + obstacle_height and \
                    new_car_pos[1] < obs_left_top[1] + obstacle_width and \
                    obs_left_top[0] < new_car_pos[0] + car_height and \
                    obs_left_top[1] < new_car_pos[1] + car_width:
                logging.debug(f'{car_id} {new_car_pos} knock with obstacle {i} {self.obstacle_left_top[i]}')
                return True
        return False

    def step(self, action_28: np.ndarray):
        self.step_num += 1
        # 初始化小车单步奖励

        if type(action_28) != np.ndarray:
            action_28 = action_28.numpy()
        if action_28.shape[0] == 28:
            action_74 = action_28.reshape(7, 4)
            action_74_ex = np.exp(action_74)
            probs_74 = action_74_ex / np.sum(action_74_ex, axis=-1, keepdims=True)
            # print(action_74)
            # print(np.mean(probs_74), np.std(probs_74))
            # print(f'probs   average:{np.mean(probs_74)} std:{np.std(probs_74)}' )
            # print(f'actions average:{np.mean(action_74)} std:{np.std(action_74)}')
            action_7 = np.zeros(7)
            for i in range(7):
                action_7[i] = np.random.choice(4, p=probs_74[i])
            # action_7 = np.argmax(action_74, axis=1)
            # print([ACTION_NAME[i] for i in action_7])
        elif action_28.shape[0] == 7:
            action_7 = action_28
        else:
            raise ValueError(F'action shape error shape:{action_28.shape}')
        if self.show:
            print([ACTION_NAME[i] for i in action_7])
        reward = np.zeros(7)

        # print(f'执行的action：{action_28.reshape(7, 4)}')
        for i, act_class in enumerate(list(action_7)):
            if self.car_left_top[i][1] >= end_line and act_class == 1:
                continue
            new_car_center = copy.deepcopy(self.car_left_top[i])
            if act_class == 0:  # 不动
                pass
            elif act_class == 1:  # 动作类别1表示前进
                new_car_center[1] = min(self.car_left_top[i][1] + car_width, map_width - car_width)
            elif act_class == 2:  # 动作类别2表示向上移动
                new_car_center[0] = max(self.car_left_top[i][0] - car_height, 0)
            else:  # 动作类别3表示向下移动
                new_car_center[0] = min(self.car_left_top[i][0] + car_height, map_height - car_height)

            if act_class != 0 and self.knock_check(new_car_center, car_id=i):
                # print(F'car id:{i} 发生碰撞')
                reward[i] = -2
            else:
                if act_class == 0:  # 不动
                    reward[i] = -0.2
                elif act_class == 1:  # 前进
                    ## 更新状态图
                    reward[i] = 0.3
                elif act_class == 2:  # 向上
                    reward[i] = -0.1
                else:  # 向下
                    # 更新状态图
                    reward[i] = -0.1
                if self.car_left_top[i][1] < end_line <= new_car_center[1]:
                    reward[i] += 10
                self.car_left_top[i] = new_car_center

        # print(self.car_center)
        done = self.is_stop()
        # reward = np.sum(reward)

        state = self.get_state()
        if self.step_num > 1000:
            done = True
        if self.show:
            print(F'reward: {reward} done:{done}')
        # print([ACTION_NAME[i] for i in action_7])
        # print(reward)

        # process_bar(self.step_num/150, start_str='', end_str='100%', total_length=15)
        # sys.stdout.write("== " + str(self.step_num/150) + "%/100%")
        return state, reward, done, None

    def show_state(self):

        # for i in range(9):
        #
        #     if i < 8:
        #         ax[i // 3, i % 3].imshow(self.state_space[i, ...], cmap='gray')
        #     else:
        #         ax[i // 3, i % 3].imshow(np.sum(self.state_space, axis=0), cmap='gray')
        show_data = np.zeros((36, 64, 3), dtype=np.uint8)
        state = self.get_render_state()
        show_data[:, :, 0] = (state[:, :, 7] > 0) * 1
        show_data[:, :, 1] = (np.sum(state[:, :, 0:7], axis=2) > 0) * 1
        show_data[:, :, 2] = 1
        # plt.imshow(np.sum(self.get_render_state(), axis=2), cmap='gray')
        plt.imshow(show_data * 255)

        plt.pause(0.1)

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
        self.show_state()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env1 = env(obstacle_left_top, car_init)
    # plt.figure()

    for i in range(100):
        s = env1.reset()
        print('-' * 50)
        for step in range(100):
            action = np.random.random(28)
            # print('step')
            # plt.imshow(np.sum(env1.get_state(),axis=0), cmap='gray')
            # plt.pause(10)

            # env1.is_stop()
            env1.step(action)
            env1.render()
            # env1.show_state()
