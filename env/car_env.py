import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure

## 地图大小
from env.EnvBase import Env

map_height = 1080
map_width = 1920

## 障碍物中心位置
obstacle_center = [[200, 400], [600, 700], [600, 1200], [600, 1500], [800, 400], [200, 700], [340, 960]]

## 障碍物的边长
obstacle_height = 150
obstacle_width = 150

## 小车的起点
car_init = [[800, 50], [200, 50], [300, 50], [400, 50], [500, 50], [600, 50], [700, 50]]

## 小车的边长
car_height = 60
car_width = 90

## 终点线
end_line = 1800


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


class env(Env):
    n_action = 7
    n_action_shape = (7,)
    # n_state_shape = (8, map_width, map_height)
    n_state_shape = (36, 64, 8)
    n_state = n_state_shape[0] * n_state_shape[1] * n_state_shape[2]

    def __init__(self, obs_shape=n_state_shape, action_shape=(28,), action_low=-1, action_high=1, show=False):
        super(env, self).__init__(obs_shape, action_shape, action_low, action_high)

        obstacle_center_init = obstacle_center

        car_center_init = car_init
        self.step_num = 0

        ## 状态
        self.obstacle_center_init = obstacle_center_init
        self.car_center_init = car_center_init
        self.state_space = init_space(obstacle_center_init, car_center_init)

        ## 障碍物中心点位置
        self.obstacle_center = self.obstacle_center_init

        ## 小车中心点位置
        self.car_center = self.car_center_init

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

    def get_state(self):
        after_maxpool = skimage.measure.block_reduce(self.state_space, (1, 30, 30), np.max)
        # state = np.swapaxes(after_maxpool, 2, 0)
        state = np.transpose(after_maxpool, (1, 2, 0))
        return state

    def get_render_state(self):
        after_maxpool = skimage.measure.block_reduce(self.state_space, (1, 30, 30), np.max)
        return after_maxpool

    def reset(self):
        print('env reset')
        print('小车位置', self.car_center, self.car_center_init)
        self.state_space = init_space(self.obstacle_center_init, self.car_center_init)

        self.obstacle_center = self.obstacle_center_init

        self.car_center = copy.deepcopy(self.car_center_init)

        # print('小车位置', self.car_center, self.car_center_init)

        self.reward = np.zeros(7)

        self.reward_sum = np.zeros(7)

        self.reward_system = 0

        self.step_num = 0

        return self.get_state()

    def is_stop(self):
        ok_car = np.where(np.array(self.car_center)[:, 1] > end_line)[0]
        ok_car = ok_car.shape[0]
        if ok_car == 7:
            return True
        else:
            return False

    def step(self, action_28: np.ndarray):
        self.step_num += 1
        # 初始化小车单步奖励
        if type(action_28) != np.ndarray:
            action_28 = action_28.numpy()
        action_74 = action_28.reshape(7, 4)
        action_74_ex = np.exp(action_74)
        probs_74 = action_74_ex / np.sum(action_74_ex, axis=-1, keepdims=True)
        action_7 = np.zeros(7)
        for i in range(7):
            action_7[i] = np.random.choice(4, p=probs_74[i])

        reward = np.zeros(7)

        # print(f'执行的action：{action_28.reshape(7, 4)}')
        for i, act_class in enumerate(list(action_7)):
            new_car_center = copy.deepcopy(self.car_center[i])
            if act_class == 0:  # 不动
                pass
            elif act_class == 1:  # 动作类别1表示前进
                new_car_center[1] = min(self.car_center[i][1] + car_width, map_width - 50)
            elif act_class == 2:  # 动作类别2表示向上移动
                new_car_center[0] = max(self.car_center[i][0] - car_height, 50)
            else:  # 动作类别3表示向下移动
                new_car_center[0] = min(self.car_center[i][0] + car_height, map_height - 50)

            if is_knock(new_car_center, self.car_center, self.obstacle_center, car_id=i):
                # print(F'car id:{i} 发生碰撞')
                reward[i] = -1
            else:
                if act_class == 0:  # 不动
                    reward[i] = -0.1
                elif act_class == 1:  # 前进
                    ## 更新状态图
                    reward[i] = 1
                    local = np.where(self.state_space[i, ...] != 0)

                    self.state_space[i, ...] = 0
                    local = (local[0], np.minimum(local[1] + car_width, map_width - 50))
                    self.state_space[i, ...][local] = 1

                    # 更新小车中心位置
                    self.car_center[i][1] = min(self.car_center[i][1] + car_width, map_width - 50)

                elif act_class == 2:  # 向上
                    reward[i] = -0.1
                    # 更新状态图
                    local = np.where(self.state_space[i, ...] != 0)
                    self.state_space[i, ...] = 0
                    local = (np.maximum(50, local[0] - car_height), local[1])
                    self.state_space[i, ...][local] = 1

                    # 更新小车中心位置
                    self.car_center[i][0] = max(self.car_center[i][0] - car_height, 50)
                else:  # 向下
                    # 更新状态图
                    reward[i] = -0.1
                    local = np.where(self.state_space[i, ...] != 0)
                    self.state_space[i, ...] = 0
                    local = (np.minimum(map_height - 50, local[0] + car_height), local[1])
                    self.state_space[i, ...][local] = 1

                    # 更新小车中心位置
                    self.car_center[i][0] = min(self.car_center[i][0] + car_height, map_height - 50)

        # print(self.car_center)
        done = self.is_stop()
        reward = np.sum(reward)
        if done:
            reward += 10

        state = self.get_state()
        if self.step_num > 100:
            done = True
        if self.show:
            print(F'reward: {reward} done:{done}')
        return state, reward, done, None

    def show_state(self):

        # for i in range(9):
        #
        #     if i < 8:
        #         ax[i // 3, i % 3].imshow(self.state_space[i, ...], cmap='gray')
        #     else:
        #         ax[i // 3, i % 3].imshow(np.sum(self.state_space, axis=0), cmap='gray')

        plt.imshow(np.sum(self.get_render_state(), axis=0), cmap='gray')

        plt.pause(0.1)

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(1, 1)
        self.show_state()


if __name__ == '__main__':

    env1 = env(obstacle_center, car_init)
    # plt.figure()

    for i in range(100):
        s = env1.reset()

        print('-' * 50)
        for step in range(100):
            action = np.random.random(28)
            # plt.imshow(np.sum(env1.get_state(),axis=0), cmap='gray')
            # plt.pause(10)

            env1.is_stop()
            env1.step(action)
            env1.render()
            # env1.show_state()
