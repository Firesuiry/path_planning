from RL.train0.ddpg import train
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    train()