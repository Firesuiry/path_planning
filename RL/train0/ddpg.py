from env.car_env import env
from RL.DDPG.TF2_DDPG_Basic import DDPG
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def train():
    gym_env = env(show=False)
    try:
        # Ensure action bound is symmetric
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')
    ddpg = DDPG(gym_env, discrete=is_discrete, memory_cap=1000000, gamma=0, sigma=0.25, actor_units=(16,),
                critic_units=(8, 16, 32), use_priority=False, lr_critic=1e-8, lr_actor=1e-10)
    ddpg.train(max_episodes=10000, max_steps=10000)


def test():
    gym_env = env(show=False)
    try:
        # Ensure action bound is symmetric
        assert (gym_env.action_space.high == -gym_env.action_space.low)
        is_discrete = False
        print('Continuous Action Space')
    except AttributeError:
        is_discrete = True
        print('Discrete Action Space')

    ddpg = DDPG(gym_env, discrete=is_discrete)
    # ddpg.load_critic("ddpg_critic_episode124.h5")
    # ddpg.load_actor("actor01.h5")

    step_nums = []
    for i in range(10):
        reward, step_num = ddpg.test()
        step_nums.append(step_num)
        print('step:{}'.format(step_num))
    print('trained mean:{}'.format(np.mean(step_nums)))

    step_nums = []
    for i in range(10):
        step_num = gym_env.test()
        step_nums.append(step_num)
        print('step:{}'.format(step_num))
    print('origin mean:{}'.format(np.mean(step_nums)))

    # ddpg.train(max_episodes=1000)


if __name__ == "__main__":
    train()
