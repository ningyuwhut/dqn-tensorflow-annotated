import gym
import random
import numpy as np
from .utils import rgb2gray, imresize

class Environment(object):
  def __init__(self, config):
    self.env = gym.make(config.env_name)

    screen_width, screen_height, self.action_repeat, self.random_start = \
        config.screen_width, config.screen_height, config.action_repeat, config.random_start

    self.display = config.display
    self.dims = (screen_width, screen_height)

    # self._screen = None
    self._screen = self.env.reset()#初始化
    self.reward = 0 #回报
    self.terminal = True #是否终止
#如果生命值为0 则重新开始一轮游戏,并采用动作0向前走一步，并绘图
#返回状态、reward、 、是否终止
  def new_game(self, from_random_game=False):
    if self.lives == 0:
      self._screen = self.env.reset()
    self._step(0)
    self.render()
    return self.screen, 0, 0, self.terminal

#返回stage、reward、action、terminal
  def new_random_game(self):
    self.new_game(True)
    xrange=range
    #先随机走n步,每步采取相同的动作0
    #最后走到的状态作为游戏开始状态
    for _ in xrange(random.randint(0, self.random_start - 1)):
      self._step(0)
    self.render()
    return self.screen, 0, 0, self.terminal
#返回state，reward，是否终止，调试信息(这里为空）
  def _step(self, action):
    self._screen, self.reward, self.terminal, _ = self.env.step(action)
#随机选择action 走一步
  def _random_step(self):
    action = self.env.action_space.sample()
    self._step(action)

  @ property
  def screen(self):
    return imresize(rgb2gray(self._screen)/255., self.dims)
    #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

  @property
  def action_size(self):
    return self.env.action_space.n

  @property
  def lives(self):
    return self.env.ale.lives()

  @property
  def state(self):
    return self.screen, self.reward, self.terminal

  def render(self):
    if self.display:
      self.env.render()
#走完一个动作后也绘图
  def after_act(self, action):
    self.render()

class GymEnvironment(Environment):
  def __init__(self, config):
    super(GymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    cumulated_reward = 0
    start_lives = self.lives

    for _ in range(self.action_repeat):
      self._step(action)
      cumulated_reward = cumulated_reward + self.reward
      #这句是啥意思
      #为啥训练时start_lives 大于 lives 了 就要terminal
      #start_lives 大于 lives 说明当前step导致丢失生命，所以终止，且cumulated_reward-1
      if is_training and start_lives > self.lives:
        cumulated_reward -= 1
        self.terminal = True

      if self.terminal:
        break

    self.reward = cumulated_reward

    self.after_act(action)
    return self.state

class SimpleGymEnvironment(Environment):
  def __init__(self, config):
    super(SimpleGymEnvironment, self).__init__(config)

  def act(self, action, is_training=True):
    self._step(action)

    self.after_act(action)
    return self.state
