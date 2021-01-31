"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import logging
import numpy as np

from .utils import save_npy, load_npy

class ReplayMemory:
  def __init__(self, config, model_dir):
    self.model_dir = model_dir

    self.cnn_format = config.cnn_format
    self.memory_size = config.memory_size
    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)
    self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype = np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.history_length = config.history_length
    self.dims = (config.screen_height, config.screen_width)
    self.batch_size = config.batch_size
    self.count = 0
    self.current = 0
    #count表示memory中已经有的样本数量，current表示当前所在的样本下标
    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)

  def add(self, screen, reward, action, terminal):
    assert screen.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = screen
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size #如果当前样本已经达到上限，则从头开始设置

#取index 之前的history-1个元素，加上index自身，组成长度为history_length的样本
  def getState(self, index):
    assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    # 此时直接将index 作为最后一个元素，取index 之前 的history-1个元素，加上index自己，共history个元素
    if index >= self.history_length - 1:
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      #由于index < history_length-1,所以，需要从memory的最后取样本
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]
#从memory 中采样一个batch
#每次 采样一个合法的index， [index-1-history_length:index-1](闭区间)之间的元素作为一个样本,加入prestates中
#[index-history_length:index](闭区间)之间的元素作为一个样本,加入poststates中
#index 不能出现如下两个现象
#1.index >=current & index -history_length < current
#2. [index-history_length, index-1](闭区间)之间不能包含终止state

#第一个条件是说选择的样本不能包含current，大概是因为这样的话选出来的样本包含的帧就不是连续的了
#第二个条件比较好理解，是指样本中不能包含终止状态的帧。但是注意这里是限制prestates，poststates的最后一个帧还是可以包含终止状态的。

#从这里可以推测出，memory中的帧都是连续的
#prestates 和poststates 是连续的帧，只是差了一帧，即prestates[i]的下一帧就是poststates[i]

#一个样本包含四个帧，还有最后一个帧对应的action、reward、是否终止态。同时还有四个帧对应的下一帧。
  def sample(self):
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      #len(indexes)始终是最后一个元素的下标
      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
      return self.prestates, actions, rewards, self.poststates, terminals

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))
