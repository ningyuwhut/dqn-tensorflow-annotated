import numpy as np
#History 存储一次喂给agent的图片帧
#默认是NCHW格式的。
#history_length 相当于channel数
class History:
  def __init__(self, config):
    self.cnn_format = config.cnn_format

    batch_size, history_length, screen_height, screen_width = \
        config.batch_size, config.history_length, config.screen_height, config.screen_width

    self.history = np.zeros(
        [history_length, screen_height, screen_width], dtype=np.float32)
#将screen 加到history最后，最前面的history丢掉
  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen
#将所有图片都置为0
  def reset(self):
    self.history *= 0

  def get(self):
    if self.cnn_format == 'NHWC':
      return np.transpose(self.history, (1, 2, 0))
    else:
      return self.history
