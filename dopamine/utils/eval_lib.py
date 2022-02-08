
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dopamine.agents.rainbow.rainbow_agent import RainbowAgent
from dopamine.agents.llamn_network.llamn_agent import AMNAgent
from dopamine.discrete_domains.llamn_run_experiment import ExpertRunner
from dopamine.discrete_domains.llamn_run_experiment import LLAMNRunner

matplotlib.use('TkAgg')


def get_frame_nb(self):
  return self._frame_nb[self._game_index]


def set_frame_nb(self, value):
  self._frame_nb[self._game_index] = value


class EvalRunner:

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(self._checkpoint_dir)
    self._start_iteration = 0

  def _run_one_step(self, action):
    outputs = super()._run_one_step(action)
    if hasattr(self, 'features_heatmap'):
      self.display_features()
    if hasattr(self, 'saliency_path'):
      self.save_saliency()
    else:
      self._environment.render('human')
      time.sleep(self.delay / 1000)
    return outputs

  def save_saliency(self):
    if not hasattr(self, 'fig'):
      self.fig, self.ax = plt.subplots()
      if isinstance(self, ExpertRunner):
        EvalRunner.frame_nb = 0
      elif isinstance(self, LLAMNRunner):
        self._frame_nb = [0 for _ in range(self._nb_envs)]
        EvalRunner.frame_nb = property(get_frame_nb, set_frame_nb)

    saliency_map = self._agent.compute_saliency(self._agent.state)

    image_path = f"{self.saliency_path}_{self.frame_nb:03}.png"

    self.ax.cla()

    # Images
    if len(saliency_map.shape) > 1:
      curr_state = self._agent.state[0, ..., -1].astype(np.uint8)
      self.ax.imshow(curr_state, cmap='gray')
      self.ax.imshow(saliency_map, cmap='Reds', alpha=0.5)

    else:
      curr_state = self._agent.state[0, ..., -1]
      saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
      saliency_map = 2 * saliency_map - 1
      img = np.vstack((curr_state, saliency_map))
      self.ax.imshow(img, cmap='bwr')
      vlines = [self._environment.n_informative, self._environment.n_redundant, self._environment.n_repeated]
      vlines = np.cumsum(vlines) - 0.5
      self.ax.vlines(vlines, -0.5, 1.5, colors='black')
      self.ax.set_axis_off()

    self.fig.savefig(image_path)

    activation_file = f"{self.saliency_path}_activations"
    with open(activation_file, 'a') as file:
      file.write(str(saliency_map.sum()) + '\n')

    self.frame_nb += 1

  def display_features(self):
    if not hasattr(self, 'agent_features'):
      if isinstance(self, ExpertRunner):
        self.agent_features = self._agent.online_convnet(self._agent.state_ph).features
      elif isinstance(self, LLAMNRunner):
        self.agent_features = self._agent.convnet(self._agent.state_ph).features

    features = self._sess.run(self.agent_features, feed_dict={self._agent.state_ph: self._agent.state})

    if not hasattr(self, 'plt_figure'):
      plt.gcf().clear()
      self.plt_figure = plt.imshow(features.reshape(16, 32), cmap='bwr')
      self.colorbar = plt.colorbar()
      plt.clim(-2, 5)
      plt.show(block=False)
      plt.pause(0.01)
    else:
      self.plt_figure.set_data(features.reshape(16, 32))
      plt.draw()
      plt.pause(0.01)


class SaliencyAgent:

  def _build_mask(self, shape):
    length = 35
    sigma = 5
    z = np.linspace(-length/2., length/2., length)
    xx, yy = np.meshgrid(z, z)
    gaussian_kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)

    dim_x = shape[0]
    dim_y = shape[1]
    nx = dim_x + 2*length
    ny = dim_y + 2*length
    mask = np.zeros((dim_x*dim_y, nx, ny))

    for i in range(dim_x):
      for j in range(dim_y):
        mask[i*dim_x+j, i+length//2+1:i+3*length//2+1, j+length//2+1:j+3*length//2+1] = gaussian_kernel
    mask = mask[:, length:length+dim_x, length:length+dim_y]

    # RGB images
    if len(shape) == 3:
      mask = np.repeat(mask[..., np.newaxis], shape[2], axis=-1)

    self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)

  def _build_saliency_op(self, shape):
    dim_x = shape[0]
    dim_y = shape[1]

    sigma_blur = 5
    float_state = tf.cast(self.state_ph[0, ..., self.stack_size-1], tf.float32)
    blurred_state = tfa.image.gaussian_filter2d(float_state, (5, 5), sigma_blur)
    perturbed_state = float_state * (1 - self.mask) + blurred_state * self.mask
    perturbed_state = tf.expand_dims(tf.cast(perturbed_state, tf.uint8), axis=-1)

    if self.stack_size > 1:
      first_frames = tf.repeat(self.state_ph[..., :self.stack_size-1], dim_x*dim_y, axis=0)
      final_states = tf.concat([first_frames, perturbed_state], axis=3)

    else:
      final_states = perturbed_state

    if isinstance(self, RainbowAgent):
      perturbed_q_values = self.online_convnet(final_states).q_values
      perturbed_softmax = tf.nn.softmax(perturbed_q_values)
      output_softmax = tf.nn.softmax(self._net_outputs.q_values)
      error = tf.norm(perturbed_softmax - output_softmax, ord=1, axis=1)
      self.saliency_map = tf.reshape(error, (dim_x, dim_y))

    elif isinstance(self, AMNAgent):
      self.saliency_map = []
      for i in range(self.nb_experts):
        perturbed_q_values = self.convnet(final_states).q_values
        expert_mask = [n_action < self.expert_num_actions[i]
                       for n_action in range(self.llamn_num_actions)]

        partial_q_values = tf.boolean_mask(perturbed_q_values, expert_mask, axis=1)
        perturbed_softmax = tf.nn.softmax(partial_q_values)
        output_softmax = tf.nn.softmax(self._net_q_output[i])
        error = tf.norm(perturbed_softmax - output_softmax, ord=1, axis=1)
        self.saliency_map.append(tf.reshape(error, (dim_x, dim_y)))

  def _build_gradient_op(self):
    if isinstance(self, RainbowAgent):
      grad = tf.abs(tf.compat.v1.gradients(self._net_outputs.q_values, self.state_ph))
      self.saliency_map = grad[0][0, ..., -1]

    elif isinstance(self, AMNAgent):
      self.saliency_map = []
      for i in range(self.nb_experts):
        grad = tf.abs(tf.compat.v1.gradients(self._net_q_output[i], self.state_ph))
        self.saliency_map.append(grad[0][0, ..., -1])

  def compute_saliency(self, state):
    if not hasattr(self, 'saliency_map'):
      shape = state.shape[1:-1]
      # Images
      if len(shape) > 1:
        self._build_mask(shape)
        self._build_saliency_op(shape)
      else:
        self._build_gradient_op()

    if isinstance(self, RainbowAgent):
      return self._sess.run(self.saliency_map, feed_dict={self.state_ph: state})

    elif isinstance(self, AMNAgent):
      return self._sess.run(self.saliency_map[self.ind_expert], feed_dict={self.state_ph: state})
