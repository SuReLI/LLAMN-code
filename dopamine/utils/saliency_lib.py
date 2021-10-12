
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from dopamine.agents.rainbow.rainbow_agent import RainbowAgent
from dopamine.agents.llamn_network.llamn_agent import AMNAgent


class SaliencyAgent:

  def _build_mask(self):
    length = 35
    sigma = 5
    z = np.linspace(-length/2., length/2., length)
    xx, yy = np.meshgrid(z, z)
    gaussian_kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    n = 84 + 2*length
    mask = np.zeros((84*84, n, n))

    for i in range(84):
      for j in range(84):
        mask[i*84+j, i+length//2+1:i+3*length//2+1, j+length//2+1:j+3*length//2+1] = gaussian_kernel
    mask = mask[:, length:length+84, length:length+84]

    self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)

  def _build_saliency_op(self):
    sigma_blur = 5
    float_state = tf.cast(self.state_ph[0, :, :, 3], tf.float32)
    blurred_state = tfa.image.gaussian_filter2d(float_state, (5, 5), sigma_blur)
    perturbed_state = float_state * (1 - self.mask) + blurred_state * self.mask
    perturbed_state = tf.expand_dims(tf.cast(perturbed_state, tf.uint8), axis=3)
    first_frames = tf.repeat(self.state_ph[:, :, :, :3], 84*84, axis=0)
    final_states = tf.concat([first_frames, perturbed_state], axis=3)

    if isinstance(self, RainbowAgent):
      perturbed_q_values = self.online_convnet(final_states).q_values
      error = tf.norm(perturbed_q_values - self._net_outputs.q_values, axis=1)
      self.saliency_map = tf.reshape(error, (84, 84))

    elif isinstance(self, AMNAgent):
      self.saliency_map = []
      for i in range(self.nb_experts):
        perturbed_q_values = self.convnet(final_states).q_values
        expert_mask = [n_action < self.expert_num_actions[i]
                       for n_action in range(self.llamn_num_actions)]

        partial_q_values = tf.boolean_mask(perturbed_q_values, expert_mask, axis=1)
        error = tf.norm(partial_q_values - self._net_q_output[i], axis=1)
        self.saliency_map.append(tf.reshape(error, (84, 84)))

  def compute_saliency(self, state):
    if not hasattr(self, 'saliency_map'):
      self._build_mask()
      self._build_saliency_op()

    if isinstance(self, RainbowAgent):
      return self._sess.run(self.saliency_map, feed_dict={self.state_ph: state})

    elif isinstance(self, AMNAgent):
      return self._sess.run(self.saliency_map[self.ind_expert], feed_dict={self.state_ph: state})
