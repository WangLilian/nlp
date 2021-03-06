# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Operations related to calculating sequence losses.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from seq2seq import graph_utils

from seq2seq.encoders.conv_encoder_utils import *


def cross_entropy_sequence_loss(logits, targets, sequence_length):
  """Calculates the per-example cross-entropy loss for a sequence of logits and
    masks out all losses passed the sequence length.

  Args:
    logits: Logits of shape `[T, B, vocab_size]`
    targets: Target classes of shape `[T, B]`
    sequence_length: An int32 tensor of shape `[B]` corresponding
      to the length of each input

  Returns:
    A tensor of shape [T, B] that contains the loss per example, per time step.
    
  """
  with tf.name_scope("cross_entropy_sequence_loss"):
      
    ###losses = cross_entropy_with_softmax_losses(logits,targets)
    losses = cross_entropy(logits,targets)
    ###tf.logging.info("losses:{}"+str(losses))
    """
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=targets)
    """
    ##tf.logging.info("sequence_length:{}"+str(sequence_length))  ###shape=(32,)
    ##tf.logging.info("targets:{}"+str(targets))   ###shape=(?, 32)
    
    # Mask out the losses we don't care about
    loss_mask = tf.sequence_mask(
        tf.to_int32(sequence_length), tf.to_int32(tf.shape(targets)[0]))
    ###tf.logging.info("loss_mask:{}"+str(loss_mask)) ### shape=(32, ?)
    losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])
    ###tf.logging.info("losses:{}"+str(losses)) ### shape=(32, ?)
    
    return losses
