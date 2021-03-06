# -*- coding: utf-8 -*-
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
""" Collection of tf.train.SessionRunHooks
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import os

import numpy as np
import six
import yaml


import tensorflow as tf
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer  # pylint: disable=E0611
from tensorflow.python.training import session_manager # pylint: disable=E0611
from tensorflow.python.client import timeline  # pylint: disable=E0611
from tensorflow import gfile

from seq2seq.configurable import Configurable, abstractstaticmethod
from seq2seq import graph_utils, global_vars
from seq2seq.data import utils

# from seq2seq.metrics import score
from seq2seq.metrics.pythonrouge import rouge_scorer

FLAGS = tf.flags.FLAGS


@six.add_metaclass(abc.ABCMeta)
class TrainingHook(tf.train.SessionRunHook, Configurable):
  """Abstract base class for training hooks.
  """

  def __init__(self, params, model_dir, run_config):
    tf.train.SessionRunHook.__init__(self)
    Configurable.__init__(self, params, tf.contrib.learn.ModeKeys.TRAIN)
    self._model_dir = model_dir
    self._run_config = run_config
    self._session = None

  @property
  def model_dir(self):
    """Returns the directory model checkpoints are written to.
    """
    return os.path.abspath(self._model_dir)

  @property
  def is_chief(self):
    """Returns true if and only if the current process is the chief.
    This is used for distributed training.
    """
    return self._run_config.is_chief

  @abstractstaticmethod
  def default_params():
    raise NotImplementedError()

  def after_create_session(self, session, coord):
    self._session = session

class MetadataCaptureHook(TrainingHook):
  """A hook to capture metadata for a single step.
  Useful for performance debugging. It performs a full trace and saves
  run_metadata and Chrome timeline information to a file.

  Args:
    step: The step number to trace. The hook is only enable for this step.
  """

  def __init__(self, params, model_dir, run_config):
    super(MetadataCaptureHook, self).__init__(params, model_dir, run_config)
    self._active = False
    self._done = False
    self._global_step = None
    self._output_dir = os.path.abspath(self.model_dir)

  @staticmethod
  def default_params():
    return {"step": 10}

  def begin(self):
    self._global_step = tf.train.get_global_step()

  def before_run(self, _run_context):
    if not self.is_chief or self._done:
      return
    if not self._active:
      return tf.train.SessionRunArgs(self._global_step)
    else:
      tf.logging.info("Performing full trace on next step.")
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) #pylint: disable=E1101
      return tf.train.SessionRunArgs(self._global_step, options=run_options)

  def after_run(self, _run_context, run_values):
    if not self.is_chief or self._done:
      return

    step_done = run_values.results
    if self._active:
      tf.logging.info("Captured full trace at step %s", step_done)
      # Create output directory
      gfile.MakeDirs(self._output_dir)

      # Save run metadata
      trace_path = os.path.join(self._output_dir, "run_meta")
      with gfile.GFile(trace_path, "wb") as trace_file:
        trace_file.write(run_values.run_metadata.SerializeToString())
        tf.logging.info("Saved run_metadata to %s", trace_path)

      # Save timeline
      timeline_path = os.path.join(self._output_dir, "timeline.json")
      with gfile.GFile(timeline_path, "w") as timeline_file:
        tl_info = timeline.Timeline(run_values.run_metadata.step_stats)
        tl_chrome = tl_info.generate_chrome_trace_format(show_memory=True)
        timeline_file.write(tl_chrome)
        tf.logging.info("Saved timeline to %s", timeline_path)

      # Save tfprof op log
      tf.contrib.tfprof.tfprof_logger.write_op_log(
          graph=tf.get_default_graph(),
          log_dir=self._output_dir,
          run_meta=run_values.run_metadata)
      tf.logging.info("Saved op log to %s", self._output_dir)
      self._active = False
      self._done = True

    self._active = (step_done >= self.params["step"])


class TrainSampleHook(TrainingHook):
  """Occasionally samples predictions from the training run and prints them.

  Params:
    every_n_secs: Sample predictions every N seconds.
      If set, `every_n_steps` must be None.
    every_n_steps: Sample predictions every N steps.
      If set, `every_n_secs` must be None.
    sample_dir: Optional, a directory to write samples to.
    delimiter: Join tokens on this delimiter. Defaults to space.
  """

  #pylint: disable=missing-docstring

  def __init__(self, params, model_dir, run_config):
    super(TrainSampleHook, self).__init__(params, model_dir, run_config)
    self._sample_dir = os.path.join(self.model_dir, "samples")
    self._timer = SecondOrStepTimer(
        every_secs=self.params["every_n_secs"],
        every_steps=self.params["every_n_steps"])
    self._pred_dict = {}
    self._should_trigger = False
    self._iter_count = 0
    self._global_step = None
    self._source_delimiter = self.params["source_delimiter"]
    self._target_delimiter = self.params["target_delimiter"]
    self._is_rl = True

  @staticmethod
  def default_params():
    return {
        "every_n_secs": None,
        "every_n_steps": 1000,
        "source_delimiter": " ",
        "target_delimiter": " "
    }

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()
    self._pred_dict = graph_utils.get_dict_from_collection("predictions")
    self._source_emb = graph_utils.get_dict_from_collection("source_emb")
    self._logits = graph_utils.get_dict_from_collection("logits")
    ##self._logits_infer = graph_utils.get_dict_from_collection("logits_infer")
    ###self._logits_softmax = graph_utils.get_dict_from_collection("logits_softmax")
    self._loss = graph_utils.get_dict_from_collection("loss")
    self._vocab_tables = graph_utils.get_dict_from_collection("vocab_tables")
    
    if "train_op_rl" in graph_utils.get_dict_from_collection("train").keys():
      self._is_rl = True
      self._pred_dict_greedy = graph_utils.get_dict_from_collection("predictions_greedy")
      self._pred_dict_sampled = graph_utils.get_dict_from_collection("predictions_sampled")
    else:
      self._is_rl = False
    self._pred_dict = graph_utils.get_dict_from_collection("predictions_train")

    # Create the sample directory
    if self._sample_dir is not None:
      gfile.MakeDirs(self._sample_dir)
 

  def before_run(self, _run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      if self._is_rl:
        fetches = {
            "predicted_tokens": self._pred_dict["predicted_tokens"],
            "predicted_tokens_greedy": self._pred_dict_greedy["predicted_tokens"],
            "predicted_tokens_sampled": self._pred_dict_sampled["predicted_tokens"],
            "target_words": self._pred_dict["labels.target_tokens"],
            "target_len": self._pred_dict["labels.target_len"]
        }
      else:
        fetches = {
            "predicted_tokens": self._pred_dict["predicted_tokens"],
            "target_words": self._pred_dict["labels.target_tokens"],
            "target_len": self._pred_dict["labels.target_len"]
        }
      sessRunArgs = tf.train.SessionRunArgs([fetches, self._global_step])

      return sessRunArgs
    return tf.train.SessionRunArgs([{}, self._global_step])

  def after_run(self, _run_context, run_values):
    result_dict, step = run_values.results
    self._iter_count = step

    """    
    source_emb_logits_fetches = [
        #self._source_emb["source_message_emb"],
        #self._source_emb["source_topic_emb"],
        #self._logits["logits_message"],
        #self._logits["logits_topic"],
        #self._logits["logits_output"],
        #self._logits["logits_message_nan"],
        #self._logits["logits_topic_nan"],
        #self._vocab_tables["topic_words_id_tensor"],
        #self._logits["topic_word_location"],
        #self._logits_softmax["logits_softmax_output"],
        #self._logits_softmax["logits_exp_sum"],
        #self._logits_softmax["topic_words_mask"],
        #self._logits_softmax["logits_message_exp_nan"],
        #self._logits_softmax["logits_topic_exp_nan"],
        #self._logits_softmax["logits_message_exp"],
        #self._logits_softmax["logits_topic_exp"],
        self._loss["losses"],
        self._loss["loss"]
      ]
    """
    
    
    #################source_message_emb, source_topic_emb, logits_message, logits_topic, logits_output, logits_message_nan,logits_topic_nan,topic_words_id_tensor, topic_word_location,logits_softmax_output,logits_exp_sum,topic_words_mask, logits_message_exp_nan,logits_topic_exp_nan,logits_message_exp,logits_topic_exp,losses, loss = self._session.run(source_emb_logits_fetches)

    """
    if step%500 == 0:
        tf.logging.info("logits_message:{}".format(logits_message))
        tf.logging.info("logits_topic:{}".format(logits_topic))
        tf.logging.info("logits_output:{}".format(logits_output))
        tf.logging.info("logits_message sum:{}".format(np.sum(np.array(logits_message),axis=2)))
        tf.logging.info("logits_topic sum:{}".format(np.sum(np.array(logits_topic),axis=2)))
        tf.logging.info("logits_output sum:{}".format(np.sum(np.array(logits_output),axis=2)))
        ###tf.logging.info("logits_softmax_output:{}".format(logits_softmax_output))
        ###tf.logging.info("logits_exp_sum:{}".format(logits_exp_sum))
    """
    
    """
    with open("log","a") as f:
        f.write("step:{}".format(step))
        f.write("source_message_emb:{}".format(source_message_emb))       
        f.write("source_message_emb sum:{}".format(np.sum(np.array(source_message_emb),axis=2)))
        f.write("source_message_emb max value:{}".format(np.amax(source_message_emb),axis=2))
        f.write("source_message_emb min value:{}".format(np.amin(source_message_emb),axis=2))
        f.write("source_message_emb mean value:{}".format(np.average(np.array(source_message_emb))))
        f.write("source_message_emb std value:{}".format(np.std(np.array(source_message_emb))))
        f.write("source_topic_emb:{}".format(source_topic_emb))
        f.write("source_topic_emb sum:{}".format(np.sum(np.array(source_topic_emb),axis=2)))
        f.write("source_topic_emb max value:{}".format(np.amax(source_topic_emb),axis=2))
        f.write("source_topic_emb min value:{}".format(np.amin(source_topic_emb),axis=2))
        f.write("source_topic_emb mean value:{}".format(np.average(np.array(source_topic_emb))))
        f.write("source_topic_emb std value:{}".format(np.std(np.array(source_topic_emb))))
        
        f.write("logits_message:{}".format(logits_message))
        f.write("logits_message max:{}".format(np.amax(logits_message)))
        f.write("logits_topic:{}".format(logits_topic))
        f.write("logits_topic max:{}".format(np.amax(logits_topic)))
        f.write("logits_output:{}".format(logits_output))  
        f.write("logits_output sum:{}".format(np.sum(np.array(logits_output),axis=2)))
        f.write("logits_exp_sum:{}".format(logits_exp_sum))
        f.write("logits_exp_sum max:{}".format(np.amax(logits_exp_sum)))
        f.write("logits_exp_sum min:{}".format(np.amin(logits_exp_sum)))
        f.write("logits_message_exp:{}".format(logits_message_exp))
        f.write("logits_topic_exp:{}".format(logits_topic_exp))
        f.write("logits_message_exp max:{}".format(np.amax(logits_message_exp)))
        f.write("logits_topic_exp max:{}".format(np.amax(logits_topic_exp)))
        ###f.write("topic_words_mask:{}".format(topic_words_mask))
        f.write("losses:{}".format(losses))
        f.write("loss:{}".format(loss))
        f.write("logits_message_nan:{}".format(logits_message_nan))
        f.write("logits_topic_nan:{}".format(logits_topic_nan))
        f.write("logits_message_exp_nan:{}".format(logits_message_exp_nan))
        f.write("logits_topic_exp_nan:{}".format(logits_topic_exp_nan))
        
    """
        
    ##tf.logging.info("loss:{}".format(loss))
    #tf.logging.info("logits_message_infer:{}".format(logits_message_infer))
    #tf.logging.info("logits_topic_infer:{}".format(logits_topic_infer))
    #tf.logging.info("topic_word_location:{}".format(topic_word_location))
    ###tf.logging.info("topic_words_id_tensor:{}".format(topic_words_id_tensor))
    
    if not self._should_trigger:
      return None
    # Convert dict of lists to list of dicts
    result_dicts = utils.convertDict(result_dict)
    
    # Print results
    
    result_str = ""
    result_str += ("=" * 100) + "\n"
    for result in result_dicts:
      target_len = result["target_len"]
      target_slice = result["target_words"][1:target_len]

      predicted_str = utils.decode_tokens_for_blue(result["predicted_tokens"], self._target_delimiter)[1] [0]
      if self._is_rl:
        predicted_str_greedy = utils.decode_tokens_for_blue(result["predicted_tokens_greedy"], self._target_delimiter)[1][0]
        # predicted_str_greedy = self._target_delimiter.encode("utf-8").join(
        #     predicted_slice_greedy).decode("utf-8")
        predicted_str_sampled = utils.decode_tokens_for_blue(result["predicted_tokens_sampled"], self._target_delimiter)[1][0]
        # predicted_str_sampled = self._target_delimiter.encode("utf-8").join(
        #     predicted_slice_sampled).decode("utf-8")

      result_str += "PREDICTED: " + predicted_str + "\n"

      if self._is_rl:
        result_str += "GREEDY: " + predicted_str_greedy + "\n"
        result_str += "SAMPLED: " + predicted_str_sampled + "\n"
      
      target_str = self._target_delimiter.encode("utf-8").join(
          target_slice).decode("utf-8")
      result_str += "REF: " + target_str + "\n"
      if self._is_rl:
        # result_str += "BLEU of pred_greedy: {}\n".format(score.evaluate_captions([[target_str]], [predicted_str_greedy]))
        # result_str += "BLEU of pred_sampled: {}\n\n".format(score.evaluate_captions([[target_str]], [predicted_str_sampled]))
        result_str += "ROUGE of pred_greedy: {}\n".format(rouge_scorer.evaluate([[[target_str]]], [[predicted_str_greedy]]))
        result_str += "ROUGE of pred_sampled: {}\n\n".format(rouge_scorer.evaluate([[[target_str]]], [[predicted_str_sampled]]))
      else:
        # result_str += "BLEU of pred_greedy: {}\n\n".format(score.evaluate_captions([[target_str]], [predicted_str]))
        result_str += "ROUGE of pred_greedy: {}\n\n".format(rouge_scorer.evaluate([[[target_str]]], [[predicted_str]]))
    result_str += ("=" * 100) + "\n\n"
    tf.logging.info(result_str)
    if self._sample_dir:
      filepath = os.path.join(self._sample_dir,
                              "samples_{:06d}.txt".format(step))
      with gfile.GFile(filepath, "w") as file:
        file.write(result_str)
    self._timer.update_last_triggered_step(self._iter_count - 1)


class PrintModelAnalysisHook(TrainingHook):
  """Writes the parameters of the model to a file and stdout.
  """

  #pylint: disable=missing-docstring
  def __init__(self, params, model_dir, run_config):
    super(PrintModelAnalysisHook, self).__init__(params, model_dir, run_config)
    self._filename = os.path.join(self.model_dir, "model_analysis.txt")

  @staticmethod
  def default_params():
    return {}

  def begin(self):
    # Dump to file on the chief worker
    if self.is_chief:
      opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
      opts['dump_to_file'] = os.path.abspath(self._filename)
      tf.contrib.tfprof.model_analyzer.print_model_analysis(
          tf.get_default_graph(), tfprof_options=opts)

    # Print the model analysis
    with gfile.GFile(self._filename) as file:
      tf.logging.info(file.read())


class VariableRestoreHook(TrainingHook):
  """A hooks that restored variables from a given checkpoints.

  Params:
    prefix: Variables matching this prefix are restored.
    checkpoint_path: Path to the checkpoint to restore variables from.
  """

  def __init__(self, params, model_dir, run_config):
    super(VariableRestoreHook, self).__init__(params, model_dir, run_config)
    self._saver = None

  @staticmethod
  def default_params():
    return {"prefix": "", "checkpoint_path": ""}

  def begin(self):
    variables = tf.contrib.framework.get_variables(scope=self.params["prefix"])

    def varname_in_checkpoint(name):
      """Removes the prefix from the variable name.
      """
      prefix_parts = self.params["prefix"].split("/")
      checkpoint_prefix = "/".join(prefix_parts[:-1])
      return name.replace(checkpoint_prefix + "/", "")

    target_names = [varname_in_checkpoint(_.op.name) for _ in variables]
    restore_map = {k: v for k, v in zip(target_names, variables)}

    tf.logging.info("Restoring variables: \n%s",
                    yaml.dump({k: v.op.name
                               for k, v in restore_map.items()}))

    self._saver = tf.train.Saver(restore_map)

  def after_create_session(self, session, coord):
    self._saver.restore(session, self.params["checkpoint_path"])
    tf.logging.info("Successfully restored all variables")


class DelayStartHook(TrainingHook, tf.train.GlobalStepWaiterHook):
  """Delays the start of the current worker process until global step
  K * task_id is reached. K is a parameter.
  """
  def __init__(self, params, model_dir, run_config):
    TrainingHook.__init__(self, params, model_dir, run_config)
    self._task_id = self._run_config.task_id
    self._delay_k = self.params["delay_k"]
    self._wait_until_step = int(self._delay_k * self._task_id)
    tf.train.GlobalStepWaiterHook.__init__(self, self._wait_until_step)

  @staticmethod
  def default_params():
    return {"delay_k": 500}


class SyncReplicasOptimizerHook(TrainingHook):
  """A SessionRunHook handles ops related to SyncReplicasOptimizer."""

  def __init__(self, params, model_dir, run_config):
    super(SyncReplicasOptimizerHook, self).__init__(
        params, model_dir, run_config)
    self._sync_optimizer = None
    self._num_tokens = -1

    self._local_init_op = None
    self._ready_for_local_init_op = None
    self._q_runner = None
    self._init_tokens_op = None

  @staticmethod
  def default_params():
    return {}

  def begin(self):
    if global_vars.SYNC_REPLICAS_OPTIMIZER is not None:
      self._sync_optimizer = global_vars.SYNC_REPLICAS_OPTIMIZER
    else:
      return

    if self._sync_optimizer._gradients_applied is False:  # pylint: disable=protected-access
      raise ValueError(
          "SyncReplicasOptimizer.apply_gradient should be called before using "
          "the hook.")
    if self.is_chief:
      self._local_init_op = self._sync_optimizer.chief_init_op
      self._ready_for_local_init_op = (
          self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = self._sync_optimizer.get_chief_queue_runner()
      self._init_tokens_op = self._sync_optimizer.get_init_tokens_op(
          self._num_tokens)
    else:
      self._local_init_op = self._sync_optimizer.local_step_init_op
      self._ready_for_local_init_op = (
          self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = None
      self._init_tokens_op = None

  def after_create_session(self, session, coord):
    """Runs SyncReplicasOptimizer initialization ops."""

    if not self._sync_optimizer:
      return

    tf.logging.info("Found SyncReplicasOptimizer. Initializing.")

    local_init_success, msg = session_manager._ready(  # pylint: disable=protected-access
        self._ready_for_local_init_op, session,
        "Model is not ready for SyncReplicasOptimizer local init.")
    if not local_init_success:
      raise RuntimeError(
          "Init operations did not make model ready for SyncReplicasOptimizer "
          "local_init. Init op: %s, error: %s" %
          (self._local_init_op.name, msg))
    session.run(self._local_init_op)
    if self._init_tokens_op is not None:
      session.run(self._init_tokens_op)
    if self._q_runner is not None:
      self._q_runner.create_threads(
          session, coord=coord, daemon=True, start=True)

class TrainUpdateLoss(TrainingHook):
  """Update loss to obtain RL loss

  Params:
    delimiter: Join tokens on this delimiter. Defaults to space.
  """

  #pylint: disable=missing-docstring

  def __init__(self, params, model_dir, run_config):
    super(TrainUpdateLoss, self).__init__(params, model_dir, run_config)
    self._timer = SecondOrStepTimer(
        every_secs=self.params["every_n_secs"],
        every_steps=self.params["every_n_steps"])
    self._log_dir = os.path.join(self.model_dir, "logs")
    self._should_trigger = False
    self._pred_dict = {}
    self._iter_count = 0
    self._global_step = None
    self._source_delimiter = self.params["source_delimiter"]
    self._target_delimiter = self.params["target_delimiter"]
    self._is_rl = True

  @staticmethod
  def default_params():
    return {
        "source_delimiter": " ",
        "target_delimiter": " ",
        "every_n_secs": None,
        "every_n_steps": 10
    }

  def begin(self):
    self._iter_count = 0
    self._global_step = tf.train.get_global_step()
    self._pred_dict = graph_utils.get_dict_from_collection("predictions_train")
    self._pred_dict_greedy = graph_utils.get_dict_from_collection("predictions_greedy")
    self._pred_dict_sampled = graph_utils.get_dict_from_collection("predictions_sampled")
    self._train_dict = graph_utils.get_dict_from_collection("train")

    # Create the log directory
    if self._log_dir is not None:
      gfile.MakeDirs(self._log_dir)

    if "train_op_rl" in self._train_dict.keys():
      self._is_rl = True
    else:
      self._is_rl = False

  def before_run(self, _run_context):
    if self._is_rl:
      fetches = {
          "target_len": self._pred_dict["labels.target_len"]
      }
    else:
      fetches = {
          "target_len": self._pred_dict["labels.target_len"]
      }
    sessRunArgs = tf.train.SessionRunArgs([fetches, self._global_step])

    return sessRunArgs

  def after_run(self, _run_context, run_values):

    _, step = run_values.results
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    self._iter_count = step

    # Get results
    if self._is_rl:
      prep_fetches = [
        self._pred_dict_greedy["predicted_tokens"],
        self._pred_dict_sampled["predicted_tokens"],
        self._pred_dict["labels.target_tokens"],
        self._pred_dict["labels.target_len"],
        self._train_dict["log_prob_sum"],
        self._train_dict["loss"],
      ]
      predicted_tokens_greedy, predicted_tokens_sampled, target_words, target_len, log_prob_sum, loss = self._session.run(prep_fetches)
      
      _, decoded_greedy = utils.decode_tokens_for_blue(predicted_tokens_greedy, self._target_delimiter)
      masks, decoded_sampled = utils.decode_tokens_for_blue(predicted_tokens_sampled, self._target_delimiter)

      norms = [sum(x) for x in masks]
      # decode tokens from byte to string and prepare for blue evaluation 
      _, ref_decoded = utils.decode_tokens_for_blue(target_words[:, 1:], self._target_delimiter)
      
      # compute bleu scores for sampled generator and greedy generator
      # r = [score.evaluate_captions([k], [v])  for k, v in zip(ref_decoded, decoded_sampled)]
      # b = [score.evaluate_captions([k], [v]) for k, v in zip(ref_decoded, decoded_greedy)]
      r = [rouge_scorer.evaluate([[[k]]], [[v]])["ROUGE-2"]  for k, v in zip(ref_decoded, decoded_sampled)]
      b = [rouge_scorer.evaluate([[[k]]], [[v]])["ROUGE-2"] for k, v in zip(ref_decoded, decoded_greedy)]
      
      log_prob_mean = np.mean(log_prob_sum)

      loss_fetches = [
        self._train_dict["sum_loss"],
        self._train_dict["loss_rl"]
      ]
      feed_dict = {
        self._train_dict["rewards"]: r,
        self._train_dict["base_line"]: b,
        self._train_dict["norms"]: norms,
        self._train_dict["log_prob_sum_"]: log_prob_sum
        }

      sum_loss, loss_rl = self._session.run(loss_fetches, feed_dict)

      fetch = [
        self._train_dict["train_op_rl"],
        ]
        
      loss_rl = self._session.run(fetch, feed_dict)
      r_mean = sum(r)*1./len(r)
      b_mean = sum(b)*1./len(b)
      
      if self._should_trigger:
        # tf.logging.info("step: {:>5}, sum_loss: {:>7.4f}, loss: {:>7}, loss_rl: {:>7.4f}, r_mean: {:>7.4f}, b_mean: {:>7.4f}, log_prob_mean: {:>7.4f}".format(step, sum_loss, loss, loss_rl, r_mean, b_mean, log_prob_mean))
        log_outputs = "step: {}, sum_loss: {}, loss: {}, loss_rl: {}, r_mean: {}, b_mean: {}, log_prob_mean: {}".format(step, sum_loss, loss, loss_rl, r_mean, b_mean, log_prob_mean)
        tf.logging.info(log_outputs)
        if self._log_dir:
          filepath = os.path.join(self._log_dir,
              "logs.txt")
          with gfile.GFile(filepath, "a") as file:
            file.write(log_outputs + "\n")
        self._timer.update_last_triggered_step(self._iter_count - 1)

