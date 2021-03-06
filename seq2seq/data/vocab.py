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
"""Vocabulary related functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from tensorflow import gfile

from seq2seq import graph_utils

import random

import numpy as np

SpecialVocab = collections.namedtuple("SpecialVocab",
                                      ["UNK", "SEQUENCE_START", "SEQUENCE_END"])


class VocabInfo(
    collections.namedtuple("VocbabInfo",
                           ["path", "vocab_size", "special_vocab"])):
  """Convenience structure for vocabulary information.
  """

  @property
  def total_size(self):
    """Returns size the the base vocabulary plus the size of extra vocabulary"""
    return self.vocab_size + len(self.special_vocab)


def get_vocab_info(vocab_path):
  """Creates a `VocabInfo` instance that contains the vocabulary size and
    the special vocabulary for the given file.

  Args:
    vocab_path: Path to a vocabulary file with one word per line.

  Returns:
    A VocabInfo tuple.
  """
  with gfile.GFile(vocab_path) as file:
    vocab_size = sum(1 for _ in file)
  special_vocab = get_special_vocab(vocab_size)
  return VocabInfo(vocab_path, vocab_size, special_vocab)


def get_special_vocab(vocabulary_size):
  """Returns the `SpecialVocab` instance for a given vocabulary size.
  """
  return SpecialVocab(*range(vocabulary_size, vocabulary_size + 3))


def random_list_generate(min_value,max_value,list_size):  ######generate random numbers which mean 0, stddev 0.1
    rarray=np.random.uniform(min_value,max_value,size=list_size)
    mean=np.average(rarray)
    stddev=np.std(rarray)*10
    return [(value-mean)/stddev for value in list(rarray)]


def create_vocabulary_lookup_table_add_topics(filename, filename_topic, default_value=None):
  """Creates a lookup table for a vocabulary file.

  Args:
    filename: Path to a vocabulary file containg one word per line.
      Each word is mapped to its line number.
    default_value: UNK tokens will be mapped to this id.
      If None, UNK tokens will be mapped to [vocab_size]

    Returns:
      A tuple (vocab_to_id_table, id_to_vocab_table,
      word_to_count_table, vocab_size). The vocab size does not include
      the UNK token.
    """
  if not gfile.Exists(filename):
    raise ValueError("File does not exist: {}".format(filename))

  # Load vocabulary into memory
  with gfile.GFile(filename) as file:
    vocab = list(line.strip("\n") for line in file)
  vocab_size = len(vocab)

  has_counts = len(vocab[0].split("\t")) == 2
  if has_counts:
    vocab, counts = zip(*[_.split("\t") for _ in vocab])  ###turple
    counts = [float(_) for _ in counts]
    vocab = list(vocab)
  else:
    counts = [-1. for _ in vocab]

  # Add special vocabulary items
  special_vocab = get_special_vocab(vocab_size)
  vocab += list(special_vocab._fields)
  vocab_size += len(special_vocab)
  counts += [-1. for _ in list(special_vocab._fields)]

  if default_value is None:
    default_value = special_vocab.UNK

  tf.logging.info("Creating vocabulary lookup table of size %d", vocab_size)

  ##############random.shuffle(vocab)  ###test
  vocab_tensor = tf.constant(vocab)
  count_tensor = tf.constant(counts, dtype=tf.float32)
  ###vocab_idx_tensor = tf.range(vocab_size, dtype=tf.int64)
  idx = list(range(vocab_size)) ###vocab_index
  vocab_idx_tensor = tf.constant(idx,dtype=tf.int64)

  # Create ID -> word mapping
  id_to_vocab_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_idx_tensor, vocab_tensor, tf.int64, tf.string)
  id_to_vocab_table = tf.contrib.lookup.HashTable(id_to_vocab_init, "UNK")

  # Create word -> id mapping
  vocab_to_id_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, vocab_idx_tensor, tf.string, tf.int64)
  vocab_to_id_table = tf.contrib.lookup.HashTable(vocab_to_id_init,
                                                  default_value)

  # Create word -> count mapping
  word_to_count_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, count_tensor, tf.string, tf.float32)
  word_to_count_table = tf.contrib.lookup.HashTable(word_to_count_init, -1)
 
  ################calculate topic words and build target vocab topic words ids index     
  words=[]
  features=[]
  emb_size=0
  topic_word_num=200
  ###f = open(self.params["topic_model.path"],"r")
  f = open(filename_topic,"r")
  texts = f.readlines()
  for line in texts: 
     emb_size=len(line.split('\t')[1].split(' '))
     words.append(line.split('\t')[0])
     features.append([float(probability) for probability in line.split('\t')[1].split(' ')[0:emb_size]])
  f.close()    
  samples_size = len(words)    
  topic_words=[]
  fw = open("topic_words.txt","w")
  for i in range(0,emb_size):
     pro_dict={}
     for j in range(0,samples_size):
        pro_dict[words[j]]=features[j][i]
     prob_list = sorted(pro_dict.items(),key=lambda d:d[1],reverse=True)
     topic_words_list = [item[0] for item in prob_list[0:topic_word_num]]
     topic_words = topic_words + topic_words_list
     fw.write(' '.join(topic_words_list))
     fw.write('\n')
  fw.close()
  topic_words = sorted(list(set(topic_words)))
  ##tf.logging.info("topic_words:"+" ".join(topic_words[0:100]))
  ###topic_words_tensor = tf.convert_to_tensor(topic_words,dtype=tf.string)
  topic_words_tensor = tf.constant(topic_words,dtype=tf.string)
  
  graph_utils.add_dict_to_collection({"topic_words_tensor": topic_words_tensor}, "topic_words_tensor")    
  ###topic_words_id_tensor = target_vocab_to_id.lookup(topic_words_tensor)
  #################
  
  ### Load topic into memory
  with gfile.GFile(filename_topic) as file:
    vocab_topic = list(line.strip("\n") for line in file)
  vocab_topic_size = len(vocab_topic)
    
  vocab_topic, topic_embedding = zip(*[_.split("\t") for _ in vocab_topic])
  ###vocab_topic, topic_embedding = zip(*[ [_.split(" ")[0], ' '.join(_.split(" ")[1:257])] for _ in vocab_topic])
  topic_embedding = [list( float(_) for _ in _.split(" ") ) for _ in topic_embedding]
  
  topic_embedding=np.array(topic_embedding)
  topic_embedding[topic_embedding>1.0]=1.0
  topic_embedding=topic_embedding.tolist()
  
  topic_emb_size = len(topic_embedding[0])
  ###print("topic_emb_size:"+str(topic_emb_size))
  """
  for i in range(100):
    print(vocab_topic[i]+":origin ")
    print(topic_embedding[i][0:20])
  """
  
  ###vocab_topic = list(vocab_topic)
  
  """
  special_vocabTopic = get_special_vocab(vocab_topic_size)
  vocab_topic += list(special_vocabTopic._fields)
  vocab_topic_size += len(special_vocabTopic)
  topic_embedding += [[float(0)]*topic_emb_size for _ in list(special_vocabTopic._fields)]
  ###print("vocab_topic_size:"+str(vocab_topic_size))
  """

  ##vacab_topic_dict = []
  vocab_topic_emb = [[float(0)]*topic_emb_size]*len(idx)
  for vocab_idx in idx:
      if vocab[vocab_idx] in vocab_topic:
         ##vacab_topic_dict.append(topic_embedding[vocab_topic.index(vocab[vocab_idx])])
         vocab_topic_emb[vocab_idx] = topic_embedding[vocab_topic.index(vocab[vocab_idx])]
      else:
         ##vacab_topic_dict.append( [float(0)]*topic_emb_size )
         ### vocab_topic_emb[vocab_idx] = [float(0)]*topic_emb_size
         vocab_topic_emb[vocab_idx] = random_list_generate(0,1,256)
         
  """       
  for i in range(100):
      print(vocab[i]+":")
      print(vocab_topic_emb[i][0:20])
  """
         
  vocab_topic_emb_tensor = tf.constant(vocab_topic_emb, dtype=tf.float32)

  tf.logging.info("Creating topic word vocabulary lookup table of size %d", vocab_topic_size)

  """
  vocabTopic_tensor = tf.constant(vocab_topic)
  topicEmbedding_tensor = tf.constant(topic_embedding, dtype=tf.float32)

  # Create word -> topic embedding mapping
  vocab_to_embedding_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocabTopic_tensor, topicEmbedding_tensor, tf.string, tf.float32)
  vocab_to_embedding_table = tf.contrib.lookup.HashTable(vocab_to_embedding_init,[float(0)]*topic_emb_size)  ###wrong
  """
    
  ###topic_embedding = tf.constant(np.array(topic_embedding))

  return vocab_to_id_table, id_to_vocab_table, word_to_count_table, vocab_topic_emb_tensor, vocab_size


def create_vocabulary_lookup_table(filename, default_value=None):
  """Creates a lookup table for a vocabulary file.

  Args:
    filename: Path to a vocabulary file containg one word per line.
      Each word is mapped to its line number.
    default_value: UNK tokens will be mapped to this id.
      If None, UNK tokens will be mapped to [vocab_size]

    Returns:
      A tuple (vocab_to_id_table, id_to_vocab_table,
      word_to_count_table, vocab_size). The vocab size does not include
      the UNK token.
    """
  if not gfile.Exists(filename):
    raise ValueError("File does not exist: {}".format(filename))

  # Load vocabulary into memory
  with gfile.GFile(filename) as file:
    vocab = list(line.strip("\n") for line in file)
  vocab_size = len(vocab)

  has_counts = len(vocab[0].split("\t")) == 2
  if has_counts:
    vocab, counts = zip(*[_.split("\t") for _ in vocab])
    counts = [float(_) for _ in counts]
    vocab = list(vocab)
  else:
    counts = [-1. for _ in vocab]

  # Add special vocabulary items
  special_vocab = get_special_vocab(vocab_size)
  vocab += list(special_vocab._fields)
  vocab_size += len(special_vocab)
  counts += [-1. for _ in list(special_vocab._fields)]

  if default_value is None:
    default_value = special_vocab.UNK

  tf.logging.info("Creating vocabulary lookup table of size %d", vocab_size)

  vocab_tensor = tf.constant(vocab)
  count_tensor = tf.constant(counts, dtype=tf.float32)
  vocab_idx_tensor = tf.range(vocab_size, dtype=tf.int64)

  # Create ID -> word mapping
  id_to_vocab_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_idx_tensor, vocab_tensor, tf.int64, tf.string)
  id_to_vocab_table = tf.contrib.lookup.HashTable(id_to_vocab_init, "UNK")

  # Create word -> id mapping
  vocab_to_id_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, vocab_idx_tensor, tf.string, tf.int64)
  vocab_to_id_table = tf.contrib.lookup.HashTable(vocab_to_id_init,
                                                  default_value)

  # Create word -> count mapping
  word_to_count_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, count_tensor, tf.string, tf.float32)
  word_to_count_table = tf.contrib.lookup.HashTable(word_to_count_init, -1)

  return vocab_to_id_table, id_to_vocab_table, word_to_count_table, vocab_size
