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
"""
An encoder that conv over embeddings, as described in
https://arxiv.org/abs/1705.03122.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from seq2seq import graph_utils

def parse_list_or_default(params_str, number, default_val, delimitor=','):
  param_list = []
  if params_str == "":
    param_list = [default_val] * number
  else:
    param_list = [int(x) for x in params_str.strip().split(delimitor)]
  return param_list


def linear_mapping_stupid(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping"):
  with tf.variable_scope(var_scope_name):
    print('name', tf.get_variable_scope().name) 
    input_shape_tensor = tf.shape(inputs)   # dynamic shape, no None
    input_shape = inputs.get_shape().as_list()    # static shape. may has None
    print('input_shape', input_shape)
    assert len(input_shape) == 3
    inputs = tf.reshape(inputs, [-1, input_shape_tensor[-1]])
     
    linear_mapping_w = tf.get_variable("linear_mapping_w", [input_shape[-1], out_dim], initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(dropout*1.0/input_shape[-1])))
    linear_mapping_b = tf.get_variable("linear_mapping_b", [out_dim], initializer=tf.zeros_initializer())
      

    output = tf.matmul(inputs, linear_mapping_w) + linear_mapping_b
    print('xxxxx_params', input_shape, out_dim)
    #output = tf.reshape(output, [input_shape[0], -1, out_dim])
    output = tf.reshape(output, [input_shape_tensor[0], -1, out_dim])
 
  return output
 
def linear_mapping(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping"):
  with tf.variable_scope(var_scope_name):
    input_shape = inputs.get_shape().as_list()    # static shape. may has None
    return tf.contrib.layers.fully_connected(inputs=inputs,num_outputs=out_dim,activation_fn=None, weights_initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(dropout*1.0/input_shape[-1])), biases_initializer=tf.zeros_initializer()) 
 
def linear_mapping_weightnorm(inputs, out_dim, in_dim=None, dropout=1.0, var_scope_name="linear_mapping"):
  tf.logging.info("linear_mapping_weightnorm,var_scope_name:{}".format(var_scope_name))
  with tf.variable_scope(var_scope_name):
    input_shape = inputs.get_shape().as_list()    # static shape. may has None
    input_shape_tensor = tf.shape(inputs)    
    # use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
    V = tf.get_variable('V', shape=[int(input_shape[-1]), out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(dropout*1.0/int(input_shape[-1]))), trainable=True)
    V_norm = tf.norm(V.initialized_value(), axis=0)  # V shape is M*N,  V_norm shape is N
    g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
    b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)   # weightnorm bias is init zero
    
    assert len(input_shape) == 3
    inputs = tf.reshape(inputs, [-1, input_shape[-1]])
    inputs = tf.matmul(inputs, V)
    inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim])
    #inputs = tf.matmul(inputs, V)    # x*v
    
    scaler = tf.div(g, tf.norm(V, axis=0))   # g/2-norm(v)
    inputs = tf.reshape(scaler,[1, out_dim])*inputs + tf.reshape(b,[1, out_dim])   # x*v g/2-norm(v) + b
    
    return inputs 
 
def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding="SAME", dropout=1.0,  var_scope_name="conv_layer"):    #padding should take attention
  tf.logging.info("conv1d_weightnorm,var_scope_name:{}".format(var_scope_name))
  ###with tf.variable_scope("conv_layer_"+str(layer_idx)):
  with tf.variable_scope(var_scope_name):
    in_dim = int(inputs.get_shape()[-1])
    V = tf.get_variable('V', shape=[kernel_size, in_dim, out_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4.0*dropout/(kernel_size*in_dim))), trainable=True)
    V_norm = tf.norm(V.initialized_value(), axis=[0,1])  # V shape is M*N*k,  V_norm shape is k  
    g = tf.get_variable('g', dtype=tf.float32, initializer=V_norm, trainable=True)
    b = tf.get_variable('b', shape=[out_dim], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
    
    # use weight normalization (Salimans & Kingma, 2016)
    W = tf.reshape(g, [1,1,out_dim])*tf.nn.l2_normalize(V,[0,1])
    inputs = tf.nn.bias_add(tf.nn.conv1d(value=inputs, filters=W, stride=1, padding=padding), b)   
    return inputs


def gated_linear_units(inputs):
  input_shape = inputs.get_shape().as_list()
  assert len(input_shape) == 3
  input_pass = inputs[:,:,0:int(input_shape[2]/2)]
  input_gate = inputs[:,:,int(input_shape[2]/2):]
  input_gate = tf.sigmoid(input_gate)
  return tf.multiply(input_pass, input_gate)
 

def conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict, mode, flag):  ############add encoder and decoder flag
  next_layer = inputs
  tf.logging.info("flag flag flag flag flag flag flag flag flag flag:{}".format(flag))
  for layer_idx in range(len(nhids_list)):
    nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx-1]
    nout = nhids_list[layer_idx]
    if nin != nout:
      #mapping for res add
      res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout=dropout_dict['src'], var_scope_name=flag+"_"+"linear_mapping_cnn_" + str(layer_idx))    
    else:
      res_inputs = next_layer
    #dropout before input to conv
    next_layer = tf.contrib.layers.dropout(
      inputs=next_layer,
      keep_prob=dropout_dict['hid'],
      is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)
   
    next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout*2, kernel_size=kwidths_list[layer_idx], padding="SAME", dropout=dropout_dict['hid'], var_scope_name=flag+"_"+"conv_layer_"+str(layer_idx)) 
    ''' 
    next_layer = tf.contrib.layers.conv2d(
        inputs=next_layer,
        num_outputs=nout*2,
        kernel_size=kwidths_list[layer_idx],
        padding="SAME",   #should take attention
        weights_initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4 * dropout_dict['hid'] / (kwidths_list[layer_idx] * next_layer.get_shape().as_list()[-1]))),
        biases_initializer=tf.zeros_initializer(),
        activation_fn=None,
        scope="conv_layer_"+str(layer_idx))
    '''    
    next_layer = gated_linear_units(next_layer)
    next_layer = (next_layer + res_inputs) * tf.sqrt(0.5)

  return next_layer 

def conv_decoder_stack(target_embed, enc_output, inputs, nhids_list, kwidths_list, dropout_dict, mode):
  next_layer = inputs
  for layer_idx in range(len(nhids_list)):
    nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx-1]
    nout = nhids_list[layer_idx]
    if nin != nout:
      #mapping for res add
      res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout=dropout_dict['hid'], var_scope_name="decoder_linear_mapping_cnn_" + str(layer_idx))      
    else:
      res_inputs = next_layer
    #dropout before input to conv
    next_layer = tf.contrib.layers.dropout(
      inputs=next_layer,
      keep_prob=dropout_dict['hid'],
      is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)
    # special process here, first padd then conv, because tf does not suport padding other than SAME and VALID
    next_layer = tf.pad(next_layer, [[0, 0], [kwidths_list[layer_idx]-1, kwidths_list[layer_idx]-1], [0, 0]], "CONSTANT")
    
    next_layer = conv1d_weightnorm(inputs=next_layer, layer_idx=layer_idx, out_dim=nout*2, kernel_size=kwidths_list[layer_idx], padding="VALID", dropout=dropout_dict['hid'], var_scope_name="decoder_conv_layer_"+str(layer_idx)) 
    '''
    next_layer = tf.contrib.layers.conv2d(
        inputs=next_layer,
        num_outputs=nout*2,
        kernel_size=kwidths_list[layer_idx],
        padding="VALID",   #should take attention, not SAME but VALID
        weights_initializer=tf.random_normal_initializer(mean=0, stddev=tf.sqrt(4 * dropout_dict['hid'] / (kwidths_list[layer_idx] * next_layer.get_shape().as_list()[-1]))),
        biases_initializer=tf.zeros_initializer(),
        activation_fn=None,
        scope="conv_layer_"+str(layer_idx))
    '''
    layer_shape = next_layer.get_shape().as_list()
    assert len(layer_shape) == 3
    # to avoid using future information 
    next_layer = next_layer[:,0:-kwidths_list[layer_idx]+1,:]

    next_layer = gated_linear_units(next_layer)
   
    # add attention
    # decoder output -->linear mapping to embed, + target embed,  query decoder output a, softmax --> scores, scores*encoder_output_c-->output,  output--> linear mapping to nhid+  decoder_output -->
    att_out = make_attention(target_embed, enc_output, next_layer, layer_idx) 
    
    ######
    att_out_size = att_out.get_shape().as_list()[-1]      #k
    ###print("att_out length:"+str(att_out.get_shape()))
    ###print("att_out_size:"+str(att_out_size))
    att_out_message, att_out_topic = tf.split(att_out,[tf.cast(att_out_size/2,tf.int64),tf.cast(att_out_size/2,tf.int64)],2)    

    ######message    
    next_layer_message = (next_layer + att_out_message) * tf.sqrt(0.5) 

    # add res connections
    next_layer_message += (next_layer_message + res_inputs) * tf.sqrt(0.5) 

    ######topic
    ###embed_size = target_embed.get_shape().as_list()[-1]
    ###encoder_output_a_message, encoder_output_a_topic = tf.split(enc_output.outputs,[embed_size,embed_size],2) ######
    ###enc_output_hidden_state=linear_mapping_weightnorm(encoder_output_a_message, next_layer.get_shape().as_list()[-1], var_scope_name="linear_mapping_enc_output_addto_before_softmax")
    next_layer_topic = (next_layer + att_out_topic + att_out_message) * tf.sqrt(0.5)       #########reinforce the message information

    # add res connections
    ###########next_layer_topic += (next_layer_topic + res_inputs) * tf.sqrt(0.5) ##########
    next_layer_topic = (next_layer_topic + res_inputs) * tf.sqrt(0.5)
    
    
    next_layer_output = tf.concat([next_layer_message,next_layer_topic],2)
  ###return next_layer
  return next_layer_output

 
def make_attention(target_embed, encoder_output, decoder_hidden, layer_idx):
  with tf.variable_scope("attention_layer_" + str(layer_idx)):
    embed_size = target_embed.get_shape().as_list()[-1]      #k
    ###print("target_embed_size:"+str(embed_size))
    dec_hidden_proj = linear_mapping_weightnorm(decoder_hidden, embed_size, var_scope_name="linear_mapping_att_query")  # M*N1*k1 --> M*N1*k
    dec_rep = (dec_hidden_proj + target_embed) * tf.sqrt(0.5)   ########highway?
 
    ###encoder_output_a = encoder_output.outputs
    ###encoder_output_c = encoder_output.attention_values    # M*N2*K
    ###print("encoder_output.attention_values length[0]:"+str(tf.shape(encoder_output.attention_values)[0])+" encoder_output.attention_values length[1]:"+str(tf.shape(encoder_output.attention_values)[1]))

    encoder_output_a_message, encoder_output_a_topic = tf.split(encoder_output.outputs,[embed_size,embed_size],2) ######
    encoder_output_c_message, encoder_output_c_topic = tf.split(encoder_output.attention_values,[embed_size,embed_size],2)  ######
    #######print("encoder_output_a_message length[0]:"+str(encoder_output_a_message.get_shape().as_list()[0])+" encoder_output_a_message:"+str(encoder_output_a_message.get_shape().as_list()[1])) 
    
    ######
    att_score_message_share = tf.matmul(dec_rep, encoder_output_a_message, transpose_b=True)  #M*N1*K  ** M*N2*K  --> M*N1*N2
    att_score_message = tf.nn.softmax(att_score_message_share)        


    length_message = tf.cast(tf.shape(encoder_output_c_message), tf.float32)

    att_out_message = tf.matmul(att_score_message, encoder_output_c_message) * length_message[1] * tf.sqrt(1.0/length_message[1])    #M*N1*N2  ** M*N2*K   --> M*N1*k     

    att_out_message = linear_mapping_weightnorm(att_out_message, decoder_hidden.get_shape().as_list()[-1], var_scope_name="linear_mapping_att_out_message")
    ######

    att_score_topic = tf.matmul(dec_rep, encoder_output_a_topic, transpose_b=True)  #M*N1*K  ** M*N2*K  --> M*N1*N2
    """it's the idea of RNN style, but maybe not work in CNN. here have a problem, the dynamic size of encoder_output_a_message
    ##embed_size = target_embed.get_shape().as_list()[-1]
    ##encoder_output_a_message, encoder_output_a_topic = tf.split(encoder_output.outputs,[embed_size,embed_size],2) ######
    ###############print("encoder_output_a_message.get_shape().as_list()[-1]:"+str(encoder_output_a_message.get_shape().as_list()[-1]))
    enc_output_hidden_state=linear_mapping_weightnorm(encoder_output_a_message, att_score_topic.get_shape().as_list()[-1], var_scope_name="linear_mapping_enc_output_addto_topic_attention")
    att_score_topic = att_score_topic + enc_output_hidden_state
    """
    ###att_score_topic = tf.nn.softmax(att_score_topic)################
    ##att_score_message = tf.matmul(dec_rep, encoder_output_a_message, transpose_b=True)
    att_score_topic = tf.nn.softmax(att_score_topic+att_score_message_share) ###try this style   
  
    length_topic = tf.cast(tf.shape(encoder_output_c_topic), tf.float32)

    att_out_topic = tf.matmul(att_score_topic, encoder_output_c_topic) * length_topic[1] * tf.sqrt(1.0/length_topic[1])    #M*N1*N2  ** M*N2*K   --> M*N1*k     

    att_out_topic = linear_mapping_weightnorm(att_out_topic, decoder_hidden.get_shape().as_list()[-1], var_scope_name="linear_mapping_att_out_topic")
    
    att_out = tf.concat([att_out_message,att_out_topic],2)

  return att_out
"""
def topic_softmax(logits_message,logits_topic):  ###(exp(Vi)+exp(Ki)) / (sum(exp(Vi))+sum(exp(Ki))) , if the word is a topic word in the mean while
    logits_message_exp = tf.exp(logits_message)
    logits_topic_exp = tf.exp(logits_topic)
    
    logits_exp_sum = tf.concat([logits_message_exp,logits_topic_exp],-1)   ##require sum of the last dim
    logits_exp_sum = tf.reduce_sum(logits_exp_sum,-1)
    logits_exp_sum = tf.expand_dims(logits_exp_sum,-1)
    
    vocab_size = logits_message_exp.get_shape().as_list()[-1]
    logits_exp_sum = tf.tile(logits_exp_sum,[1,1,vocab_size])
    
    logits_output = (logits_message_exp + logits_topic_exp)/logits_exp_sum
    
    return logits_output
"""

def topic_softmax(logits_message,logits_topic,batch_size):  ###(exp(Vi)+exp(Ki)) / (sum(exp(Vi))+sum(exp(Ki))) , if the word is a topic word in the mean while
    logits_message_exp = tf.exp(logits_message*0.1)
    logits_topic_exp = tf.exp(logits_topic*0.1)
    
    #logits_message_exp = tf.clip_by_norm(logits_message_exp,0.1)
    #logits_topic_exp = tf.clip_by_norm(logits_topic_exp,0.1)
    
    
    logits_message_exp_nan=tf.is_nan(logits_message_exp)
    logits_message_exp_nan=tf.where(logits_message_exp_nan)
    
    logits_topic_exp_nan=tf.is_nan(logits_topic_exp)
    logits_topic_exp_nan=tf.where(logits_topic_exp_nan)
    
    
    topic_words_id_tensor = graph_utils.get_dict_from_collection("vocab_tables")["topic_words_id_tensor"]  
    vocab_size = logits_topic_exp.get_shape().as_list()[-1]
    topic_word_onehot = tf.contrib.layers.one_hot_encoding(topic_words_id_tensor,num_classes=vocab_size)
    topic_word_location = tf.reduce_sum(topic_word_onehot,0)
    topic_word_location = tf.expand_dims(topic_word_location, 0)
    topic_words_mask = tf.tile(topic_word_location, [batch_size,1])
    
    tf.logging.info("logits_topic_exp:{}".format(logits_topic_exp))
    tf.logging.info("topic_words_mask:{}".format(topic_words_mask))
            
    logits_exp_sum = tf.concat([logits_message_exp, topic_words_mask*logits_topic_exp],-1)   ##require sum of the last dim
    ###logits_exp_sum = logits_topic_exp   ##require sum of the last dim
    ###logits_exp_sum = logits_message_exp   ##require sum of the last dim
    ###logits_exp_sum = tf.concat([logits_message_exp, logits_topic_exp],-1)   ##require sum of the last dim
    ###logits_exp_sum = tf.add(logits_message_exp, topic_words_mask*logits_topic_exp)   ##require sum of the last dim
    logits_exp_sum = tf.reduce_sum(logits_exp_sum,-1)
    logits_exp_sum = tf.expand_dims(logits_exp_sum,-1)
    
    tf.logging.info("logits_exp_sum:{}".format(logits_exp_sum))
    
    ###if tf.contrib.learn.ModeKeys.TRAIN:
    ###logits_exp_sum = tf.tile(logits_exp_sum,[1,1,vocab_size])  ###ke you ke wu
    
    ###logits_exp_sum = tf.clip_by_value(logits_exp_sum,10000,logits_exp_sum)
    logits_softmax_output = (logits_message_exp + topic_words_mask*logits_topic_exp)/logits_exp_sum
    ###logits_softmax_output = logits_topic_exp / logits_exp_sum
    ###logits_softmax_output = logits_message_exp / logits_exp_sum
    ###logits_softmax_output = (logits_message_exp + logits_topic_exp)/logits_exp_sum
    ###logits_softmax_output = tf.add(logits_message_exp, topic_words_mask*logits_topic_exp)/logits_exp_sum
    graph_utils.add_dict_to_collection({
      #"logits_message_exp": logits_message_exp, 
      #"logits_topic_exp": logits_topic_exp,
      #"logits_message_exp_nan":logits_message_exp_nan,
      #"logits_topic_exp_nan":logits_topic_exp_nan,
      #"logits_exp_sum":logits_exp_sum,
      #"logits_softmax_output": logits_softmax_output,
      "topic_words_mask":topic_words_mask
      }, "logits_softmax")
        
    return logits_softmax_output

def cross_entropy_with_softmax_losses(logits,labels):
    logits_size = logits.get_shape().as_list()[-1]
    batch_size = labels.get_shape().as_list()[1]
    logits_message, logits_topic = tf.split(logits,[tf.cast(logits_size/2,tf.int64),tf.cast(logits_size/2,tf.int64)],2)
    vocab_size = logits_message.get_shape().as_list()[-1]    
    targets_one_hot = tf.contrib.layers.one_hot_encoding(labels,num_classes=vocab_size) 
    tf.logging.info("targets_one_hot shape:{}"+str(targets_one_hot))
    logits_softmax_output = topic_softmax(logits_message,logits_topic,batch_size)
    tf.logging.info("logits_softmax_output shape:{}"+str(logits_softmax_output))
    losses = -tf.reduce_sum(targets_one_hot * tf.log(tf.clip_by_value(logits_softmax_output,1e-10,1.0)), -1) ###cross_entropy
    ###losses = -tf.reduce_sum(targets_one_hot * tf.log(logits_softmax_output,1e-10,1.0), -1)
    ###losses = tf.transpose(losses)
    tf.logging.info("losses shape:{}"+str(losses))
    return losses

def cross_entropy(logits_softmax,labels):
    vocab_size = logits_softmax.get_shape().as_list()[-1]    
    targets_one_hot = tf.contrib.layers.one_hot_encoding(labels,num_classes=vocab_size) 
    tf.logging.info("targets_one_hot shape:{}"+str(targets_one_hot))
    losses = -tf.reduce_sum(targets_one_hot * tf.log(tf.clip_by_value(logits_softmax,1e-10,1.0)), -1)
    ###losses = -tf.reduce_sum(targets_one_hot * tf.log(logits_softmax_output,1e-10,1.0), -1)
    ###losses = tf.transpose(losses)
    tf.logging.info("losses shape:{}"+str(losses))
    return losses
