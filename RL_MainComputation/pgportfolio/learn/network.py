#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import tflearn


class NeuralNetWork:
    def __init__(self, feature_number, rows, columns, layers, device):
        tf_config = tf.ConfigProto()
        self.session = tf.Session(config=tf_config)
        if device == "cpu":
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        else:
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.input_num    = tf.placeholder(tf.int32, shape=[])
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, feature_number, rows, columns])
        self.previous_w   = tf.placeholder(tf.float32, shape=[None, rows])
        self._rows        = rows
        self._columns     = columns

        self.layers_dict  = {}
        self.layer_count  = 0

        self.output       = self._build_network(layers)

    def _build_network(self, layers):
        pass


class CNN(NeuralNetWork):
    # input_shape (features, rows, columns)
    def __init__(self, feature_number, rows, columns, layers, device):
        NeuralNetWork.__init__(self, feature_number, rows, columns, layers, device)

    def add_layer_to_dict(self, layer_type, tensor, weights=True):

        self.layers_dict[layer_type + '_' + str(self.layer_count) + '_activation'] = tensor
        self.layer_count += 1

    # generate the operation, the forward computaion
    def _build_network(self, layers):
        network  = tf.transpose(self.input_tensor, [0, 2, 3, 1])
        network  = network / network[:, :, -1, 0, None, None]
        network1 = network         
        for layer_number, layer in enumerate(layers):
            if layer["type"] == "DenseLayer":
                network = tflearn.layers.core.fully_connected(network,
                                                              int(layer["neuron_number"]),
                                                              layer["activation_function"],
                                                              regularizer=layer["regularizer"],
                                                              weight_decay=layer["weight_decay"] )
                self.add_layer_to_dict(layer["type"], network)
            elif layer["type"] == "DropOut":
                network = tflearn.layers.core.dropout(network, layer["keep_probability"])
            elif layer["type"] == "EIIE_Dense":
                width = network.get_shape()[2]
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 [1, width],
                                                 [1, 1],
                                                 "valid",
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                self.add_layer_to_dict(layer["type"], network)
                network1 = tflearn.layers.conv_2d(network1, int(layer["filter_number"]),
                                                 [1, width],
                                                 [1, 1],
                                                 "valid",
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
            elif layer["type"] == "ConvLayer":
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 allint(layer["filter_shape"]),
                                                 allint(layer["strides"]),
                                                 layer["padding"],
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                self.add_layer_to_dict(layer["type"], network)
                network1 = tflearn.layers.conv_2d(network1, int(layer["filter_number"]),
                                                 allint(layer["filter_shape"]),
                                                 allint(layer["strides"]),
                                                 layer["padding"],
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
            elif layer["type"] == "MaxPooling":
                network = tflearn.layers.conv.max_pool_2d(network, layer["strides"])
            elif layer["type"] == "AveragePooling":
                network = tflearn.layers.conv.avg_pool_2d(network, layer["strides"])
            elif layer["type"] == "LocalResponseNormalization":
                network = tflearn.layers.normalization.local_response_normalization(network)
            elif layer["type"] == "EIIE_Output_WithW":
                width    = network.get_shape()[2]
                height   = network.get_shape()[1]
                features = network.get_shape()[3]
                network  = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                network1 = tf.reshape(network1, [self.input_num, int(height), 1, int(width*features)])
                w        = tf.reshape(self.previous_w, [-1, int(height), 1, 1])
                network  = tf.concat([network, w], axis=3)
                network  = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                network1 = tflearn.layers.conv_2d(network1, 1, [1, 1], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                self.add_layer_to_dict(layer["type"], network)
                network  = network[:, :, 0, 0]
                network1 = network1[:, :, 0, 0]
                network1 = tf.contrib.layers.fully_connected(network1,1,activation_fn=tf.sigmoid)
                network1 = network1 * 2

                self.voting = network
                self.add_layer_to_dict('voting', network, weights=False)
                network     = tflearn.layers.core.activation(network, activation="softmax")
                network     = tf.multiply(network1,network)
                self.add_layer_to_dict('softmax_layer', network, weights=False)
            
            else:
                raise ValueError("the layer {} not supported.".format(layer["type"]))
        return network


def allint(l):
    return [int(i) for i in l]

