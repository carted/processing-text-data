# Copyright (c) Carted.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ml_collections
import tensorflow as tf


def get_bert_encoder_config(
    preprocessor_path: str,
    encoder_path: str,
    embedding_dim: int,
    max_seq_len: int,
    trainable: bool = False,
) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.name = "bert"
    config.input_type = tf.int32
    config.preprocessor_path = preprocessor_path
    config.encoder_path = encoder_path
    config.encoder_inputs = ["input_word_ids", "input_type_ids", "input_mask"]
    config.embedding_dim = embedding_dim
    config.max_seq_len = max_seq_len
    config.trainable = trainable
    config.output_dim = embedding_dim
    return config
