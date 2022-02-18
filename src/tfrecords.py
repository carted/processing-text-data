# Copyright (c) Carted.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
import apache_beam as beam

from typing import Dict, Union


def _bytes_feature(bytes_input: bytes) -> tf.train.Feature:
    """Encodes given data as a byte feature."""
    bytes_list = tf.train.BytesList(value=[bytes_input])
    return tf.train.Feature(bytes_list=bytes_list)


def _floats_feature(value):
    """Returns a float_list from a float / double."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tfr_example(
    raw_features: Dict[str, Union[Dict[str, str], tf.Tensor, str]]
) -> tf.Tensor:
    """Creates a tf.train.Example instance from high level features."""
    feature = {
        "summary": _bytes_feature(raw_features["summary"].encode("utf-8")),
        "summary_average_embeddings": _floats_feature(
            raw_features["summary_average_embeddings"].numpy().tolist()
        ),
        "label": _bytes_feature(raw_features["label"].encode("utf-8")),
    }
    # Wrap as a training example.
    feature = tf.train.Features(feature=feature)
    example = tf.train.Example(features=feature)
    return example


class FeaturesToSerializedExampleFn(beam.DoFn):
    """DoFn class to create a tf.train.Example from high level features."""

    def process(self, features):
        example = create_tfr_example(features)
        yield example.SerializeToString()
