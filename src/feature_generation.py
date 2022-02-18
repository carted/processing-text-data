# Copyright (c) Carted.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
import tensorflow_hub as hub
import apache_beam as beam
import pandas as pd

from typing import Union, Dict, Any, List, Tuple
from sentence_splitter import split_text_into_sentences
from ml_collections import ConfigDict


def generate_features(example: Union[pd.Series, Dict[str, Any]], config: ConfigDict):
    """Generates embeddings and labels from the dataset examples.
    
    Many of the utilities used in this function were written by Nilabhra
    Roy Chowdhury (@Nilabhra).
    """
    if not hasattr(generate_features, "tokenizer"):
        generate_features.tokenizer = hub.load(config.preprocessor_path).tokenize

    def _tokenize_text(text: List[str],) -> Tuple[tf.RaggedTensor, List[int]]:
        """Tokenizes a list of sentences.
        Args:
            text (List[str]): A list of sentences.
        Returns:
            Tuple[tf.RaggedTensor, List[int]]: Tokenized and indexed sentences, list
                containing the number of tokens per sentence.
        """
        token_list = generate_features.tokenizer(tf.constant(text))
        token_lens = [tokens.flat_values.shape[-1] for tokens in token_list]
        return token_list, token_lens

    text_features = {}
    split_sentences = split_text_into_sentences(example["summary"], language="en")

    text_features["summary"] = example["summary"]
    tokenized_text_feature, token_lens = _tokenize_text(split_sentences)
    text_features["summary_tokens"] = tokenized_text_feature
    text_features["summary_token_lens"] = token_lens
    text_features["summary_num_sentences"] = len(token_lens)

    text_features["label"] = example["genre"]
    return [text_features]


class DecodeFromTextLineDoFn(beam.DoFn):
    # Mind the space.
    def __init__(self, delimiter=" ::: "):
        self.delimiter = delimiter

    def process(self, text_line):
        splits = text_line.split(self.delimiter)
        genre = splits[-2]
        summary = splits[-1]
        packed_dict = {"summary": summary, "genre": genre}
        yield packed_dict
