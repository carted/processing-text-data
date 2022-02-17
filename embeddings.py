from ml_collections import ConfigDict
from typing import List, Dict
import tensorflow_text
import tensorflow_hub as hub
import tensorflow as tf


def contiguous_group_average_vectors(vectors, groups):
    """Works iff sum(groups) == len(vectors)

    Example:
        Inputs: vectors: A dense 2D tensor of shape = (13, 3)
                groups : A dense 1D tensor with values [2, 5, 1, 4, 1]
                indicating that there are 5 groups.

        Objective: Compute a 5x3 matrix where the first row
                    is the average of the rows 0-1 of `vectors`,
                    the second row is the average of rows 2-6 of
                    vectors, the third row is the row 7 of vectors,
                    the fourth row is the average of rows 8-11 of
                    vectors and the fifth and final row is the row
                    12 of vectors.

        Logic: A selection mask matrix is generated
                mask = [[1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                        [0. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
                        [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
                        [0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 0.]
                        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]

                This mask is then multiplied with `vectors` to get a
                matrix of shape (5, 3) called `summed_vectors` where
                each row contains the group sums.

                `summed_vectors` is then devided by `groups` to
                obtain the averages.
        Author: Nilabhra Roy Chowdhury (Nilabhra@)
    """
    groups = tf.expand_dims(tf.cast(groups, dtype=tf.int32), axis=1)
    group_cumsum = tf.cumsum(groups)

    mask = tf.repeat(
        tf.expand_dims(tf.range(tf.shape(vectors)[0]), axis=0),
        repeats=tf.shape(groups)[0],
        axis=0,
    )
    mask = tf.cast(mask < group_cumsum, dtype=tf.float32)

    def complete_mask(mask):
        neg_mask = tf.concat(
            (tf.expand_dims(tf.ones_like(mask[0]), axis=0), 1 - mask[:-1]), axis=0
        )
        return mask * neg_mask

    mask = tf.cond(
        tf.greater(tf.shape(groups)[0], 1),
        true_fn=lambda: complete_mask(mask),
        false_fn=lambda: mask,
    )

    summed_vectors = tf.matmul(mask, vectors)
    averaged_vectors = summed_vectors / tf.cast(groups, dtype=tf.float32)

    return averaged_vectors


def get_text_encodings(examples: List, config: ConfigDict, chunk_size=50):
    """Generates average text encodings from text descriptions.
    
    Many of the utilities used in this function were written by Nilabhra
    Roy Chowdhury (@Nilabhra).
    """
    # Loading the text preprocessor and setting it as an attribute on the
    # first invocation.
    if not hasattr(get_text_encodings, "preprocessor"):
        get_text_encodings.preprocessor = hub.load(config.preprocessor_path)

    # Loading the encoder and setting it as an attribute on the first invocation.
    if not hasattr(get_text_encodings, "encoder"):
        get_text_encodings.encoder = hub.load(config.encoder_path)

    def prepare_bert_inputs(tokens: List[tf.RaggedTensor], token_lens: List[tf.Tensor]):
        """Pack the tokens w.r.t BERT inputs."""
        max_token_len = tf.reduce_max(token_lens)
        packer = hub.KerasLayer(
            get_text_encodings.preprocessor.bert_pack_inputs,
            arguments={
                "seq_length": tf.math.minimum(
                    max_token_len + 2,
                    config.max_seq_len,  # +2 to consider the [CLS] and [SEP] tokens.
                )
            },
        )
        bert_inputs = packer([tokens])
        return bert_inputs

    def encode_with_bert(inputs: Dict[str, tf.Tensor]):
        """Computes encodings with BERT."""
        bert_outputs = get_text_encodings.encoder(inputs)
        return bert_outputs["pooled_output"]

    def compute_text_encoding(
        tokens: List[tf.RaggedTensor], token_lens: List[tf.Tensor]
    ):
        """Packs BERT inputs and then computes text encodings."""
        bert_inputs = prepare_bert_inputs(tokens, token_lens)
        bert_outputs = encode_with_bert(bert_inputs)
        return bert_outputs

    # Gather text related features.
    text_tokens = examples["summary_tokens"]
    text_token_lens = examples["summary_token_lens"]
    text_num_sentences = examples["summary_num_sentences"]

    # Encode with BERT in a batch-wise manner to prevent OOM.
    len_idx = len(text_token_lens)
    all_bert_encodings = []

    # Sort sequences to reduce compute waste.
    sort_idx = tf.argsort(text_token_lens, direction="DESCENDING", axis=0)
    unsort_idx = tf.argsort(
        sort_idx, direction="ASCENDING", axis=0
    )  # indices to unsort the sorted embeddings

    sorted_all_text_tokens = tf.gather(text_tokens, sort_idx, axis=0)
    sorted_all_text_token_lens = tf.gather(text_token_lens, sort_idx, axis=0)

    for idx in range(0, len_idx, chunk_size):
        bert_encodings = compute_text_encoding(
            sorted_all_text_tokens[idx : idx + chunk_size],
            sorted_all_text_token_lens[idx : idx + chunk_size],
        )
        all_bert_encodings.append(bert_encodings)

    all_bert_encodings = tf.concat(all_bert_encodings, axis=0)

    # Unsort the encodings.
    all_bert_encodings = tf.gather(all_bert_encodings, unsort_idx, axis=0)

    # Perform averaging.
    averaged_encodings = contiguous_group_average_vectors(
        all_bert_encodings, text_num_sentences
    )

    return averaged_encodings
