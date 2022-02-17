import apache_beam as beam
import pandas as pd


from .configs import get_bert_encoder_config

PREPROCESSOR_PATH = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
ENCODER_PATH = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1"
EMBEDDING_DIM = 768
MAX_SEQ_LEN = 512


ENCODING_CONFIG = get_bert_encoder_config(
    PREPROCESSOR_PATH, ENCODER_PATH, EMBEDDING_DIM, MAX_SEQ_LEN
)

with beam.Pipeline() as pipeline:
    (
        pipeline
        | "Filename" >> beam.Create(["train_data.txt"])
        # Each element is a Pandas DataFrame, so we can do any Pandas operation.
        | "Read CSV"
        >> beam.Map(pd.read_csv, sep=" ::: ", names=["id", "movie", "genre", "summary"])
        # We yield each element of all the DataFrames into a PCollection of dictionaries.
        | "To dictionaries" >> beam.FlatMap(lambda df: df.to_dict("records"))
        # Reshuffle to make sure parallelization is balanced.
        | "Reshuffle" >> beam.Reshuffle()
        # Print the elements in the PCollection.
        | "Print" >> beam.Map(lambda x: (x["id"]))
        | beam.Map(print)
    )

