from apache_beam.io.tfrecordio import WriteToTFRecord
from datetime import datetime
import apache_beam as beam
import pandas as pd
import functools
import argparse
import logging
import pprint
import math
import sys
import os

logging.getLogger().setLevel(logging.INFO)

SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(SCRIPT_DIR))

from configs import get_bert_encoder_config
from tfrecords import FeaturesToSerializedExampleFn
from text_tokenization import generate_features
from embeddings import get_text_encodings


# Initialize encoder configuration.
PREPROCESSOR_PATH = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
ENCODER_PATH = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1"
EMBEDDING_DIM = 768
MAX_SEQ_LEN = 512


def main(
    project: str,
    gcs_bucket: str,
    region: str,
    machine_type: str,
    max_num_workers: str,
    runner: str,
    chunk_size: int,
):
    job_timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")
    pipeline_args_dict = {
        "job_name": f"dataflow-text-processing-{job_timestamp}",
        "machine_type": machine_type,
        "num_workers": "1",
        "max_num_workers": max_num_workers,
        "runner": runner,
        "setup_file": os.path.join(SCRIPT_DIR, "setup.py"),
        "project": project,
        "region": region,
        "gcs_location": f"gs://{gcs_bucket}",
        "temp_location": f"gs://{gcs_bucket}/temp",
        "staging_location": f"gs://{gcs_bucket}/staging",
        "save_main_session": "True",
    }

    # Convert the dictionary to a list of (argument, value) tuples and flatten the list.
    pipeline_args = [(f"--{k}", v) for k, v in pipeline_args_dict.items()]
    pipeline_args = [x for y in pipeline_args for x in y]

    # Load the dataframe for counting the total number of samples it has. For larger
    # datasets, this should be performed separately.
    train_df = pd.read_csv(
        "train_data.txt",
        engine="python",
        sep=" ::: ",
        names=["id", "movie", "genre", "summary"],
    )
    total_examples = len(train_df)

    logging.info(
        f"Executing beam pipeline with args:\n{pprint.pformat(pipeline_args_dict)}"
    )

    with beam.Pipeline(argv=pipeline_args) as pipeline:
        encoding_config = get_bert_encoder_config(
            PREPROCESSOR_PATH, ENCODER_PATH, EMBEDDING_DIM, MAX_SEQ_LEN
        )
        configured_encode_examples = functools.partial(
            get_text_encodings, config=encoding_config, chunk_size=chunk_size
        )
        _ = (
            pipeline
            | "Read file" >> beam.Create(["train_data.txt"])
            | "Read CSV"
            >> beam.Map(
                pd.read_csv, sep=" ::: ", names=["id", "movie", "genre", "summary"]
            )
            | "To dictionaries" >> beam.FlatMap(lambda df: df.to_dict("records"))
            | "Generate features"
            >> beam.ParDo(generate_features, config=encoding_config)
            | "Intelligently Batch examples"
            >> beam.BatchElements(min_batch_size=chunk_size, max_batch_size=1000)
            | "Encode the text features" >> beam.ParDo(configured_encode_examples)
            | "Create TF Train example" >> beam.ParDo(FeaturesToSerializedExampleFn())
            | "Write TFRecord to GS Bucket"
            >> WriteToTFRecord(
                file_path_prefix=f"gs://{gcs_bucket}/tfrecords",
                file_name_suffix=f"_{job_timestamp}.tfrecord",
                num_shards=math.ceil(total_examples / 50),
            )
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Beam pipeline for generating TFRecords from a Pandas dataframe."
    )
    parser.add_argument(
        "-p",
        "--project",
        default="carted-gcp",
        type=str,
        help="The name of the GCP project.",
    )
    parser.add_argument(
        "-b",
        "--gcs-bucket",
        default="processing-text-data",
        type=str,
        help="The Google Cloud Storage bucket name.",
    )
    parser.add_argument(
        "-reg", "--region", default="us-central1", type=str, help="The GCP region.",
    )
    parser.add_argument(
        "-m",
        "--machine-type",
        type=str,
        default="n1-standard-1",
        help="Machine type for the Dataflow workers.",
    )
    parser.add_argument(
        "-w",
        "--max-num-workers",
        default="25",
        type=str,
        help="Number of maximum workers for Dataflow",
    )
    parser.add_argument(
        "-r",
        "--runner",
        type=str,
        choices=["DirectRunner", "DataflowRunner"],
        help="The runner for the pipeline.",
    )
    parser.add_argument(
        "-cs",
        "--chunk-size",
        type=int,
        default=50,
        help="Chunk size to use during BERT encoding.",
    )
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(**args)
