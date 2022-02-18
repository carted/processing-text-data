# Processing text data at scale with Apache Beam and Cloud Dataflow

Presents an optimized [Apache Beam](https://beam.apache.org/) pipeline for generating sentence embeddings (runnable on [Cloud Dataflow](https://cloud.google.com/dataflow)). This repository 
accompanies our blog post: [Improving Dataflow Pipelines for Text Data Processing](https://www.carted.com/blog/improving-dataflow-pipelines-for-text-data-processing/).

We assume that you already have a billing enabled [Google Cloud Platform (GCP)](https://cloud.google.com/) project in case
you wanted to run the pipeline on Cloud Dataflow.

## Running the code locally

To run the code locally, first install the dependencies: `pip install -r requirements`. If you cannot
create a [Google Cloud Storage (GCS)](https://cloud.google.com/storage) Bucket then download the data using from 
[here](https://www.kaggle.com/rohitganji13/film-genre-classification-using-nlp). We just need the
`train_data.txt` file for our purpose. Also, note that without a GCS Bucket, one cannot
run the pipeline on Cloud Dataflow which is the main objective of this repository.

After downloading the dataset, make changes to the respective paths and command-line
arguments that use GCS in `main.py`.

Then execute `python main.py -r DirectRunner`.

## Running the code on Cloud Dataflow

1. Create a GCS Bucket and note its name. 
2. Then create a folder called `data` inside the Bucket.
3. Copy over the `train_data.txt` file to the `data` folder: `gsutil cp train_data.txt gs://<BUCKET-NAME>/data`.
4. Then run the following from the terminal:

    ```shell
    python main.py \
        --project <GCP-Project> \
        --gcs-bucket <BUCKET-NAME>
        --runner DataflowRunner
    ```

For more details please refer to our blog post: [Improving Dataflow Pipelines for Text Data Processing](https://www.carted.com/blog/improving-dataflow-pipelines-for-text-data-processing/).
