# Copyright (c) Carted.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools


NAME = "processing_text_data"
VERSION = "0.1.0"
REQUIRED_PACKAGES = [
    "apache-beam[gcp]==2.31.0",
    "tensorflow==2.6.0",
    "tensorflow-estimator==2.6.0",
    "keras==2.6.0",
    "sentence_splitter==1.4",
    "seaborn==0.11.2",
    "pandas==1.3.2",
    "tensorflow_hub==0.12.0",
    "tensorflow_text==2.6.0",
    "ml_collections==0.1.0",
    "protobuf==3.18.0",
    "python-snappy==0.6.0",
    "google-apitools==0.5.31",
]


setuptools.setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
)
