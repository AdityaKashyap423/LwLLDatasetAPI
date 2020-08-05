# LwLL API

## Downloading Data

A CLI has been built for easy access to datasets that have been wrapped. From the root of this directory, you can use the CLI for the following pieces of functionality:

Before using the cli you want to make sure you have the required dependencies.

```sh
virtualenv -p python3.6 env
source env/bin/activate
pip install -r cli_requirements.txt
```

Using `conda`

```sh
conda create -n lwll python=3.6
conda activate lwll
pip install -r cli_requirements.txt
```

### List Available Datasets

-   `list_data` or `list`

List the available datasets that are live and ready for model training and development. Optionally, use a query to search for a specific string.

#### Ex. List data in the live lwll-dataset bucket

```sh
python download.py list_data
```

Example output:

```output
cifar100
coco2014
face_detection
imagenet_1k
mnist
voc2009
```

#### Ex. Search for cifar datasets

```sh
python download.py list cifar
```

Example output:

```output
cifar100
```

### Download Data

-   `download_data` or `download`

Download and unpacks datasets matching the query string. If the `external` stage is specified, labels are downloaded as well.

The `dataset` argument can be set to a single dataset name, such as `mnist` and `cifar100`; use `list_data` or `list` for the list of available datasets.

The `stage` argument can be either `development` or `external`, specifying which dataset variant to download for your current stage of development. Available shorthands: `d`, `dev`, `develop`, `e`, `ext`

The `output` parameter is used to specify the directory to put the datasets. It's value defaults to the current directory (`.`) if not specified.

The `overwrite` parameter is used to optionally redownload a dataset if it already exists in the target output directory. It checks if a directory with the `dataset` name already exists in the `output` directory.

You can also optionally download ALL of the datasets for a particular stage by passing in `ALL` to the `dataset` argument. This is helpful if you want to do something like daily checks that you have all of the available datasets and skip if you already have the datasets. In this case you could do something like:

```sh
python download.py download_data --dataset ALL --stage development --output ~/lwll_datasets --overwrite False
python download.py download_data --dataset ALL --stage external --output ~/lwll_datasets --overwrite False
```

#### Ex. Download CIFAR-100 Explicitly

Download the cifar100 dataset explicitly in the `development` stage:

```sh
python download.py download_data --dataset cifar100 --stage development --output ~/lwll_datasets --overwrite True
```

Example output:

```output
live/datasets/cifar100.tar.gz 119537664/146285881 (81.5359070777309%)
live/datasets/cifar100.tar.gz 146285881/146285881 (100.0%)
Extracting tarball...
Cleaning up...
Finished downloading: "cifar" to
        "~/lwll_datasets/development"
```

#### Ex. Download CIFAR-100 with Labels

Download the newest version of cifar100 dataset, force overwriting any version you may have in that location.

```sh
python download.py download_data --dataset cifar100 --stage external --output ~/lwll_datasets --overwrite True
```

Example output:

```output
live/datasets/cifar100.tar.gz 146285881/146285881 (100.0%)
Extracting tarball...
Cleaning up...
Downloading labels...
live/labels/cifar100/full/labels_test.feather 232776/232776 (100.0%)
live/labels/cifar100/full/labels_train.feather 1162776/1162776 (100.0%)
live/labels/cifar100/sample/labels_test.feather 23464/23464 (100.0%)
live/labels/cifar100/sample/labels_train.feather 116368/116368 (100.0%)
Finished downloading: "cifar" to
        "~/lwll_datasets/external"
```

### File Structure

Upon downloading a dataset, it will be extracted within the output directory (ie. `~/lwll_datasets`) you've specified. Remember, labels aren't downloaded for the `development` stage.

```output
-- ~/lwll_datsets
-- -- external
-- -- -- cifar100
-- -- -- -- cifar100_full
-- -- -- -- -- test
-- -- -- -- -- -- img*.*
-- -- -- -- -- -- etc.
-- -- -- -- -- train
-- -- -- -- -- -- img*.*
-- -- -- -- -- -- etc.
-- -- -- -- cifar100_sample
-- -- -- -- -- test
-- -- -- -- -- -- img*.*
-- -- -- -- -- -- etc.
-- -- -- -- -- train
-- -- -- -- -- -- img*.*
-- -- -- -- -- -- etc.
-- -- -- -- labels_full
-- -- -- -- -- labels_test.feather
-- -- -- -- -- labels_train.feather
-- -- -- -- labels_sample
-- -- -- -- -- labels_test.feather
-- -- -- -- -- labels_train.feather
-- -- development
-- -- -- cifar100
-- -- -- -- cifar100_full
-- -- -- -- -- test
-- -- -- -- -- -- img*.*
-- -- -- -- -- -- etc.
-- -- -- -- -- train
-- -- -- -- -- -- img*.*
-- -- -- -- -- -- etc.
-- -- -- -- cifar100_sample
-- -- -- -- -- test
-- -- -- -- -- -- img*.*
-- -- -- -- -- -- etc.
-- -- -- -- -- train
-- -- -- -- -- -- img*.*
-- -- -- -- -- -- etc.
```

## Machine Translation Monolingual Corpora

For machine translation tasks, monolingual corpora are often very useful. Since these monolingual datasets do not fit into one of our defined buckets of `development` or `external` because of the non uniform schema, we allow you to download the monolingual corpora in a similar fashion calling a different utility script. When downloading, you should place these to the same place you are downloading your other datasets, however, these will be placed into a sub directory called `monolingual_corpora`.

Currently there are 2 corpora, which are wikipedia dumps in both English and Arabic. Because of this and that this will suffice for the time being and the forseeable future, you can download these individually like so:

```sh
python download_monolingual.py download_data --dataset wiki-ar --output ~/lwll_datasets --overwrite False
```

The two availale options for `--dataset` include `wiki-ar` and `wiki-en`.

The files are UTF-8 encoded text files with one wiki article per line.

## Development

This repo is for transforming the given datasets into the appropriate form and going the the process of:

1. Download from original source
2. Any manipulation of internal form
3. Upload to the destinations

### Current Functionality

Currently we are going down the path of having dataset prep be consistent for task types. The example dataset to build out this proof of concept is using the MNIST dataset. The idea is to have each dataset implement a `Processer` that is responsible for implementing a `download`, `process`, and `transfer` method. The `Processer` inherits from the `BaseProcesser` class that contains general purpose functionality in helping transform source datasets into the LwLL format.

The format is looking like we will keep a consistent folder schema throughout task types, but it may make sense in the future to in addition to the main `BaseProcesser` class, to also have "original format" Processers. i.e. Object classification original source that is arranged in class / folder structure vs. object classification original source arranged in label file format. This is not super important for the moment, but something to consider as making PR's to this code base.

Because we adhere to the interface of the `BaseProcesser` class, we will be able to write a main script that calls these methods uniformly. An example of going through data collection to processing of the example MNIST dataset is as follows:

```python
from datasets.mnist.main import MNISTProcesser

proc = MNISTProcesser()
proc.download()
proc.process()

```

The output files are transformed to the desired folder schema with consistent random seed based sampling in the users home directory under `~/lwll_datasets`.

### Dataset Metadata Schema

The dataset metadata is put into 3 different locations for proper consumption by appropriate resources.

#### Within Directory Structure

We store the base images within the `datasets/lwll_datasets` directory, these are what are included in the compressed files for reference. These do not include labels because those are expected to be queried through the api.

Schema for `image_classification` and `object_detection` tasks:

```sh
-- datasets
-- -- mnist
-- -- -- mnist_sample
-- -- -- -- train
-- -- -- -- -- img1.png
-- -- -- -- -- img2.png
-- -- -- -- -- etc...
-- -- -- -- test
-- -- -- -- -- img1000.png
-- -- -- -- -- img1001.png
-- -- -- -- -- etc...
-- -- -- mnist_full (Same as sample version but full dataset in this directory)
-- -- -- -- etc...
```

Schema for `machine_translation` tasks:

```sh
-- datsets
-- -- europarl-da-en
-- -- -- europarl-da-en_sample
-- -- -- -- train_data.feather
-- -- -- -- test_data.feather
-- -- -- europarl-da-en_full
-- -- -- -- train_data.feather
-- -- -- -- test_data.feather
```

## Task Type Standardized Label Schemas

We standardize the label format for consumption by the api for querying and delivering labels to the TA1 systems. The label file is standardized by task type as such (\*Note that the examples say 'class1', the actual descriptive lables are present, we do not obfuscate the actual labels in any way):

### Image Classification

```csv
id, label
'img_1.png', 'class3'
'img_2.png', 'class4'
'img_3.png', 'class2'
...
etc...
```

### Object Detection

```csv
id, label, class
'img_1.png', '<x_min>, <y_mmin>, <x_max>, <y_max>', 'class1'
'img_1.png', '<x_min>, <y_mmin>, <x_max>, <y_max>', 'class1'
'img_2.png', '<x_min>, <y_mmin>, <x_max>, <y_max>', 'class5'
'img_2.png', '<x_min>, <y_mmin>, <x_max>, <y_max>', 'class2'
'img_2.png', '<x_min>, <y_mmin>, <x_max>, <y_max>', 'class1'

```

### Machine Translation

```csv
TBD...
```
