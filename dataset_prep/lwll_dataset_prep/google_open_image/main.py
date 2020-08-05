from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from dataclasses import dataclass, field
import csv
from typing import List, Dict
from pathlib import Path
import pandas as pd
from lwll_dataset_prep.logger import log

@dataclass
class google_open_image(BaseProcesser):
    """
    Source data: https://storage.googleapis.com/openimages/web/download.html
    V6 Subset with Bounding Boxes
    1.7 million images from 600 boudning box classes
    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'google_open_image'
    _task_type: str = 'object_detection'
    _urls: List[str] = field(default_factory=lambda: ['s3://open-images-dataset/tar/train_0.tar.gz',
                                                      's3://open-images-dataset/tar/train_1.tar.gz',
                                                      's3://open-images-dataset/tar/train_2.tar.gz',
                                                      's3://open-images-dataset/tar/train_3.tar.gz',
                                                      's3://open-images-dataset/tar/train_4.tar.gz',
                                                      's3://open-images-dataset/tar/train_5.tar.gz',
                                                      's3://open-images-dataset/tar/train_6.tar.gz',
                                                      's3://open-images-dataset/tar/train_7.tar.gz',
                                                      's3://open-images-dataset/tar/train_8.tar.gz',
                                                      's3://open-images-dataset/tar/train_9.tar.gz',
                                                      's3://open-images-dataset/tar/train_a.tar.gz',
                                                      's3://open-images-dataset/tar/train_b.tar.gz',
                                                      's3://open-images-dataset/tar/train_c.tar.gz',
                                                      's3://open-images-dataset/tar/train_d.tar.gz',
                                                      's3://open-images-dataset/tar/train_e.tar.gz',
                                                      's3://open-images-dataset/tar/train_f.tar.gz',
                                                      's3://open-images-dataset/tar/validation.tar.gz',
                                                      's3://open-images-dataset/tar/test.tar.gz',
                                                      'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',
                                                      'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',
                                                      'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv',
                                                      'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'])
    _sample_size_train: int = 20000
    _sample_size_test: int = 1000
    _valid_extensions: List[str] = field(default_factory=lambda: ['.jpg'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

    def download(self) -> None:
        # Download
        # We need to enable S3 downloading in self.download_data_from_url
        # For now, the dataset is donwloaded separatedly
        # for url, fname in zip(self._urls, self._fnames):
        #    self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=False)
        log.info("Done")

    def process(self) -> None:
        # Extract the tar
        for fname in self._fnames:
            self.extract_tar(dir_name=self._path_name, fname=fname)

        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)

        class_dict: Dict[str, str] = {}
        class_csv = self.path.joinpath('class-descriptions-boxable.csv')
        with open(class_csv) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                class_dict[row[0]] = row[1]
        classes = list(set(class_dict.values()))
        log.info(class_dict['/m/011k07'])

        training_ids = []
        training_boxes = []
        training_cates = []
        testing_ids = []
        testing_boxes = []
        testing_cates = []
        orig_train_paths = []
        orig_test_paths = []

        label_csv_train = self.path.joinpath('oidv6-train-annotations-bbox.csv')
        label_csv_val = self.path.joinpath('validation-annotations-bbox.csv')

        with open(label_csv_train) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                training_ids.append(f"{row[0]}.jpg")
                training_boxes.append(f"{int(float(row[4]))}, {int(float(row[6]))}, {int(float(row[5]))}, {int(float(row[7]))}")
                training_cates.append(class_dict[row[2]])
                orig_train_paths.append(self.path.joinpath(f"train/{row[0]}.jpg"))

        with open(label_csv_val) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                testing_ids.append(f"{row[0]}.jpg")
                testing_boxes.append(f"{int(float(row[4]))}, {int(float(row[6]))}, {int(float(row[5]))}, {int(float(row[7]))}")
                testing_cates.append(class_dict[row[2]])
                orig_test_paths.append(self.path.joinpath(f"validation/{row[0]}.jpg"))

        print(len(training_ids), len(testing_ids))
        print(len(list(set(orig_train_paths))), len(list(set(orig_test_paths))))
        # Create our data schema
        df_train = pd.DataFrame({'id': training_ids, 'bbox': training_boxes, 'class': training_cates})
        df_test = pd.DataFrame({'id': testing_ids, 'bbox': testing_boxes, 'class': testing_cates})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test, many_to_one=True)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=list(set(orig_train_paths)), dest_type='train', delete_original=False)

        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=list(set(orig_test_paths)), dest_type='test', delete_original=False)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='object_detection',
                                 sample_number_of_samples_train=self._sample_size_train,
                                 sample_number_of_samples_test=self._sample_size_test,
                                 sample_number_of_classes=len(classes),
                                 full_number_of_samples_train=len(df_train),
                                 full_number_of_samples_test=len(df_test),
                                 full_number_of_classes=len(classes),
                                 number_of_channels=3,
                                 classes=classes,
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='https://storage.googleapis.com/openimages/web/factsfigures.html',
                                 license_requirements='The annotations are licensed by Google LLC under CC BY 4.0\
                                 license. The images are listed as having a CC BY 2.0 license. Note: while we\
                                 tried to identify images that are licensed under a Creative Commons Attribution\
                                 license, we make no representations or warranties regarding the license status\
                                 of each image and you should verify the license for each image yourself.',
                                 license_citation='@article{kuznetsova2018open,title={The open images dataset v4:\
                                 Unified image classification, object detection, and visual relationship\
                                 detection at scale}, author={Kuznetsova, Alina and Rom, Hassan and Alldrin,\
                                 Neil and Uijlings, Jasper and Krasin, Ivan and Pont-Tuset, Jordi and Kamali,\
                                 Shahab and Popov, Stefan and Malloci, Matteo and Duerig, Tom and others},\
                                 journal={arXiv preprint arXiv:1811.00982}, year={2018}}',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        log.info("move train/ and validation/ back to $DATA/google-open-image/")
        log.info("Done")

        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
