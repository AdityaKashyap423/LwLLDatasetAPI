from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import pandas as pd
import shutil
from lwll_dataset_prep.logger import log


@dataclass
class mnist(BaseProcesser):
    """
    Our source data is in the form:
    mnist_png
    - training
    - - 0
    - - - img1.png
    - - - img2.png
    - - - etc.
    - - 1
    - - - imgx.png
    - - - imgy.png
    - - - etc.
        ...
    - - 9
    - - - imgx.png
    - - - imgx.png
    - - - etc.
    - testing (same as training)

    Roughly 60k in training folder and 10k in testing folder

    We will transform this into our LwLL format for image problems (See this repo's README for that format)

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'mnist'
    _task_type: str = 'image_classification'
    _urls: List[str] = field(default_factory=lambda: ['https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz'])
    _sample_size_train: int = 5000
    _sample_size_test: int = 1000
    _valid_extensions: List[str] = field(default_factory=lambda: ['.png'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

    def download(self) -> None:
        # Download
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=False)
            log.info("Done")

    def process(self) -> None:
        # Extract the tar
        for fname in self._fnames:
            self.extract_tar(dir_name=self._path_name, fname=fname)

        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)

        # Create our data schema
        df_train = pd.DataFrame(columns=['id', 'class'])
        df_test = pd.DataFrame(columns=['id', 'class'])

        for _class in range(10):
            imgs = pd.DataFrame([{'id': p.name, 'class': str(_class)} for p in self.path.joinpath(f'mnist_png/training/{_class}').iterdir()])
            df_train = pd.concat([df_train, imgs])
            imgs = pd.DataFrame([{'id': p.name, 'class': str(_class)} for p in self.path.joinpath(f'mnist_png/testing/{_class}').iterdir()])
            df_test = pd.concat([df_test, imgs])

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)

        # Move the raw data files
        orig_paths = [_p for _p in self.path.joinpath(f'mnist_png/training').glob('*/*') if _p.suffix in self._valid_extensions]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_paths, dest_type='train')

        orig_paths = [_p for _p in self.path.joinpath(f'mnist_png/testing').glob('*/*') if _p.suffix in self._valid_extensions]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_paths, dest_type='test')

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='image_classification',
                                 sample_number_of_samples_train=self._sample_size_train,
                                 sample_number_of_samples_test=self._sample_size_test,
                                 sample_number_of_classes=10,
                                 full_number_of_samples_train=len(df_train),
                                 full_number_of_samples_test=len(df_test),
                                 full_number_of_classes=10,
                                 number_of_channels=1,
                                 classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='http://yann.lecun.com/exdb/mnist/',
                                 license_requirements='None',
                                 license_citation='[LeCun et al., 1998a] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. http://yann.lecun.com/exdb/publis/index.html#lecun-98',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        for f in self._fnames:
            # We assume the tar files have no '.'s in their name before `.tar.gz` or just `.tar`
            d = f.split('.')[0]
            shutil.rmtree(self.path.joinpath(d))

        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        # self.push_data_to_cloud(dir_name=self._path_name, dataset_type='external', task_type=self._task_type)
        log.info("Done")
