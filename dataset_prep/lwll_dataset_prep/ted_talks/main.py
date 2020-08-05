from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import os
import pandas as pd
from lwll_dataset_prep.logger import log

@dataclass
class ted_talks(BaseProcesser):
    """
    Our Source Data consists of sentance aligned pairs of tab delimented English next to Arabic

    Ex:

            english	 arabic
        0	(Music)	(موسيقى)
        1	(Applause)	(تصفيق)
        2	Worldwide, over 1.5 billion people experience ...	يعاني أكثر من 1.5 مليار شخص حول العالم من صراع...
        3	In response, people are forced to flee their c...	ونتيجة لذلك، يُجبر الناس على الهرب من بلادهم، ...
        4	Children, without a doubt, are the most innoce...	إن الأطفال من دون شك، هم أكثر الضحايا براءة و ...

    """
    _path_name: str = 'ted_talks'
    _task_type: str = 'machine_translation'
    _urls: List[str] = field(default_factory=lambda: ['http://data.statmt.org/ted-talks/en-ar.v2.aligned.clean.xz'])
    _sample_percent: float = 0.35
    _test_size: int = 2000

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = ['en-ar.v2.aligned.clean.xz']

    def download(self) -> None:
        # Download
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=False)
            log.info("Done")

    def process(self) -> None:
        # Extract the tar files
        for fname in self._fnames:
            self.extract_xz(dir_name=self._path_name, fname=fname)

        df = pd.read_csv(f"{self.path}/en-ar.v2.aligned.clean", sep='\t', header=None,
                         names=['english', 'arabic'], error_bad_lines=False)
        log.info(f"Read in file...")
        df.fillna('', inplace=True)
        log.info(f"{df.head()}")

        # Compute target Lengths
        df['target_size'] = df['english'].apply(lambda x: len(x))

        df.reset_index(inplace=True)
        df.rename({'index': 'id'}, inplace=True, axis=1)
        df.rename({'english': 'target'}, inplace=True, axis=1)
        df.rename({'arabic': 'source'}, inplace=True, axis=1)

        # Create Sample vs. Full and Train vs. Test Splits
        log.info('Generating splits...')
        train_full, test_full, train_sample, test_sample = self.generate_mt_splits(df, self._test_size, self._sample_percent)
        log.info(f"Dataset splits output:")
        log.info(f"train_full: {len(train_full)}")
        log.info(f"test_full: {len(test_full)}")
        log.info(f"train_sample: {len(train_sample)}")
        log.info(f"test_sample: {len(test_sample)}")

        sample_total_codecs = int(train_sample['target_size'].sum())
        full_total_codecs = int(train_full['target_size'].sum())

        # Resetting defualt indices for feather saving
        train_full.reset_index(inplace=True, drop=True)
        test_full.reset_index(inplace=True, drop=True)
        train_sample.reset_index(inplace=True, drop=True)
        test_sample.reset_index(inplace=True, drop=True)

        # Set up paths
        log.info('Creating Local Paths...')
        sample_path = str(self.data_path.joinpath(f"{self._path_name}").joinpath(f'{self._path_name}_sample'))
        full_path = str(self.data_path.joinpath(f"{self._path_name}").joinpath(f'{self._path_name}_full'))
        sample_labels_path = str(self.labels_path.joinpath(f"{self._path_name}").joinpath(f'{self._path_name}_sample'))
        full_labels_path = str(self.labels_path.joinpath(f"{self._path_name}").joinpath(f'{self._path_name}_full'))
        Path(sample_path).mkdir(exist_ok=True, parents=True)
        Path(full_path).mkdir(exist_ok=True, parents=True)
        Path(sample_labels_path).mkdir(exist_ok=True, parents=True)
        Path(full_labels_path).mkdir(exist_ok=True, parents=True)

        # Save our to paths
        log.info('Saving processed dataset out...')
        train_sample[['id', 'source']].to_feather(f"{sample_path}/train_data.feather")
        train_full[['id', 'source']].to_feather(f"{full_path}/train_data.feather")
        test_sample[['id', 'source']].to_feather(f"{sample_path}/test_data.feather")
        test_full[['id', 'source']].to_feather(f"{full_path}/test_data.feather")

        train_sample[['id', 'target']].to_feather(f"{sample_labels_path}/labels_train.feather")
        train_full[['id', 'target']].to_feather(f"{full_labels_path}/labels_train.feather")
        test_sample[['id', 'target']].to_feather(f"{sample_labels_path}/labels_test.feather")
        test_full[['id', 'target']].to_feather(f"{full_labels_path}/labels_test.feather")

        test_sample[['id']].to_feather(f"{sample_labels_path}/test_label_ids.feather")
        test_full[['id']].to_feather(f"{full_labels_path}/test_label_ids.feather")

        dataset_doc = DatasetDoc(name=f"{self._path_name}",
                                 dataset_type='machine_translation',
                                 sample_number_of_samples_train=len(train_sample),
                                 sample_number_of_samples_test=len(test_sample),
                                 sample_number_of_classes=None,
                                 full_number_of_samples_train=len(train_full),
                                 full_number_of_samples_test=len(test_full),
                                 full_number_of_classes=None,
                                 number_of_channels=None,
                                 classes=None,
                                 language_from='ara',
                                 language_to='eng',
                                 sample_total_codecs=sample_total_codecs,
                                 full_total_codecs=full_total_codecs,
                                 license_link='http://data.statmt.org/ted-talks/',
                                 license_requirements='None',
                                 license_citation='N/A',
                                 )
        log.info('Saving Metadata...')
        self.save_dataset_metadata(dir_name=f"{self._path_name}", metadata=dataset_doc)

        # Clean up unecessary files in directory for tar
        os.remove(f"{self.path}/en-ar.v2.aligned.clean.xz")
        os.remove(f"{self.path}/en-ar.v2.aligned.clean")

        # Push data to DynamoDB
        log.info('Pushing labels to Dynamo')
        self.push_to_dynamo(pd.concat([train_full, test_full]), 'ted_talks_full', 'id', 'target', 'target_size')
        self.push_to_dynamo(pd.concat([train_sample, test_sample]), 'ted_talks_sample', 'id', 'target', 'target_size')
        return

    def transfer(self) -> None:
        log.info(f"Pushing artifacts to appropriate cloud resources for {self._path_name}...")
        name = str(f"{self._path_name}")
        self.push_data_to_cloud(dir_name=name, dataset_type='development', task_type=self._task_type, is_mt=True)
        self.push_data_to_cloud(dir_name=name, dataset_type='external', task_type=self._task_type, is_mt=True)
        log.info("Done")
        return
