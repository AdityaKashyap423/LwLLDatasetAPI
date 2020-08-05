#!/usr/bin/env python
# ========================================================================================== #
""" Participant dataset download functions and core utilities.
"""

# ========================================================================================== #
# Imports
# ------------------------------------------------------------------------------------------ #

from pathlib import Path
import fire
import boto3
import sys
import os
import tarfile
import shutil
from typing import Any, Union
from lwll_dataset_prep.logger import log

# ========================================================================================== #
# Variables
# ------------------------------------------------------------------------------------------ #
BASE_DATADIR = 'live/'
BUCKET_ID = 'lwll-datasets'

# ========================================================================================== #
# Classes
# ------------------------------------------------------------------------------------------ #

class ProgressPercentage(object):
    def __init__(self, o_s3bucket, key_name: str) -> None:  # type: ignore
        self._key_name = key_name
        boto_client = o_s3bucket.meta.client
        # ContentLength is an int
        self._size = boto_client.head_object(Bucket=o_s3bucket.name, Key=key_name)['ContentLength']
        self._seen_so_far = 0
        sys.stdout.write('\n')

    def __call__(self, bytes_amount: int) -> None:
        self._seen_so_far += bytes_amount
        percentage = (float(self._seen_so_far) / float(self._size)) * 100
        TERM_UP_ONE_LINE = '\033[A'
        TERM_CLEAR_LINE = '\033[2K'
        sys.stdout.write('\r' + TERM_UP_ONE_LINE + TERM_CLEAR_LINE)
        sys.stdout.write('{} {}/{} ({}%)\n'.format(self._key_name, str(self._seen_so_far), str(self._size), str(percentage)))
        sys.stdout.flush()

class CLI:
    """
    This utility is to get easy access and download of the compressed datasets that are put into the proper LwLL form.

    Functions:
    - download_data
        This is to download the different datasets

    - list_data
        This is to list what datasets are available for download if you only want to download a subset
    """

    def __init__(self) -> None:
        # These are read only keys to this bucket, they are hardcoded in here because they are so that any performer can download
        # the datasets onto non DMC hardware
        self.session = boto3.Session(
            aws_access_key_id='AKIAXNTA46J3YJ6LRKO7',
            aws_secret_access_key='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',
            region_name='us-east-1',
        )
        self._bucket_name = BUCKET_ID
        self._data_path = os.path.join(BASE_DATADIR, 'datasets/')
        self._label_path = os.path.join(BASE_DATADIR, 'labels/')
        self.client = self.session.client('s3')
        self.bucket = self.session.resource('s3').Bucket(self._bucket_name)  # noqa # pylint: disable=no-member

    def download_data(self,
                      dataset: str,
                      stage: str = 'external',
                      output: str = '.',
                      overwrite: bool = False
                      ) -> str:
        """
        Utility to method to download and unzip the compressed datasets from our S3 bucket

         Args:
            dataset: The dataset name, which is required.
            stage: Either 'development' or 'external', for collecting different variants of the datasets.
                Default is 'external' which also downloads the corresponding labels (development does not).
                Available shorthands: ['d', 'dev', 'develop', 'e', 'ext']
            output: Directory to put the datasets in.
            overwrite: Determines whether or not to do an overwrite the dataset location locally. If `True`, and a directory exists
                with the name already, we will not attempt to download and unzip.

        Returns:
            Done

        Raises:
            Invalid dataset if the specified dataset is not in the s3 bucket.

        """
        assert any(stage == x for x in ['d', 'dev', 'develop', 'development', 'e', 'ext', 'external']), \
            f"Stage must be either 'development' or 'external', including shorthands for either; got {stage}."
        log.debug(f"Data Path: {self._data_path}")
        contents: list = self._get_contents(self._data_path)
        log.debug(f"Contents: {contents}")
        datasets = [d['Key'] for d in contents]
        dataset_names = [d.split('/')[-1].split('.')[0] for d in datasets]
        output_dir = os.path.join(output, 'external') if any(stage == x for x in ['e', 'ext', 'external']) else os.path.join(output, 'development')
        datasets_downloaded = []

        if dataset not in dataset_names + ['ALL']:
            log.info(f"`{dataset}` not in available datasets and not keyword `ALL`. Returning...")
            return ''
        else:
            if dataset != 'ALL':
                # ugly way of doing this for now, will need to come back to rewrite this api
                # TODO: Fix this script to have a clearer API and less redundant code.
                ds = [d for d in datasets if d.split('/')[-1].split('.')[0] == dataset][0]
                self._download_helper(ds, output_dir, overwrite, stage)
                datasets_downloaded.append(dataset)
            else:
                for ds in datasets:
                    self._download_helper(ds, output_dir, overwrite, stage)
                    datasets_downloaded.append(ds)

        return f'Finished downloading: "{datasets_downloaded}" to\n\t"{os.path.abspath(output_dir)}"'

    def _download_helper(self, dataset: str, output_dir: str, overwrite: bool, stage: str) -> None:
        self._download_dataset(dataset, output_dir, overwrite)
        if any(stage == x for x in ['e', 'ext', 'external']):
            log.info('Downloading labels...')
            log.debug(self._label_path)
            log.debug(dataset)
            dataset_name = dataset.split('/')[-1].split('.')[0]
            labels: list = self._get_contents(os.path.join(self._label_path, dataset_name))
            log.debug(f"Labels: {labels}")
            for label in labels:
                self._download_dataset(label['Key'], output_dir, overwrite)
        return

    def download(self, *args: Any, **kwargs: Any) -> str:
        return self.download_data(*args, **kwargs)

    def _download_dataset(self, data_path: str, output_dir: str, overwrite: bool) -> None:
        log.debug(f"Data path: {data_path}")
        if 'labels/' in data_path:
            dataset, labelset, labelfile = data_path.rsplit('labels/')[1].split('/')
            # Since datasets are tarballed as 'dataset/dataset_full/*' & 'dataset/dataset_sample/*'\
            # this matches the location and style as '*dataset*/labels_*labelset*/*labelfile*', etc.
            fname, fext = labelfile.split('.', 1)
            output_dir = os.path.join(output_dir, dataset, f"labels_{labelset}")
            output = os.path.join(output_dir, fname)
            dataset = f"{dataset}-{labelset}"

        else:
            dataset, fext = data_path.rsplit('/', 1)[1].split('.', 1)
            output = os.path.join(output_dir, dataset)

        if Path(output).is_dir() and not overwrite:
            log.info(f" `{dataset}` is already downloaded and `overwrite` is set to False. Skipping `{dataset}`\n\t*Note: This does not guarantee the newest version of the dataset...")  # noqa

        else:
            if Path(output).exists():
                log.info(f"Deleting existing {dataset} data...")
                shutil.rmtree(output)

            if not Path(output_dir).exists():
                os.makedirs(output_dir)
            # ········································································ #
            # Download
            # ········································································ #
            progress = ProgressPercentage(self.bucket, data_path)
            self.bucket.download_file(data_path, f'{output}.{fext}', Callback=progress)

            # ········································································ #
            # Extract
            # ········································································ #
            if fext == 'tar.gz':
                log.info('Extracting tarball...')
                tarfile.open(f'{output}.tar.gz', 'r:gz').extractall(f'{output_dir}')
                # Remove Zip
                log.info('Cleaning up...')
                os.remove(f'{output}.tar.gz')
        return

    def list_data(self, query: str = None) -> list:
        """ Utility method to list all available datasets currently processed.

        Args:
            query: Dataset name or string to search for.
        Returns:
            Set of dataset names
        """
        prefix, recurse = (self._data_path, False) if query is None else (os.path.join(self._data_path, query), True)
        contents: list = self._get_contents(objects=prefix, recurse=recurse)
        keys = [d['Key'].split('/')[-1].split('.')[0] for d in contents]
        return keys

    def list(self, query: str = None) -> list:
        return self.list_data(query)

    def _get_contents(self, objects: Union[str, dict], recurse: bool = True) -> list:  # type: ignore
        """ Utility method to list all available datasets by crawling aws prefixes.

        Args:
            objects: Bucket objects or directory strings.
            recurse: Whether to loop over directories in the S3 bucket.
        Returns:
            content: Set of all bucket object metadata dictionaries.
        """

        content: list = []
        if isinstance(objects, str):
            contents = self.client.list_objects(Bucket=self._bucket_name, Prefix=objects, Delimiter='/')
            return self._get_contents(objects=contents, recurse=recurse)
        elif isinstance(objects, dict):
            keys = objects.keys()
            if 'Contents' in keys:
                content.extend(objects['Contents'])
            if 'CommonPrefixes' in keys and recurse:
                for prefix in objects['CommonPrefixes']:
                    content.extend(self._get_contents(objects=prefix['Prefix'], recurse=recurse))
        return content

# ========================================================================================== #
# Call / Runner
# ------------------------------------------------------------------------------------------ #

def main() -> None:
    fire.Fire(CLI)


if __name__ == '__main__':

    fire.Fire(CLI)