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
import gzip
from lwll_dataset_prep.logger import log

# ========================================================================================== #
# Variables
# ------------------------------------------------------------------------------------------ #
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
        self._data_path = 'compressed_datasets/external/machine_translation'
        self.client = self.session.client('s3')
        self.bucket = self.session.resource('s3').Bucket(self._bucket_name)  # noqa # pylint: disable=no-member

    def download_data(self,
                      dataset: str,
                      output: str = '.',
                      overwrite: bool = False
                      ) -> str:
        """
        Utility to method to download and unzip the compressed datasets from our S3 bucket

         Args:
            dataset: The dataset name, which is required.
            output: Directory to put the datasets in.
            overwrite: Determines whether or not to do an overwrite the dataset location locally. If `True`, and a directory exists
                with the name already, we will not attempt to download and unzip.

        Returns:
            Done

        Raises:
            Invalid dataset if the specified dataset is not in the s3 bucket.

        """
        log.debug(f"Data Path: {self._data_path}")
        output_dir = os.path.join(output, 'monolingual_corpora')

        monolingual_datasets = ['wiki-en', 'wiki-ar']
        if dataset not in monolingual_datasets:
            log.info(f"`{dataset}` not in vailable monolingual datasets. Expected one of: {monolingual_datasets}")
            return ''
        else:
            # ugly way of doing this for now, will need to come back to rewrite this api
            # TODO: Fix this script to have a clearer API and less redundant code.
            # ds = [d for d in datasets if d.split('/')[-1].split('.')[0] == dataset][0]
            # self._download_helper(ds, output_dir, overwrite, stage)
            # datasets_downloaded.append(dataset)
            if dataset == 'wiki-en':
                self._download_dataset('wiki-en', f"{self._data_path}/wiki-en-table.gz", output_dir, overwrite)
            elif dataset == 'wiki-ar':
                self._download_dataset('wiki-ar', f"{self._data_path}/wiki-ar-table.gz", output_dir, overwrite)

        return f'Finished downloading: "{dataset}" to\n\t"{os.path.abspath(output_dir)}"'

    def _download_dataset(self, dataset: str, data_path: str, output_dir: str, overwrite: bool) -> None:
        # log.debug(f"Data path: {data_path}")

        output = f'{os.path.join(output_dir, dataset)}.txt'

        # log.info(f"Output Path: {output}")

        if Path(output).exists() and not overwrite:
            log.info(f" `{dataset}` is already downloaded and `overwrite` is set to False. Skipping `{dataset}`\n\t*Note: This does not guarantee the newest version of the dataset...")  # noqa

        else:
            if Path(output).exists():
                log.info(f"Deleting existing {dataset} data...")
                Path(output).unlink()

            if not Path(output_dir).exists():
                Path(output_dir).mkdir(exist_ok=True, parents=True)
            # ········································································ #
            # Download
            # ········································································ #
            # log.info(f"Data Path: {data_path}")
            # log.info(f"Output:: {output}")
            progress = ProgressPercentage(self.bucket, data_path)

            self.bucket.download_file(data_path, f'{output}.gz', Callback=progress)

            # ········································································ #
            # Extract
            # ········································································ #
            log.info(f"Uncompressing file...")
            contents = gzip.open(f'{output}.gz', 'rb').read()

            log.info(f"Writing out file...")
            f = open(output, 'wb')
            f.write(contents)
            f.close()

            log.info("Cleaning up...")
            Path(f'{output}.gz').unlink()
        return


def main() -> None:
    fire.Fire(CLI)


if __name__ == '__main__':

    fire.Fire(CLI)
