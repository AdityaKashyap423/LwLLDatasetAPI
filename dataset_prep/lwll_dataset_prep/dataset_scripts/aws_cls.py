import pandas as pd
from lwll_dataset_prep.logger import log
import boto3
from io import BytesIO
from pyarrow.feather import write_feather

import threading
import os
import sys
from boto3.s3.transfer import TransferConfig

class ProgressPercentage(object):
    def __init__(self, filename: str) -> None:
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount: int) -> None:
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


class S3_cls(object):

    def __init__(self) -> None:
        self.bucket_name = 'lwll-datasets'
        self.session = boto3.Session(profile_name='lwll_creds')
        self.s3 = self.session.client('s3')

    def read_path(self, path: str) -> pd.DataFrame:
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=path)
            df = pd.read_feather(obj['Body'])
            return df
        except FileNotFoundError as e:
            log.error(e)
        return

    def save_df_to_path(self, df: pd.DataFrame, path: str) -> None:
        with BytesIO() as f:
            write_feather(df, f)
            self.s3.Object(self.bucket_name, path).put(Body=f.getvalue())

    def multi_part_upload_with_s3(self, path_from: str, path_to: str) -> None:
        # Multipart upload
        config = TransferConfig(multipart_threshold=1024 * 25, max_concurrency=10,
                                multipart_chunksize=1024 * 25, use_threads=True)
        # file_path = os.path.dirname(__file__) + '/largefile.pdf'
        # key_path = 'multipart_files/largefile.pdf'

        self.s3.upload_file(path_from, self.bucket_name, path_to,
                            Config=config,
                            Callback=ProgressPercentage(path_from)
                            )


s3_operator = S3_cls()
