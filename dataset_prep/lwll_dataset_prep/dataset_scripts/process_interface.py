from dataclasses import dataclass, asdict
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from pathlib import Path
import requests
from fastprogress.fastprogress import progress_bar
import tarfile
from zipfile import ZipFile
from tarfile import TarInfo
import pandas as pd
from typing import List, Tuple, Optional, Any
import shutil
import json
from lwll_dataset_prep.logger import log
from lwll_dataset_prep.dataset_scripts.aws_cls import s3_operator
from lwll_dataset_prep.dataset_scripts.firebase import fb_store_public, fb_store_private
import lzma
import boto3
from tqdm import tqdm
import os
import sys
import glob

# ========================================================================================== #
# Variables
# ------------------------------------------------------------------------------------------ #
BUCKET_ID = 'lwll-datasets'

@dataclass
class BaseProcesser:
    """
    We define an interface that will act as the processor for all of our datasets
    This is defined so that we can gaurentee functional methods that can be called across
    all datasets at the same time and can write scripts that execute only specific parts of the
    process for subsets of the datasets.
    """
    data_path: Path = Path('/datasets/lwll_datasets')
    labels_path: Path = Path('/datasets/lwll_labels')
    tar_path: Path = Path('/datasets/lwll_compressed_datasets')

    def download(self) -> None:
        raise NotImplementedError

    def process(self) -> None:
        raise NotImplementedError

    def transfer(self) -> None:
        raise NotImplementedError

    def download_data_from_url(self, url: str, dir_name: str,
                               file_name: str, overwrite: bool = False, s3_download: Optional[bool] = False,
                               drive_download: Optional[bool] = False) -> None:
        dir_path = self.data_path.joinpath(dir_name)
        full_path = dir_path.joinpath(file_name)
        if not full_path.exists() or overwrite:
            dir_path.mkdir(parents=True, exist_ok=True)
            if s3_download:
                self.download_s3_url(url, full_path)
            elif drive_download:
                self.download_google_drive_url(url, full_path)
            else:
                self.download_url(url, full_path)
            log.info(f"Finished Downloading `{dir_name}`")
        else:
            log.info(f"`{dir_name}` already exists and `overwrite` is set to `False`. Skipping...")

    def download_url(self, url: str, dest: Path, show_progress: bool = True,
                     chunk_size: int = 1024*1024, timeout: int = 4, retries: int = 5) -> None:

        s = requests.Session()
        s.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
        u = s.get(url, stream=True, timeout=timeout)
        try:
            file_size = int(u.headers["Content-Length"])
        except Exception as e:
            log.info(f'Error: `{e}`')
            show_progress = False

        with open(str(dest), 'wb') as f:
            nbytes = 0
            if show_progress:
                pbar = progress_bar(range(file_size), auto_update=False,
                                    leave=False, parent=None)
                try:
                    for chunk in u.iter_content(chunk_size=chunk_size):
                        nbytes += len(chunk)
                        if show_progress:
                            pbar.update(nbytes)
                        f.write(chunk)
                except requests.exceptions.ConnectionError as e:
                    log.error(f'Error: `{e}`')
                    fname = str(dest).split('/')[-1]
                    p = "/".join(str(dest).split('/')[:-1])
                    timeout_txt = (f'\n Download of {url} has failed after {retries} retries\n'
                                   f' Fix the download manually:\n'
                                   f'$ mkdir -p {p}\n'
                                   f'$ cd {p}\n'
                                   f'$ wget -c {url}\n'
                                   f'$ tar -zxvf {fname}\n\n'
                                   f'And re-run your code once the download is successful\n')
                    log.error(timeout_txt)

    def download_s3_url(self, data_path: str, dest: Path) -> None:
        log.debug(f"Getting s3: {data_path}")
        session = boto3.Session(
            aws_access_key_id='AKIAXNTA46J3YJ6LRKO7',
            aws_secret_access_key='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',
            region_name='us-east-1',
        )
        bucket = session.resource('s3').Bucket(BUCKET_ID)
        bucket.download_file(data_path, str(dest))

    def download_google_drive_url(self, id: str, dest: Path) -> None:
        '''
        Credited to
        https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
        author: https://stackoverflow.com/users/1475331/user115202
        '''
        def get_confirm_token(response: Any) -> Optional[Any]:
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response: Any, destination: Path) -> None:
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            bar.update(CHUNK_SIZE)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        log.debug(f"Getting google drive: {id}")
        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, dest)

    def extract_tar(self, dir_name: str, fname: str) -> None:
        p = self.data_path.joinpath(dir_name)
        f = p.joinpath(fname)
        log.info(f"Extracting tar: `{str(f)}`")
        if 'gz' in fname:
            tarfile.open(str(f), 'r:gz').extractall(str(p))
        else:
            tarfile.open(str(f), 'r:*').extractall(str(p))

    def remove_hidden(self, dir_name: str, fname: str) -> None:
        """
        We had to add this remove hidden files function for the pool_car_detection dataset which had hidden files that
        were being discovered by glob.
        """
        p = self.data_path.joinpath(dir_name)
        f = p.joinpath(fname)
        log.info(f"Remove hidden files from: `{str(f)}`")
        label_files = glob.glob(f'{f}/*/*/*/.*')
        for hidden_file in label_files:
            os.remove(hidden_file)

    def extract_xz(self, dir_name: str, fname: str) -> None:
        p = self.data_path.joinpath(dir_name)
        _file = p.joinpath(fname)
        _to = p.joinpath('.'.join(fname.split('.')[:-1]))
        with lzma.open(str(_file)) as f, open(_to, 'wb') as fout:
            file_content = f.read()
            fout.write(file_content)

    def extract_zip(self, dir_name: str, fname: str) -> None:
        p = self.data_path.joinpath(dir_name)
        f = p.joinpath(fname)
        valid_extensions = ['.zip']
        if f.suffix in valid_extensions:
            log.info(f"Extracting zip: `{str(f)}`")
            with ZipFile(str(f), 'r') as zipObj:
                zipObj.extractall(str(p))

    def setup_folder_structure(self, dir_name: str) -> None:
        log.info("Setting up folder structure...")
        p = self.data_path.joinpath(dir_name)
        for _type in ['sample', 'full']:
            p.joinpath(f'{dir_name}_{_type}').mkdir(parents=True, exist_ok=True)
            p.joinpath(f'{dir_name}_{_type}/train').mkdir(parents=True, exist_ok=True)
            p.joinpath(f'{dir_name}_{_type}/test').mkdir(parents=True, exist_ok=True)

    def create_label_files(self, dir_name: str, df_train: pd.DataFrame, df_test: pd.DataFrame,
                           samples_train: int, samples_test: int, many_to_one: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Takes initial label files and creates a reproducable sample label files
        """
        p = self.labels_path.joinpath(dir_name)
        p.joinpath(f'{dir_name}_sample').mkdir(parents=True, exist_ok=True)
        p.joinpath(f'{dir_name}_full').mkdir(parents=True, exist_ok=True)
        log.info("Creating subsets of full data...")
        df_train_sample = self.create_sample(df_train, samples_train, many_to_one)
        df_test_sample = self.create_sample(df_test, samples_test, many_to_one)

        df_train.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        df_train_sample.reset_index(drop=True, inplace=True)
        df_test_sample.reset_index(drop=True, inplace=True)

        log.info("Saving full dataset labels out...")
        df_train.to_feather(p.joinpath(f'{dir_name}_full/labels_train.feather'))
        df_test.to_feather(p.joinpath(f'{dir_name}_full/labels_test.feather'))
        log.info("Saving sample dataset labels out...")
        df_train_sample.to_feather(p.joinpath(f'{dir_name}_sample/labels_train.feather'))
        df_test_sample.to_feather(p.joinpath(f'{dir_name}_sample/labels_test.feather'))

        # s3_operator.save_df_to_path(df_train, f'{dir_name}_full/labels_train.feather')
        # s3_operator.save_df_to_path(df_test, f'{dir_name}_full/labels_test.feather')
        # s3_operator.save_df_to_path(df_train_sample, f'{dir_name}_sample/labels_train.feather')
        # s3_operator.save_df_to_path(df_test_sample, f'{dir_name}_sample/labels_test.feather')

        return df_train_sample, df_test_sample

    def original_paths_to_destination(self, dir_name: str, orig_paths: List[Path], dest_type: str,
                                      delete_original: bool = True, new_names: Optional[List[str]] = None) -> None:
        """
        new_names argument was added in order to accomodate very dumb format of face_detection dataset where mappings were messed up and
        new ids had to be generated
        """
        p = self.data_path.joinpath(dir_name)
        if dest_type not in ['train', 'test']:
            raise Exception
        log.info(f"Moving images for `{dest_type}` into place...")
        for _idx, _p in enumerate(orig_paths):
            name = _p.name if new_names is None else new_names[_idx]
            dest = p.joinpath(f'{dir_name}_full/{dest_type}/{name}')
            if delete_original:
                shutil.move(str(_p), str(dest))
            else:
                shutil.copy(str(_p), str(dest))

    def create_sample(self, df: pd.DataFrame, n: int = 1000, many_to_one: bool = False, random_state: int = 1) -> pd.DataFrame:
        if not many_to_one:
            _df = df.sample(n=n, random_state=random_state).reset_index(drop=True)
        else:
            # taking care of the sampling case for objecct detection where we have multiple
            # items in the 'id' column that are repeats because there are multiple bounding boxes
            _ids = df['id'].unique().tolist()[:n]
            _df = df[df['id'].isin(_ids)].reset_index(drop=True)
        return _df

    def copy_from_full_to_sample_destination(self, dir_name: str, df_train_sample: pd.DataFrame, df_test_sample: pd.DataFrame) -> None:
        p = self.data_path.joinpath(dir_name)
        train_dir = p.joinpath(f'{dir_name}_full/train')
        test_dir = p.joinpath(f'{dir_name}_full/test')
        train_sample_dir = p.joinpath(f'{dir_name}_sample/train')
        test_sample_dir = p.joinpath(f'{dir_name}_sample/test')

        log.info("Copying original images to sample directory...")
        log.info("Training sample..")
        for img in df_train_sample['id'].tolist():
            shutil.copy(str(train_dir.joinpath(img)), str(train_sample_dir.joinpath(img)))
        log.info("Testing sample..")
        for img in df_test_sample['id'].tolist():
            shutil.copy(str(test_dir.joinpath(img)), str(test_sample_dir.joinpath(img)))

    def save_dataset_metadata(self, dir_name: str, metadata: DatasetDoc) -> None:
        log.info("Saving out dataset.json")
        p = self.labels_path.joinpath(dir_name)
        with open(str(p.joinpath('dataset.json')), 'w') as fp:
            json.dump(asdict(metadata), fp, sort_keys=True, indent=4)

    def push_data_to_cloud(self, dir_name: str, dataset_type: str, task_type: str, is_mt: bool = False) -> None:
        """
        Creates the tar files of the datadir info
        Pushes data tars to S3
        Pushes uncompressed label files to S3
        Pushes metadata files to Firebase
        """

        if dataset_type not in ['external', 'development', 'evaluation']:
            log.error('dataset_type has to be either external, development, or evaluation, exit!')
            sys.exit()

        data_path = self.data_path.joinpath(dir_name)
        self.tar_path.mkdir(parents=True, exist_ok=True)
        tar_path = self.tar_path.joinpath(f'{dir_name}.tar.gz')
        label_path = self.labels_path.joinpath(dir_name)

        if dataset_type == 'development':
            # Step 1: development
            # Make tar from data and upload compressed data
            log.info(f'compressed_datasets/{dataset_type}/{task_type}/{dir_name}.tar.gz')
            log.info('(1/4) Creating compressed data file')
            self._make_tarfile(str(tar_path.absolute()), str(data_path.absolute()))
            log.info('(2/4) Uploading compressed data file to S3')
            s3_operator.multi_part_upload_with_s3(path_from=str(tar_path.absolute()),
                                                  path_to=f'scratch/datasets/{dir_name}.tar.gz')

            # Push up label files
            log.info('(3/4) Uploading label files to S3')
            _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('labels_train.feather').absolute())
            s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/full/labels_train.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('labels_test.feather').absolute())
            s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/full/labels_test.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('labels_train.feather').absolute())
            s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/sample/labels_train.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('labels_test.feather').absolute())
            s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/sample/labels_test.feather')

            if is_mt:
                _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('test_label_ids.feather').absolute())
                s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/full/test_label_ids.feather')

                _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('test_label_ids.feather').absolute())
                s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'scratch/labels/{dir_name}/sample/test_label_ids.feather')

            # Push up datasetdoc to firebase
            log.info('(4/4) Pushing dataset metadata to Firebase')
            with open(str(label_path.joinpath('dataset.json').absolute())) as f:
                meta = json.load(f)

            fb_store_public.collection('DatasetMetadata').document(dir_name).set(meta)
            fb_store_private.collection('DatasetMetadata').document(dir_name).set(meta)

        # Step2: external dataset
        elif dataset_type == 'external':
            log.info(f'compressed_datasets/{dataset_type}/{task_type}/{dir_name}.tar.gz')
            log.info(data_path.joinpath('labels'))
            log.info(label_path.joinpath(f'{dir_name}_full'))
            shutil.copytree(label_path.joinpath(f'{dir_name}_full'), data_path.joinpath('labels'))

            log.info('(1/3) Creating compressed data file')
            self._make_tarfile(str(tar_path.absolute()), str(data_path.absolute()))

            log.info('(2/3) Uploading compressed data file to S3')
            s3_operator.multi_part_upload_with_s3(path_from=str(tar_path.absolute()),
                                                  path_to=f'compressed_datasets/{dataset_type}/{task_type}/{dir_name}.tar.gz')

            log.info('(3/3) Pushing dataset metadata to Firebase')
            with open(str(label_path.joinpath('dataset.json').absolute())) as f:
                meta = json.load(f)

            fb_store_public.collection('DatasetMetadata').document(dir_name).set(meta)
            fb_store_private.collection('DatasetMetadata').document(dir_name).set(meta)

        elif dataset_type == 'evaluation':
            # Step 1: development
            # Make tar from data and upload compressed data
            from lwll_dataset_prep.dataset_scripts.admin_s3_cls import s3_operator as admin_s3_operator

            log.info(f'compressed_datasets/{dataset_type}/{task_type}/{dir_name}.tar.gz')
            log.info('(1/4) Creating compressed data file')
            self._make_tarfile(str(tar_path.absolute()), str(data_path.absolute()))
            log.info('(2/4) Uploading compressed data file to S3')
            admin_s3_operator.multi_part_upload_with_s3(path_from=str(tar_path.absolute()),
                                                        path_to=f'live/datasets/{dir_name}.tar.gz')

            # Push up label files
            log.info('(3/4) Uploading label files to S3')
            _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('labels_train.feather').absolute())
            admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/full/labels_train.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('labels_test.feather').absolute())
            admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/full/labels_test.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('labels_train.feather').absolute())
            admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/sample/labels_train.feather')

            _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('labels_test.feather').absolute())
            admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/sample/labels_test.feather')

            if is_mt:
                _name_from = str(label_path.joinpath(f'{dir_name}_full').joinpath('test_label_ids.feather').absolute())
                admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/full/test_label_ids.feather')

                _name_from = str(label_path.joinpath(f'{dir_name}_sample').joinpath('test_label_ids.feather').absolute())
                admin_s3_operator.multi_part_upload_with_s3(path_from=_name_from, path_to=f'live/labels/{dir_name}/sample/test_label_ids.feather')

            # Push up datasetdoc to firebase
            log.info('(4/4) Pushing dataset metadata to Firebase')
            with open(str(label_path.joinpath('dataset.json').absolute())) as f:
                meta = json.load(f)

            fb_store_public.collection('DatasetMetadata').document(dir_name).set(meta)
            fb_store_private.collection('DatasetMetadata').document(dir_name).set(meta)

        else:
            log.error(f'{dataset_type} not supported, exit!')
            sys.exit()

    def generate_mt_splits(self, df: pd.DataFrame, test_count: int, sample_percent: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generates MT Data splits from a standard MT DataFrame where we have source and target in same DataFrame

        After feedback here: https://gitlab.lollllz.com/lwll/lwll_api/-/issues/82 we only keep 2000 labels for test set.
        We make the choice to keep 2000 for the sample dataset as well and the sample percent is only applied to training data
        """
        df_test_cutoff = len(df) - test_count
        train_full = df.iloc[:df_test_cutoff]
        test_full = df.iloc[df_test_cutoff:]
        train_sample = train_full.iloc[:int(len(train_full) * sample_percent)]
        test_sample = test_full.iloc[0:len(test_full)]  # essentially entire test set from full, keeping explicit indexing to follow train_sample pattern

        # Verify id data type as str
        train_full['id'] = train_full['id'].astype(str)
        test_full['id'] = test_full['id'].astype(str)
        train_sample['id'] = train_sample['id'].astype(str)
        test_sample['id'] = test_sample['id'].astype(str)

        return train_full, test_full, train_sample, test_sample

    def push_to_dynamo(self, df: pd.DataFrame, dataset_name: str, index_col: str, target_col: str, size_col: str) -> None:
        """
        Pushes a DataFrame to Dynamo with target_col as the text
        """
        session = boto3.Session(profile_name='jpl-sso', region_name='us-east-1')
        dynamodb = session.client('dynamodb')
        MAX_DYNAMO_PUT = 25

        data = df.to_dict(orient='records')
        # TODO: Should do appropriate error handling on each of these responses
        # responses = []
        for i in tqdm(range(int(len(data)/MAX_DYNAMO_PUT) + 1)):
            chunk = data[MAX_DYNAMO_PUT*i:MAX_DYNAMO_PUT*(i+1)]
            put_requests = [{"PutRequest": {
                "Item":
                {
                    'datasetid_sentenceid': {"S": f"{dataset_name}_{str(item[index_col])}"},
                    'target': {"S": item[target_col]},
                    'size': {"N": f"{item[size_col]}"},
                }
            }
            }
                for item in chunk]
            _ = dynamodb.batch_write_item(RequestItems={
                'machine_translation_target_lookup': put_requests
            })
            # responses.append(response)

        return

    def _make_tarfile(self, tar_path: str, path_from: str) -> None:
        def _no_compressed(thing: TarInfo) -> Optional[TarInfo]:
            bad_extensions = ['tar.gz', 'tar']
            suffix = ".".join(thing.name.split(".")[1:])
            if suffix in bad_extensions:
                return None
            else:
                return thing

        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(path_from, arcname=os.path.basename(path_from), filter=_no_compressed)

    def validate_output_structure(self) -> None:
        # TODO: Should write validation logic that checks the output schema
        # For the dataset and assures we have all required directories / formats / fields
        pass
