from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
import shutil
from lwll_dataset_prep.logger import log
import os
import random
import xml.etree.ElementTree as ET
import glob

@dataclass
class imagenet_bbox(BaseProcesser):
    """
    Number of images: 7818

    3000 categories in total:
    ex. kit fox, croquette, airplane, frog

        Our source data is in the form:
    imagenet_1k
    - train
    - - n01440764
    - - - n01440764_10026.JPEG
    - - - n01440764_10027.JPEG
    - - - etc.
    - - n01443537
    - - - n01443537_10007.JPEG
    - - - n01443537_10014.JPEG
    - - - etc.
        ...
    - - n01484850
    - - - n01484850_10016.JPEG
    - - - n01484850_10036.JPEG
    - - - etc.
    - test (same as training)

    1,281,167 in training folder and 50,000 in testing folder

    We will transform this into our LwLL format for image problems (See this repo's README for that format)

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'imagenet_bbox'
    _urls: List[str] = field(default_factory=lambda: ['http://www.image-net.org/archive/stanford/fall11_whole.tar',
                                                      'http://www.image-net.org/Annotation/Annotation.tar.gz'])
    _task_type: str = 'object_detection'
    _sample_size_train: int = 20000
    _sample_size_test: int = 100
    _valid_extensions: List[str] = field(default_factory=lambda: ['.JPEG'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

    def download(self) -> None:
        # Download
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=True)
            log.info("Done")

    def process(self) -> None:
        # Extract the tar
        for fname in self._fnames:
            self.extract_tar(dir_name=self._path_name, fname=fname)

        synets_tar = glob.glob(f'{self.path}/*.gz')

        for synet in synets_tar:
            self.extract_tar(dir_name=self._path_name, fname=os.path.basename(synet))

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

        # Iterate all label files to find available images
        label_files, l_to_img = self._find_path()

        # shuffle the label_files
        random.shuffle(label_files)

        classes = set()

        split_idx = int(len(label_files) * 0.8)
        paths_train = label_files[:split_idx]
        paths_test = label_files[split_idx:]

        training_ids = []
        training_boxes = []
        training_cates = []
        testing_ids = []
        testing_boxes = []
        testing_cates = []
        orig_train_paths = []
        orig_test_paths = []

        for p in paths_train:
            orig_train_paths.append(l_to_img[p])
            tree = ET.parse(str(p))
            root = tree.getroot()
            '''
            <annotation>
                <folder>n03376595</folder>
                <filename>n03376595_10035</filename>
                <source>
                    <database>ImageNet database</database>
                </source>
                <size>
                    <width>500</width>
                    <height>375</height>
                    <depth>3</depth>
                </size>
                <segmented>0</segmented>
                <object>
                    <name>n03376595</name>
                    <pose>Unspecified</pose>
                    <truncated>0</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>10</xmin>
                        <ymin>26</ymin>
                        <xmax>455</xmax>
                        <ymax>374</ymax>
                    </bndbox>
                </object>
            </annotation>
            '''
            fname_elt = self._find(root, 'filename')
            fname = f'{self._text(fname_elt)}.JPEG'

            for obj in root.findall('object'):
                cls_elt = self._find(obj, 'name')
                cls = self._text(cls_elt)
                bbox = self._find(obj, 'bndbox')

                x_max_elt = self._find(bbox, 'xmax')
                x_min_elt = self._find(bbox, 'xmin')
                y_max_elt = self._find(bbox, 'ymax')
                y_min_elt = self._find(bbox, 'ymin')

                x_max = self._text(x_max_elt)
                x_min = self._text(x_min_elt)
                y_max = self._text(y_max_elt)
                y_min = self._text(y_min_elt)
                classes.add(cls)
                training_cates.append(cls)
                training_boxes.append(f"{int(float(x_min))}, {int(float(y_min))}, {int(float(x_max))}, {int(float(y_max))}")
                training_ids.append(fname)

        for p in paths_test:
            orig_test_paths.append(l_to_img[p])
            tree = ET.parse(str(p))
            root = tree.getroot()

            fname_elt = self._find(root, 'filename')
            fname = f'{self._text(fname_elt)}.JPEG'

            for obj in root.findall('object'):
                cls_elt = self._find(obj, 'name')
                cls = self._text(cls_elt)
                bbox = self._find(obj, 'bndbox')

                x_max_elt = self._find(bbox, 'xmax')
                x_min_elt = self._find(bbox, 'xmin')
                y_max_elt = self._find(bbox, 'ymax')
                y_min_elt = self._find(bbox, 'ymin')

                x_max = self._text(x_max_elt)
                x_min = self._text(x_min_elt)
                y_max = self._text(y_max_elt)
                y_min = self._text(y_min_elt)

                testing_cates.append(cls)
                testing_boxes.append(f"{int(float(x_min))}, {int(float(y_min))}, {int(float(x_max))}, {int(float(y_max))}")
                testing_ids.append(fname)

        # Create our data schema
        df_train = pd.DataFrame({'id': training_ids, 'bbox': training_boxes, 'class': training_cates})
        df_test = pd.DataFrame({'id': testing_ids, 'bbox': testing_boxes, 'class': testing_cates})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test, many_to_one=True)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train', delete_original=False)

        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test', delete_original=False)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='object_detection',
                                 sample_number_of_samples_train=self._sample_size_train,
                                 sample_number_of_samples_test=self._sample_size_test,
                                 sample_number_of_classes=len(list(classes)),
                                 full_number_of_samples_train=len(df_train),
                                 full_number_of_samples_test=len(df_test),
                                 full_number_of_classes=len(list(classes)),
                                 number_of_channels=3,
                                 classes=list(classes),
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='http://image-net.org/download-faq',
                                 license_requirements='agree_to_termsLicense Link at http://image-net.org/download-faq',
                                 license_citation='N/A',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        for f in self._fnames:
            # We assume the tar files have no '.'s in their name before `.tar.gz` or just `.tar`
            d = f.split('.')[0]
            shutil.rmtree(self.path.joinpath(d))

        for synet in synets_tar:
            os.remove(synet)

        shutil.rmtree(self.path.joinpath("ILSVRC2012_img_train"))
        shutil.rmtree(self.path.joinpath("ILSVRC2012_img_val"))
        shutil.rmtree(self.path.joinpath("Annotation"))

        log.info("Done")

        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")

    def _find(self, element: ET.Element, tag: str) -> ET.Element:
        result = element.find(tag)
        assert result is not None, ('No tag "{tag}" found '
                                    'in element "{element}".'
                                    .format(tag=tag,
                                            element=element))
        return result

    def _text(self, element: ET.Element) -> str:
        result = element.text
        assert result is not None, ('No tag "text" found '
                                    'in element "{element}".'
                                    .format(element=element))
        return result

    def _find_path(self) -> Tuple[List[str], Dict[str, Path]]:
        label_files: List[str] = []
        l_to_img = {}
        for _class in [os.path.basename(c) for c in glob.glob(os.path.join(self.path, "ILSVRC2012_img_train/*"))]:
            files = (glob.glob(f'{self.path.joinpath("Annotation")}/{_class}/*.xml'))
            for p in files:
                tree = ET.parse(str(p))
                root = tree.getroot()
                folder_elt = self._find(root, 'folder')
                dname = f'{self._text(folder_elt)}'

                fname_elt = self._find(root, 'filename')
                fname = f'{self._text(fname_elt)}.JPEG'

                train_path = self.path.joinpath(f'ILSVRC2012_img_train/{dname}/{fname}')
                test_path = self.path.joinpath(f'ILSVRC2012_img_val/{dname}/{fname}')

                if train_path.exists():
                    l_to_img[p] = train_path
                    label_files.append(p)
                elif test_path.exists():
                    l_to_img[p] = test_path
                    label_files.append(p)
                else:
                    continue
        return label_files, l_to_img
