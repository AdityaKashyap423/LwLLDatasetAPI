from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import pandas as pd
import shutil
from lwll_dataset_prep.logger import log
import math

@dataclass
class face_detection(BaseProcesser):
    """
    Source data: http://vis-www.cs.umass.edu/fddb/index.html
    5171 faces in a set of 2845 images


    Our source data is an object detection task with various images in a folder structure
    with 4 levels and then the image such as:
        - 2002/20/04/big/img_417.jpg

    The original labels file is formatted as such:
    <image_path>
    <number of annotations for this image>
    <major_axis_radius minor_axis_radius angle center_x center_y 1>
    ...

    So a couple of copied examples would look something like this:

    2002/08/11/big/img_591
    1
    123.583300 85.549500 1.265839 269.693400 161.781200  1
    2002/08/26/big/img_265
    3
    67.363819 44.511485 -1.476417 105.249970 87.209036  1
    41.936870 27.064477 1.471906 184.070915 129.345601  1

    Because of this, we need to convert the ellipse format to a bounding box format:

    For LwLL we decide on a standard of bounding box coordinates to be:
    (x_min, y_min, x_max, y_max) --> which is essentially the top left coordinate followed by the bottom right coordinate

    We can accomplish this with the following formula:

    ux = major_axis_r * cos(angle)
    uy = major_axis_r * sin(angle)
    vx = minor_axis_r * cos(angle + PI/2)
    vy = minor_axis_r * sin(angle + PI/2)

    bbox_halfwidth = sqrt(ux*ux + vx*vx)
    bbox_halfheight = sqrt(uy*uy + vy*vy)

    x_min = center_x - bbox_halfwidth
    y_min = center_y - bbox_halfheight
    x_max = center_x + bbox_halfwidth
    y_max = center_y + bbox_halfheight


    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'face_detection'
    _task_type: str = 'object_detection'
    _urls: List[str] = field(default_factory=lambda: ['http://tamaraberg.com/faceDataset/originalPics.tar.gz',
                                                      'http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'])
    _sample_size_train: int = 500
    _sample_size_test: int = 150
    _valid_extensions: List[str] = field(default_factory=lambda: ['.png', '.jpg'])

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
        print(self._fnames)
        for fname in self._fnames:
            self.extract_tar(dir_name=self._path_name, fname=fname)

        self.data_path.joinpath(self._path_name).joinpath('originalPics').mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.data_path.joinpath(self._path_name).joinpath('2002')), str(
            self.data_path.joinpath(self._path_name).joinpath('originalPics').joinpath('2002')))
        shutil.move(str(self.data_path.joinpath(self._path_name).joinpath('2003')), str(
            self.data_path.joinpath(self._path_name).joinpath('originalPics').joinpath('2003')))

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

        # IMPORTANT NOTE THAT'S NOT MENTIONED IN THIS DATASET README
        # THERE ARE 28204 UNIQUE PATHS AND 3539 UNIQUE NAMES HERE, THIS IS MORE THAN DATA THAT IS
        # ANNOTATED, FOR THIS REASON, WE WILL SELECT A SUBSET OF THE IMAGES WITH ANNOATIONS FURTHER BELOW...

        label_files = [_p for _p in self.path.joinpath('FDDB-folds').iterdir() if 'ellipse' in str(_p)]
        data_dict: dict = {}
        for file in label_files:
            _d = self._read_label_file(file)
            data_dict = {**data_dict, **_d}

        split_idx = int(len(data_dict.keys()) * 0.8)
        img_paths = list(data_dict.keys())
        paths_train = img_paths[:split_idx]
        paths_test = img_paths[split_idx:]

        # Now our data_dict has key of file path string and values of an array of the ellipse coordinates
        # print(data_dict)
        training_ids = []
        training_boxes = []
        testing_ids = []
        testing_boxes = []

        # print("DAta dict itesms")
        # print(len(data_dict.keys()))
        # We need to create new ids in this special scenario because their are repeat base image names
        # for different paths..... WHY?!?!?!? WHY WOULD SOMEONE DO THAT?!?!??
        pth_to_id = {}
        id_to_pth = {}
        cnt = 0
        for key, val in data_dict.items():
            cnt += 1
            new_id = f'img_{cnt}.jpg'
            pth_to_id[key] = new_id
            id_to_pth[new_id] = key
            train = True
            if key in paths_train:
                train = True
            elif key in paths_test:
                train = False
            else:
                print(key)
                raise Exception("Didn't find image id")

            for ellipse in val:
                maj_r = float(ellipse[0])
                min_r = float(ellipse[1])
                angle = float(ellipse[2])
                center_x = float(ellipse[3])
                center_y = float(ellipse[4])
                b_box = str(self._ellipse_to_bonding_box(maj_r, min_r, angle, center_x, center_y))

                if train:
                    training_ids.append(new_id)
                    training_boxes.append(b_box)
                else:
                    testing_ids.append(new_id)
                    testing_boxes.append(b_box)

        # print(len(training_ids))
        # print(len(testing_ids))
        # print(training_ids[:3])
        # print(training_boxes[:3])

        # Create our data schema
        df_train = pd.DataFrame({'id': training_ids, 'bbox': training_boxes, 'class': ['face' for _ in range(len(training_boxes))]})
        df_test = pd.DataFrame({'id': testing_ids, 'bbox': testing_boxes, 'class': ['face' for _ in range(len(testing_boxes))]})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test, many_to_one=True)
        # print("lengs")
        # print(len(df_train['id'].unique()))
        # print(len(df_test['id'].unique()))
        # print(len(df_train_sample['id'].unique()))
        # print(len(df_test_sample['id'].unique()))

        # Move the raw data files
        # print(df_traindf_train['id'].unique().tolist())
        orig_train_paths = [self.path.joinpath('originalPics').joinpath(id_to_pth[_id] + '.jpg') for _id in df_train['id'].unique().tolist()]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train',
                                           delete_original=False, new_names=df_train['id'].unique().tolist())

        orig_test_paths = [self.path.joinpath('originalPics').joinpath(id_to_pth[_id] + '.jpg') for _id in df_test['id'].unique().tolist()]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test',
                                           delete_original=False, new_names=df_test['id'].unique().tolist())

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type=self._task_type,
                                 sample_number_of_samples_train=self._sample_size_train,
                                 sample_number_of_samples_test=self._sample_size_test,
                                 sample_number_of_classes=1,
                                 full_number_of_samples_train=len(df_train),
                                 full_number_of_samples_test=len(df_test),
                                 full_number_of_classes=1,
                                 number_of_channels=3,
                                 classes=['face'],
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='http://vis-www.cs.umass.edu/fddb/',
                                 license_requirements='None',
                                 license_citation='@TechReport{fddbTech,author = {Vidit Jain and Erik Learned-Miller},title =  {FDDB: A Benchmark for Face Detection in Unconstrained Settings},institution =  {University of Massachusetts, Amherst},year = {2010},number = {UM-CS-2010-009}}',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        for f in self._fnames:
            # We assume the tar files have no '.'s in their name before `.tar.gz` or just `.tar`
            d = f.split('.')[0]
            shutil.rmtree(self.path.joinpath(d))

        log.info("Done")

        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='external', task_type=self._task_type)
        log.info("Done")

    def _read_label_file(self, file: Path) -> dict:
        _file: str = str(file)
        data: dict = {}
        with open(_file) as fp:
            line = fp.readline()
            img_path = line.strip()
            data[img_path] = []
            while line:
                if len(line.strip().split('/')) > 1:
                    # Scenario we have the file path
                    img_path = line.strip()
                    data[img_path] = []
                elif len(line.strip().split(' ')) > 1:
                    # Scenario we have a bounding ellipse
                    data[img_path].append(line.strip().split(' '))
                line = fp.readline()
        return data

    def _ellipse_to_bonding_box(self, maj_r: float, min_r: float, angle: float, center_x: float, center_y: float) -> str:
        """
        Converting the default ellipse format to the lwll format bounding box
        """

        ux = maj_r * math.cos(angle)
        uy = maj_r * math.sin(angle)
        vx = min_r * math.cos(angle + math.pi/2)
        vy = min_r * math.sin(angle + math.pi/2)

        bbox_halfwidth = math.sqrt(ux*ux + vx*vx)
        bbox_halfheight = math.sqrt(uy*uy + vy*vy)

        x_min = center_x - bbox_halfwidth
        y_min = center_y - bbox_halfheight
        x_max = center_x + bbox_halfwidth
        y_max = center_y + bbox_halfheight

        assert x_min < x_max
        assert y_min < y_max

        return f"{int(x_min)}, {int(y_min)}, {int(x_max)}, {int(y_max)}"
