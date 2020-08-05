from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import pandas as pd
import os
import shutil
from lwll_dataset_prep.logger import log

@dataclass
class domain_net(BaseProcesser):
    """
    Our source data has six distinct domains and is in the form:
    clipart
    - zigzag
    - - img1.png
    - - img2.png
    - - etc.
    - zebra
    - - imgx.png
    - - imgy.png
    - - etc.
        ...
    - airplane
    - - imgx.png
    - - imgx.png
    - - etc.
    quickdraw
    - zigzag
    - - img1.png
    - - img2.png
    - - etc.
    - zebra
    - - imgx.png
    - - imgy.png
    - - etc.
        ...
    - airplane
    - - imgx.png
    - - imgx.png
    - - etc.
    quickdraw_train.txt
    etc.

    0.6M images of 345 object categories from 6 domains: Clipart, Infograph, Painting, Quickdraw, Photo, Sketch

    We will transform this into our LwLL format for image problems (See this repo's README for that format)

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _task_type: str = "image_classification"
    _path_name: str = 'domain_net'
    _urls: List[str] = field(default_factory=lambda: ['http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_test.txt'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]

    def download(self) -> None:
        # Download
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=False)
            log.info("Done")

    def process(self) -> None:
        # todo: add more comments
        # Extract only the zip files
        for fname in self._fnames:
            self.extract_zip(dir_name=self._path_name, fname=fname)

        domains = set([_p for _p in self.path.iterdir() if _p.suffix not in ['.zip', '.txt'] and os.path.isdir(_p)])
        print(domains)

        all_classes = ['lollipop',
                       'apple',
                       'diamond',
                       'helmet',
                       'skull',
                       'palm_tree',
                       'lipstick',
                       'cat',
                       'rhinoceros',
                       'peanut',
                       'animal_migration',
                       'pond',
                       'ant',
                       'fire_hydrant',
                       'jacket',
                       'blueberry',
                       'microwave',
                       'remote_control',
                       'tree',
                       'paintbrush',
                       'butterfly',
                       'see_saw',
                       'crown',
                       'leaf',
                       'boomerang',
                       'drill',
                       'toaster',
                       'lightning',
                       'toe',
                       'garden_hose',
                       'sword',
                       'fork',
                       'pear',
                       'hand',
                       'fireplace',
                       'sandwich',
                       'strawberry',
                       'raccoon',
                       'bench',
                       'ice_cream',
                       'piano',
                       'basket',
                       'chandelier',
                       'elbow',
                       'sun',
                       'cactus',
                       'car',
                       'crab',
                       'cello',
                       'peas',
                       'pig',
                       'hot_air_balloon',
                       'tractor',
                       'hammer',
                       'ocean',
                       'canoe',
                       'screwdriver',
                       'river',
                       'feather',
                       'snail',
                       'eye',
                       'bed',
                       'violin',
                       'golf_club',
                       'tooth',
                       'diving_board',
                       'yoga',
                       'hockey_stick',
                       'rain',
                       'cup',
                       'calendar',
                       'stereo',
                       'radio',
                       'angel',
                       'trombone',
                       'snowman',
                       'sweater',
                       'microphone',
                       'aircraft_carrier',
                       'calculator',
                       'camouflage',
                       'shovel',
                       'string_bean',
                       'television',
                       'hourglass',
                       'saw',
                       'rollerskates',
                       'bottlecap',
                       'steak',
                       'donut',
                       'eraser',
                       'mushroom',
                       'squiggle',
                       'stethoscope',
                       'rifle',
                       'dog',
                       'alarm_clock',
                       'clarinet',
                       'bee',
                       'belt',
                       'face',
                       'couch',
                       'foot',
                       'spreadsheet',
                       'dolphin',
                       'soccer_ball',
                       'scorpion',
                       'postcard',
                       'onion',
                       'garden',
                       'candle',
                       'speedboat',
                       'birthday_cake',
                       'giraffe',
                       'bear',
                       'grass',
                       'flower',
                       'harp',
                       'potato',
                       'bridge',
                       'mailbox',
                       'penguin',
                       'zebra',
                       'camera',
                       'drums',
                       'underwear',
                       'swing_set',
                       'moustache',
                       'baseball',
                       'sheep',
                       'tennis_racquet',
                       'square',
                       'panda',
                       'mosquito',
                       'lobster',
                       'duck',
                       'cruise_ship',
                       'shoe',
                       'moon',
                       'trumpet',
                       'church',
                       'camel',
                       'owl',
                       'tiger',
                       'rake',
                       'blackberry',
                       'lantern',
                       'firetruck',
                       'van',
                       'streetlight',
                       'whale',
                       'stitches',
                       'power_outlet',
                       'oven',
                       'crayon',
                       'crocodile',
                       'guitar',
                       'chair',
                       'wheel',
                       'sink',
                       'windmill',
                       'helicopter',
                       'bus',
                       'headphones',
                       'dishwasher',
                       'triangle',
                       'dresser',
                       'The_Great_Wall_of_China',
                       'picture_frame',
                       'matches',
                       'ladder',
                       'ceiling_fan',
                       'nose',
                       'mouth',
                       'The_Eiffel_Tower',
                       'snowflake',
                       'sailboat',
                       'key',
                       'motorbike',
                       'hexagon',
                       'snorkel',
                       'hot_dog',
                       'basketball',
                       'ambulance',
                       'vase',
                       'light_bulb',
                       'zigzag',
                       'submarine',
                       'megaphone',
                       'watermelon',
                       'beard',
                       'passport',
                       'police_car',
                       'cell_phone',
                       'telephone',
                       'tent',
                       'mouse',
                       'ear',
                       'smiley_face',
                       'hockey_puck',
                       'saxophone',
                       'pants',
                       'frying_pan',
                       'bowtie',
                       'toilet',
                       'roller_coaster',
                       'tornado',
                       'stove',
                       'envelope',
                       'teddy-bear',
                       'star',
                       'hospital',
                       'pillow',
                       't-shirt',
                       'house_plant',
                       'map',
                       'truck',
                       'campfire',
                       'barn',
                       'traffic_light',
                       'bucket',
                       'bird',
                       'parachute',
                       'wristwatch',
                       'cooler',
                       'hot_tub',
                       'sock',
                       'shorts',
                       'line',
                       'table',
                       'waterslide',
                       'grapes',
                       'octagon',
                       'fence',
                       'skyscraper',
                       'parrot',
                       'nail',
                       'airplane',
                       'kangaroo',
                       'skateboard',
                       'cloud',
                       'mug',
                       'book',
                       'rainbow',
                       'leg',
                       'dragon',
                       'syringe',
                       'sleeping_bag',
                       'suitcase',
                       'train',
                       'jail',
                       'umbrella',
                       'house',
                       'spider',
                       'coffee_cup',
                       'binoculars',
                       'broom',
                       'brain',
                       'monkey',
                       'flashlight',
                       'eyeglasses',
                       'broccoli',
                       'spoon',
                       'bread',
                       'lighthouse',
                       'circle',
                       'hat',
                       'rabbit',
                       'scissors',
                       'mermaid',
                       'bathtub',
                       'cookie',
                       'compass',
                       'asparagus',
                       'school_bus',
                       'bat',
                       'washing_machine',
                       'bush',
                       'fan',
                       'knee',
                       'sea_turtle',
                       'cannon',
                       'banana',
                       'swan',
                       'octopus',
                       'beach',
                       'wine_bottle',
                       'axe',
                       'floor_lamp',
                       'castle',
                       'hamburger',
                       'backpack',
                       'toothpaste',
                       'bicycle',
                       'arm',
                       'frog',
                       'laptop',
                       'mountain',
                       'paint_can',
                       'marker',
                       'hurricane',
                       'lighter',
                       'paper_clip',
                       'computer',
                       'wine_glass',
                       'hedgehog',
                       'anvil',
                       'purse',
                       'pizza',
                       'flying_saucer',
                       'The_Mona_Lisa',
                       'toothbrush',
                       'horse',
                       'stop_sign',
                       'popsicle',
                       'pool',
                       'flamingo',
                       'fish',
                       'stairs',
                       'pineapple',
                       'squirrel',
                       'goatee',
                       'bracelet',
                       'finger',
                       'cow',
                       'baseball_bat',
                       'pickup_truck',
                       'pencil',
                       'teapot',
                       'keyboard',
                       'cake',
                       'pliers',
                       'lion',
                       'clock',
                       'bulldozer',
                       'necklace',
                       'carrot',
                       'flip_flops',
                       'shark',
                       'door',
                       'snake',
                       'knife',
                       'elephant',
                       'bandage',
                       'dumbbell']
        # create dirs for all
        self.setup_folder_structure(dir_name=f"{self._path_name}-all")
        global_training_ids = []
        global_training_class = []
        global_testing_ids = []
        global_testing_class = []

        for domain_path in domains:
            domain = os.path.basename(domain_path)
            self.setup_folder_structure(dir_name=f"{self._path_name}-{domain}")

            training_ids = []
            training_class = []
            testing_ids = []
            testing_class = []

            with open(self.path.joinpath(f'{domain}_train.txt')) as f:
                for line in f:
                    id_path = line.split()[0]
                    training_ids.append(id_path.split('/')[-1])
                    training_class.append(id_path.split('/')[-2])

            with open(self.path.joinpath(f'{domain}_test.txt')) as f:
                for line in f:
                    id_path = line.split()[0]
                    testing_ids.append(id_path.split('/')[-1])
                    testing_class.append(id_path.split('/')[-2])

            # Create our data schema
            df_train = pd.DataFrame({'id': training_ids, 'class': training_class})
            df_test = pd.DataFrame({'id': testing_ids, 'class': testing_class})

            sample_size_train = int(len(df_train) * 0.1)
            sample_size_test = int(len(df_test) * 0.1)
            df_train_sample, df_test_sample = self.create_label_files(dir_name=f"{self._path_name}-{domain}", df_train=df_train,
                                                                      df_test=df_test,
                                                                      samples_train=sample_size_train,
                                                                      samples_test=sample_size_test)

            orig_train_paths = []
            orig_test_paths = []
            for index, row in df_train.iterrows():
                orig_train_paths.append(self.path.joinpath(f'{domain}/{row["class"]}/{row["id"]}'))

            for index, row in df_test.iterrows():
                orig_test_paths.append(self.path.joinpath(f'{domain}/{row["class"]}/{row["id"]}'))

            self.original_paths_to_destination(dir_name=f"{self._path_name}-{domain}", orig_paths=orig_train_paths, dest_type='train', delete_original=False)
            self.original_paths_to_destination(dir_name=f"{self._path_name}-{domain}", orig_paths=orig_test_paths, dest_type='test', delete_original=False)

            # copy images to domain_net-all/all-full/
            self.original_paths_to_destination(dir_name=f"{self._path_name}-all", orig_paths=orig_train_paths, dest_type='train', delete_original=False)
            self.original_paths_to_destination(dir_name=f"{self._path_name}-all", orig_paths=orig_test_paths, dest_type='test', delete_original=False)

            self.copy_from_full_to_sample_destination(dir_name=f"{self._path_name}-{domain}", df_train_sample=df_train_sample,
                                                      df_test_sample=df_test_sample)

            # Create our `dataset.json` metadata file
            dataset_doc = DatasetDoc(name=f"{self._path_name}-{domain}",
                                     dataset_type='image_classification',
                                     sample_number_of_samples_train=sample_size_train,
                                     sample_number_of_samples_test=sample_size_test,
                                     sample_number_of_classes=345,
                                     full_number_of_samples_train=len(df_train),
                                     full_number_of_samples_test=len(df_test),
                                     full_number_of_classes=345,
                                     number_of_channels=3,
                                     classes=all_classes,
                                     language_from=None,
                                     language_to=None,
                                     sample_total_codecs=None,
                                     full_total_codecs=None,
                                     license_link='http://ai.bu.edu/M3SDA/',
                                     license_requirements='None',
                                     license_citation='{{@article{peng2018moment,title={Moment Matching for Multi-Source Domain Adaptation},author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},journal={arXiv preprint arXiv:1812.01754},year={2018}}}}',  # noqa
                                     )

            self.save_dataset_metadata(dir_name=f"{self._path_name}-{domain}", metadata=dataset_doc)

            # get training/testing id/class for all
            global_training_ids.extend(training_ids)
            global_training_class.extend(training_class)
            global_testing_ids.extend(testing_ids)
            global_testing_class.extend(testing_class)

        print(len(global_training_ids), len(global_training_class), len(global_testing_ids), len(global_testing_class))
        # Create data schema for all
        df_train = pd.DataFrame({'id': global_training_ids, 'class': global_training_class})
        df_test = pd.DataFrame({'id': global_testing_ids, 'class': global_testing_class})
        sample_size_train = int(len(df_train) * 0.1)
        sample_size_test = int(len(df_test) * 0.1)
        df_train_sample, df_test_sample = self.create_label_files(dir_name=f"{self._path_name}-all", df_train=df_train,
                                                                  df_test=df_test,
                                                                  samples_train=sample_size_train,
                                                                  samples_test=sample_size_test)
        self.copy_from_full_to_sample_destination(dir_name=f"{self._path_name}-all", df_train_sample=df_train_sample,
                                                  df_test_sample=df_test_sample)

        dataset_doc = DatasetDoc(name=f"{self._path_name}-all",
                                 dataset_type='image_classification',
                                 sample_number_of_samples_train=sample_size_train,
                                 sample_number_of_samples_test=sample_size_test,
                                 sample_number_of_classes=345,
                                 full_number_of_samples_train=len(df_train),
                                 full_number_of_samples_test=len(df_test),
                                 full_number_of_classes=345,
                                 number_of_channels=3,
                                 classes=all_classes,
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='http://ai.bu.edu/M3SDA/',
                                 license_requirements='None',
                                 license_citation='{{@article{peng2018moment,title={Moment Matching for Multi-Source Domain Adaptation},author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},journal={arXiv preprint arXiv:1812.01754},year={2018}}}}',  # noqa
                                 )

        self.save_dataset_metadata(dir_name=f"{self._path_name}-all", metadata=dataset_doc)

        # Cleanup zip extraction intermediate
        log.info("Cleaning up extracted zip copy..")
        for _f in self._fnames:
            d = _f.split('.')[0]
            shutil.rmtree(self.path.joinpath(d))

        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", "all"]
        for domain in domains:
            print(f"{self._path_name}-{domain}")
            self.push_data_to_cloud(dir_name=f"{self._path_name}-{domain}", dataset_type='development', task_type=self._task_type)
        log.info("Done")
