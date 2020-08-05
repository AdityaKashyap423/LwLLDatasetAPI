#!/usr/bin/env python
# ========================================================================================== #
"""
This is a small utility that we can call from command line in order to easily test and process datasets one at a time.

The location of the output is meant to mirror exactly the DMC setup where we save out to the mounted drive at location `/datasets/lwll_datasets`.
You can test locally on your own machine, but in order to do so, you should create the directory and apply appropriate
permissions since it's out of your user directory.
    - sudo mkdir /datasets
    - sudo chmod 777 /datasets


Usage:
    - python proc.py --dataset mnist --download --process --transfer

Output:
    - Should result in going through the process and putting the data content into the location `/datasets/lwll_datasets`

"""
# ========================================================================================== #
# Imports
# ------------------------------------------------------------------------------------------ #

import argparse
from pathlib import Path
import sys
import importlib


# ========================================================================================== #
# Helpers
# ------------------------------------------------------------------------------------------ #

def valid_dataset_file_name(p: Path) -> bool:
    if p.is_dir() and p.name != 'dataset_scripts' and p.name[:2] != '__':
        return True
    else:
        return False

# ========================================================================================== #
# Call / Runner
# ------------------------------------------------------------------------------------------ #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--dataset', required=True, help='Dataset name or folder path in S3.')
    parser.add_argument('--download', '-d', action='store_true', default=False, help='Toggle download operation ("--download" or "-d").')
    parser.add_argument('--process', '-p', action='store_true', default=False, help='Toggle data processing operation ("--process" or "-p").')
    parser.add_argument('--transfer', '-t', action='store_true', default=False, help='Toggle transfer operation ("--transfer" or "-t").')
    # parser.add_argument('--stage', '-s', default=None, choices=[None, 'dev', 'develop', 'eval', 'external'], help='Dataset segment (aka. types or sets) \
    #                                                                                 for different algorithm training stages ("--stage", "-s").')
    args = parser.parse_args()

    print(f"Starting Dataset Process...\nDATASET: {args.dataset}\nDOWNLOAD: {args.download}\nPROCESS: {args.process}\nTRANSFER: {args.transfer}\n\n")

    # Validate the arguments and the dataset name
    if not any(x for x in [args.download, args.process, args.transfer]):
        args.download, args.process, args.transfer = True, True, True

    valid_datasets = [p.name for p in Path('lwll_dataset_prep').iterdir() if valid_dataset_file_name(p)]

    if args.dataset not in valid_datasets:
        print(f'Invalid dataset name, not in list: `{valid_datasets}`\nExiting')
        sys.exit()

    # Got past validation, let's execute the logic
    module = importlib.import_module(f"lwll_dataset_prep.{args.dataset}.main")
    class_ = getattr(module, args.dataset)
    proc = class_()

    if args.download:
        proc.download()
    if args.process:
        proc.process()
    if args.transfer:
        proc.transfer()

    # Section to do validation on output schema
    # We want to check that we have the separate directory with the labels file and that labels files are not within the datasets
    # base_path = Path('/datasets')
    # datasets_path = base_path.joinpath('lwll_datasets')
    # labels_path = base_path.joinpath('lwll_labels')
