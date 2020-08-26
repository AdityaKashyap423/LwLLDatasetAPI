import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests

from ImageTranslate import train_image_mt, translate, create_mt_batches
from ImageTranslate.mt_options import TrainOptions, TranslateOptions
from ImageTranslate.textprocessor import TextProcessor

url = 'https://api-dev.lollllz.com'
data_type = 'full'
task_id = '06023f86-a66b-4b2c-8b8b-951f5edd0f22'  # For machine translation


def get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--secret', '-secret', required=True)
    parser.add_argument('--data_folder', '-data_folder')
    parser.add_argument('--save_path', '-save_path')
    parser.add_argument('--enc', '-enc', type=int, default=6)
    parser.add_argument('--dec', '-dec', type=int, default=6)
    parser.add_argument('--embed', '-embed', type=int, default=768)
    parser.add_argument('--iter', '-iter', type=int, default=100000)
    parser.add_argument('--beam', '-beam', type=int, default=5)
    parser.add_argument('--batch', '-batch', type=int, default=60000)
    parser.add_argument('--capacity', '-capacity', type=int, default=1500)
    args = vars(parser.parse_args())
    global secret

    secret = args["secret"]
    data_path = args["data_folder"]
    save_path = args["save_path"]
    if data_path is None or save_path is None:
        print("--data_folder and --save_path is required!")
        exit(1)

    return args


def create_session():
    headers = {'user_secret': secret}

    r = requests.post(f"{url}/auth/create_session",
                      json={'session_name': 'testing', 'data_type': data_type, 'task_id': task_id}, headers=headers)
    session_token = r.json()['session_token']

    print("Created session with token: ", session_token)

    return session_token


def deactivate_session(session_token, session="all"):
    headers_session = {'user_secret': secret}
    r = requests.get(f"{url}/list_active_sessions", headers=headers_session)
    active_sessions = r.json()
    all_active_sessions = active_sessions['active_sessions']

    headers_active_session = {'user_secret': secret, 'session_token': session_token}

    if session == "all":
        for current_session in set(all_active_sessions) - set(
                [session_token]):  # Deactivate all sessions apart from the current session
            print("Deactivation Session: " + current_session)
            r = requests.post(f"{url}/deactivate_session", json={'session_token': current_session},
                              headers=headers_active_session)
    else:
        if session not in all_active_sessions:
            print("ERROR: This session does not exist!")
        else:
            r = requests.post(f"{url}/deactivate_session", json={'session_token': session},
                              headers=headers_active_session)


def get_train_data_mt(dataset_path: Path, session_token: str) -> List[str]:
    """
    Helper method to dynamically get the test labels and give us the possible classes that can be submitted
    for the current dataset

    Params
    ------

    dataset_path : Path
        The path to the `development` dataset downloads

    session_token : str
        Your current session token so that we can look up the current session metadata

    Returns
    -------

    pd.DataFrame
        The DataFrame on which you can make queries against
    """
    # Then we can just reference our current metadata to get our dataset name and use that in the path
    headers = {'user_secret': secret, 'session_token': session_token}
    r = requests.get(f"{url}/session_status", headers=headers)
    current_dataset = r.json()['Session_Status']['current_dataset']
    current_dataset_name = current_dataset['name']

    test_df = pd.read_feather(
        str(dataset_path.joinpath(f"{current_dataset_name}/{current_dataset_name}_{data_type}/train_data.feather")))
    return test_df


def get_test_data_mt(dataset_path: Path, session_token: str) -> List[str]:
    """
    Helper method to dynamically get the test labels and give us the possible classes that can be submitted
    for the current dataset

    Params
    ------

    dataset_path : Path
        The path to the `development` dataset downloads

    session_token : str
        Your current session token so that we can look up the current session metadata

    Returns
    -------

    pd.DataFrame
        The DataFrame on which you must make predictions from a 'source' column
    """
    # Then we can just reference our current metadata to get our dataset name and use that in the path
    headers = {'user_secret': secret, 'session_token': session_token}
    r = requests.get(f"{url}/session_status", headers=headers)
    current_dataset = r.json()['Session_Status']['current_dataset']
    current_dataset_name = current_dataset['name']

    _path = str(dataset_path.joinpath(f"{current_dataset_name}/{current_dataset_name}_{data_type}/test_data.feather"))
    test_df = pd.read_feather(_path)
    return test_df


def get_training_labels(all_training_ids, already_queried, train_df, session_token):
    retrieval_cap = 25000  # set by the LwLL server

    train_df = df_to_dict(train_df)

    headers = {'user_secret': secret, 'session_token': session_token}
    r = requests.get(f"{url}/session_status", headers=headers).json()

    required_ids = list(set(all_training_ids) - set(already_queried))

    num_parts = int(len(required_ids) / retrieval_cap) + 1

    training_labels = []

    repeat = True

    while repeat:

        for i in range(num_parts):
            required_ids_part = required_ids[i * retrieval_cap:(i + 1) * retrieval_cap]
            query = {'example_ids': required_ids_part}

            r = requests.post(f"{url}/query_labels", json=query, headers=headers).json()

            for instance in r['Labels']:
                ID = instance['id']
                eng = instance['text']
                ar = train_df[instance['id']]
                current_dict = {}
                current_dict['id'] = ID
                current_dict['english'] = eng
                current_dict['arabic'] = ar
                training_labels.append(current_dict)
                already_queried.append(ID)

            budget_left = r['Session_Status']['budget_left_until_checkpoint']

            if budget_left == 0:
                repeat = False
                break

        if budget_left > 0:
            already_queried = []
            required_ids = all_training_ids[:]
            num_parts = int(len(required_ids) / retrieval_cap) + 1

    return training_labels, already_queried


def df_to_dict(df):
    final_dict = {}

    df = df.to_dict('records')
    for instance in df:
        final_dict[instance['id']] = instance['source']
    return final_dict


def save_to_file(data, save_path, filename):
    with open(os.path.join(save_path, filename), "w") as f:
        f.write("\n".join(data))
        f.write("\n")


def save_data(data, already_queried, session_token, checkpoint_number, data_type, save_path, args):
    all_data = {}

    if checkpoint_number == 8:
        already_queried = []  # Because we start with a different dataset at checkpoint 9

    metadata = {}
    metadata['already_queried'] = already_queried
    metadata['session_token'] = session_token
    metadata['checkpoint_number'] = checkpoint_number

    all_data["metadata"] = metadata
    tok_path = os.path.dirname(os.path.realpath(__file__)) + "/tok"

    if data_type == "train":
        np.save(os.path.join(save_path, "metadata.npy"), all_data)
        eng = []
        ar = []
        for element in data:
            eng.append(element["english"].replace("\r", ""))
            ar.append(element["arabic"].replace("\r", ""))

        if checkpoint_number != 1:
            ar_prev, eng_prev = load_training_data(save_path)
            ar = ar_prev + ar
            eng = eng_prev + eng

        save_to_file(eng, save_path, "english.train")
        save_to_file(ar, save_path, "arabic.train")
        tokenizer = TextProcessor(tok_path)

        create_mt_batches.write(text_processor=tokenizer, output_file=os.path.join(save_path, "train.batch"),
                                src_txt_file=os.path.join(save_path, "arabic.train"),
                                dst_txt_file=os.path.join(save_path, "english.train"))
        train_options = TrainOptions()
        train_options.mt_train_path = os.path.join(save_path, "train.batch")
        num_iters = max(10, (len(eng) / (train_options.batch / 100)) * 10)
        train_options.step = int(min(args["iter"], num_iters))
        print("Training for", train_options.step, "iterations!")
        train_options.model_path = os.path.join(save_path, "train.model")
        train_options.tokenizer_path = tok_path
        train_options.encoder_layer = args["enc"]
        train_options.decoder_layer = args["dec"]
        train_options.embed_dim = args["embed"]
        train_options.beam_width = args["beam"]
        train_image_mt.ImageMTTrainer.train(train_options)
        print("Training Done!")


    elif data_type == "test":
        ar = []
        ID = []
        for key in data.keys():
            ID.append(key)
            ar.append(data[key])

        save_to_file(ar, save_path, "arabic.test")
        save_to_file(ID, save_path, "ID.test")

        print("Translating ...")
        translate_options = TranslateOptions()
        translate_options.mt_train_path = os.path.join(save_path, "train.batch")
        translate_options.model_path = os.path.join(save_path, "train.model")
        translate_options.tokenizer_path = tok_path
        translate_options.input_path = os.path.join(save_path, "arabic.test")
        translate_options.output_path = os.path.join(save_path, "english.test.output")
        translate_options.beam_width = args["beam"]
        translate.translate(translate_options)
        print("Translating done!")


def load_previous_checkpoint_data(save_path):
    save_path = os.path.join(save_path, "metadata.npy")
    data = np.load(save_path, allow_pickle=True).item()
    return data['metadata']


def training_data_new(path, save_path, args):
    checkpoint_number = 1

    session_token = create_session()
    deactivate_session(session_token)  # Deactivates all previous sessions

    DATASETS_PATH = Path(path)
    train_df = get_train_data_mt(DATASETS_PATH, session_token)  # 63947 training instances: id, source
    all_training_ids = list(train_df["id"])

    already_queried = []

    training_data, already_queried = get_training_labels(all_training_ids, already_queried, train_df, session_token)

    save_data(training_data, already_queried, session_token, checkpoint_number, "train", save_path, args)

    get_test_data(session_token, DATASETS_PATH, save_path, args)


def training_data_continue(path, save_path, args):
    metadata = load_previous_checkpoint_data(save_path)

    session_token = metadata['session_token']
    already_queried = metadata['already_queried']
    checkpoint_number = metadata['checkpoint_number'] + 1

    DATASETS_PATH = Path(path)
    train_df = get_train_data_mt(DATASETS_PATH, session_token)  # 63947 training instances: id, source
    all_training_ids = list(train_df["id"])

    training_data, already_queried = get_training_labels(all_training_ids, already_queried, train_df, session_token)
    save_data(training_data, already_queried, session_token, checkpoint_number, "train", save_path, args)
    get_test_data(session_token, DATASETS_PATH, save_path, args)


def get_test_data(session_token, path, save_path, args):
    test_df = get_test_data_mt(path, session_token)
    test_df = df_to_dict(test_df)
    save_data(test_df, [], session_token, "N/A", "test", save_path, args)


def create_random_predictions(save_path):
    with open(os.path.join(save_path, "ID.test")) as f:
        data_len = len(f.read().split("\n")[:-1])

    random_pred = ["This is a random prediction" for _ in range(data_len)]
    with open(os.path.join(save_path, "english.test"), "w") as f:
        for line in random_pred:
            f.write(line)
            f.write("\n")


def submit_predictions(save_path):
    metadata = load_previous_checkpoint_data(save_path)

    session_token = metadata['session_token']
    checkpoint_number = metadata['checkpoint_number']

    pred_df = []
    with open(os.path.join(save_path, "ID.test")) as f:
        IDs = f.read().split("\n")[:-1]

    with open(os.path.join(save_path, "english.test.output")) as f:
        preds = f.read().split("\n")[:-1]

    for ID, pred in zip(IDs, preds):
        pred_df.append({"id": ID, "text": pred})

    pred_df = pd.DataFrame(pred_df)

    headers = {'user_secret': secret, 'session_token': session_token}
    r = requests.post(f"{url}/submit_predictions", json={'predictions': pred_df.to_dict()}, headers=headers)

    with open(os.path.join(save_path, "Submission_Response.txt"), "w") as f:
        f.write(str(r.json()))


def load_training_data(save_path):
    with open(os.path.join(save_path, "english.train")) as f:
        eng = f.read().split("\n")[:-1]

    with open(os.path.join(save_path, "arabic.train")) as f:
        ar = f.read().split("\n")[:-1]

    return ar, eng


if __name__ == '__main__':

    args = get_command_line_arguments()
    secret = args["secret"]
    data_folder = args["data_folder"]
    save_path = args["save_path"]

    for i in range(16):
        print("Starting round", (i + 1), "training!")
        if i == 0:
            training_data_new(data_folder, save_path, args)
            submit_predictions(save_path)
        else:
            training_data_continue(data_folder, save_path, args)
            submit_predictions(save_path)
        print("Done with submitting predictions")
