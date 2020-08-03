import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
import re

url = 'https://api-dev.lollllz.com'
data_type = 'full'
task_id = '06023f86-a66b-4b2c-8b8b-951f5edd0f22'  # For machine translation


def get_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-secret', '--secret', required=True)
    parser.add_argument('-mode', '--mode', required=True)
    parser.add_argument('-data_folder', '--data_folder')
    parser.add_argument('-pred_path', '--pred_path')
    parser.add_argument('-save_path', '--save_path')
    args = vars(parser.parse_args())
    global secret
    secret = args["secret"]
    mode = args["mode"]
    data_path = args["data_folder"]
    pred_path = args["pred_path"]
    save_path = args["save_path"]

    if (mode == "training_data_new" or mode == "training_data_continue") and (data_path is None or save_path is None):
        print("For mode = training_data_new/continue, --data_folder and --save_path is required!")
        exit(1)

    if mode == "submit_predictions" and (pred_path is None or save_path is None):
        print("For mode = submit_predictions, --pred_path and --save_path is required!")
        exit(1)

    return mode, data_path, pred_path, secret, save_path


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


def save_to_file(data,save_path,filename):
    with open(save_path + filename, "w") as f:
        f.write("\n".join(data))
        f,write("\n")



def save_data(data, already_queried, session_token, checkpoint_number, data_type, save_path):
    all_data = {}

    if checkpoint_number == 8:
        already_queried = []  # Because we start with a different dataset at checkpoint 9

    metadata = {}
    metadata['already_queried'] = already_queried
    metadata['session_token'] = session_token
    metadata['checkpoint_number'] = checkpoint_number

    all_data["metadata"] = metadata

    if data_type == "train":
        np.save(save_path + "metadata.npy", all_data)
        eng = []
        ar = []
        for element in data:
            eng.append(element["english"].replace("\r", ""))
            ar.append(element["arabic"].replace("\r", ""))

        save_to_file(eng,save_path,"english.train")
        save_to_file(ar,save_path,"arabic.train")


    elif data_type == "test":

        ar = []
        ID = []
        for key in data.keys():
            ID.append(key)
            ar.append(data[key])


        save_to_file(ar,save_path,"arabic.test")
        save_to_file(ID,save_path,"ID.test")


def load_previous_checkpoint_data(save_path):
    save_path = save_path + "metadata.npy"
    data = np.load(save_path, allow_pickle=True).item()
    return data['metadata']


def training_data_new(path, save_path):
    checkpoint_number = 1

    session_token = create_session()
    deactivate_session(session_token)  # Deactivates all previous sessions

    DATASETS_PATH = Path(path)
    train_df = get_train_data_mt(DATASETS_PATH, session_token)  # 63947 training instances: id, source
    all_training_ids = list(train_df["id"])

    already_queried = []

    training_data, already_queried = get_training_labels(all_training_ids, already_queried, train_df, session_token)

    save_data(training_data, already_queried, session_token, checkpoint_number, "train", save_path)
    get_test_data(session_token, DATASETS_PATH, save_path)


def training_data_continue(path, save_path):
    metadata = load_previous_checkpoint_data(save_path)

    session_token = metadata['session_token']
    already_queried = metadata['already_queried']
    checkpoint_number = metadata['checkpoint_number'] + 1

    DATASETS_PATH = Path(path)
    train_df = get_train_data_mt(DATASETS_PATH, session_token)  # 63947 training instances: id, source
    all_training_ids = list(train_df["id"])

    training_data, already_queried = get_training_labels(all_training_ids, already_queried, train_df, session_token)
    save_data(training_data, already_queried, session_token, checkpoint_number, "train", save_path)
    get_test_data(session_token, DATASETS_PATH, save_path)


def get_test_data(session_token, path, save_path):
    test_df = get_test_data_mt(path, session_token)
    test_df = df_to_dict(test_df)
    save_data(test_df, [], session_token, "N/A", "test", save_path)

def create_random_predictions(save_path):
    with open(save_path + "ID.test") as f:
        data_len = len(f.read().split("\n")[:-1])

    random_pred = ["This is a random prediction" for _ in range(data_len)]    
    with open(save_path + "english.test","w") as f:
        for line in random_pred:
            f.write(line)
            f.write("\n")


def submit_predictions(pred_path, save_path):
    metadata = load_previous_checkpoint_data(save_path)

    session_token = metadata['session_token']
    checkpoint_number = metadata['checkpoint_number']

    pred_df = []
    with open(save_path + "ID.test") as f:
        IDs = f.read().split("\n")[:-1]

    with open(pred_path) as f:
        preds = f.read().split("\n")[:-1]     

    for ID,pred in zip(IDs,preds):
        pred_df.append({"id":ID,"text":pred})

    pred_df = pd.DataFrame(pred_df)


    headers = {'user_secret': secret, 'session_token': session_token}
    r = requests.post(f"{url}/submit_predictions", json={'predictions': pred_df.to_dict()}, headers=headers)

    with open(save_path + "Submission_Response.txt", "w") as f:
        f.write(str(r.json()))



if __name__ == '__main__':

    mode, data_folder, pred_path, secret, save_path = get_command_line_arguments()

    if mode == 'training_data_new':
        training_data_new(data_folder, save_path)

    elif mode == 'training_data_continue':
        training_data_continue(data_folder, save_path)

    elif mode == 'submit_predictions':
        # create_random_predictions(save_path)
        submit_predictions(pred_path, save_path)
 
# for i in range(16): 
#     if i == 0:
#         training_data_new(data_folder, save_path)
#     else:
#         training_data_continue(data_folder, save_path)


#     create_random_predictions(save_path)
#     submit_predictions(pred_path,save_path)


