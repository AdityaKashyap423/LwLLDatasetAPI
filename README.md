# LwLLDatasetAPI

## Docker Commands

Here is a simple example for running the docker:
* Install docker
* Install NVIDIA-DOCKER
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

```
* Download the repository
```
https://github.com/AdityaKashyap423/LwLLDatasetAPI
cd LwLLDatasetAPI/dockers/gpu/
git clone https://gitlab.lollllz.com/lwll/dataset_prep.git
```


* Build the docker in command line:
```bash
docker build dockers/cpu/ -t ady --no-cache
```

* Start the docker
```bash
docker run --gpus all -it  ady
```
Then we can run the following command for the first checkpoint
```bash
python dataset_prep/download.py download_data --dataset global_voices --stage development --output $DATA_FOLDER/.. --overwrite True
python dataset_prep/download.py download_data --dataset ted_talks --stage development --output $DATA_FOLDER/.. --overwrite True
export SECRET_KEY= [PUT your secret key here!]
CUDA_VISIBLE_DEVICES=0 python3 -u LwLLDataAPI.py --secret $SECRET_KEY --mode new --data_folder $DATA_FOLDER --save_path $SAVE_PATH --enc 1 --dec 1 --embed 96 --iter 1 --beam 1
```

## Details


The script *LwLLDataAPI.py* takes the following command line arguments:
* *--secret* The secret key
* *--mode*
	* *training_data_new* Creates a new session and saves the training and test data (*arabic.train,english.train,arabic.test*) for the first checkpoint to the given folder. All previous sessions are deactivated
	* *training_data_continue*  Continues with the most recent session and saves the training and test data for the next checkpoint
	* *submit_predictions* Submits the given predictions (*english.test*) to the LwLL website
* *--data_folder* The folder in which the *ted_talks* and *global_voices* lwll datasets have been downloaded. The download instructions are provided below
* *--pred_path* The path of the file containing the predictions on the test data (*english.test*). This should be a text file with 1 line per prediction.   
* *--save_path* The folder in which:
	* the formatted training data and test data will be saved (*arabic.train,english.train,arabic.test*)
	* the prediction submission response will be saved (*Submission_Response.txt*)

## Step 1: Download the LWLL datasets

The first thing that needs to be done is to download the LWLL datasets from lollllz.com/wiki/display/WA/LwLL+Infrastructure+Resources using the commands:

	python download.py download_data --dataset global_voices --stage development --output DATA_FOLDER --overwrite True
	python download.py download_data --dataset ted_talks --stage development --output DATA_FOLDER --overwrite True


## Step 2: Create a new session and save formatted training and test files

Run the following command:
	
	python LwLLDataAPI.py --secret SECRET_KEY --mode training_data_new --data_folder DATA_FOLDER --save_path SAVE_PATH

This should create three files (*arabic.train,english.train,arabic.test*) in the SAVE_PATH folder

## Step 3: Create Prediction File and Submit it

First, create the prediction file (*english.test*) with one prediction per line

Next, submit the prediction file using the following command:
	
	python LwLLDataAPI.py --secret SECRET_KEY --mode submit_predictions --pred_path PRED_FILE_PATH --save_path SAVE_PATH

This will create a file *Submission_Response.txt* in the SAVE_PATH folder that records the response from the LWLL server. This completes the requirement for the first checkpoint.

## Step 4: Obtain new training data and test data for the next checkpoint

Run the following command:
	
	python LwLLDataAPI.py --secret SECRET_KEY --mode training_data_continue --data_folder DATA_FOLDER --save_path SAVE_PATH

Similar to step 2, it will save new files (*arabic.train,english.train,arabic.test*) to the SAVE_PATH folder.

Following this, repeat Step 3.

**NOTE:** The training data changes every checkpoint. However, the test data for the first 8 checkpoints is the same, and changes for the following 8 checkpoints. There are a total of 16 checkpoints. Once the prediction submissions for all 16 checkpoints are completed, the BLEU scores for all checkpoints will be sent back by the server. (saved in SAVE_PATH/Submission_Response.txt)

## Step 5: Continue receiving training and test data, and submitting predicitons

Repeat step 4 14 more times for a total of 16 checkpoints. Look at lines 325 to 330 of *LwLLDataAPI.py* for a rough sketch of the pipeline
	 











