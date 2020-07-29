# LwLLDatasetAPI


The script *LwLLDataAPI.py* takes the following command line arguments:
* *--secret* The secret key
* *--mode*
	* *training_data_new* Creates a new session and saves the training and test data for the first checkpoint to the given folder. All previous sessions are deactivated
	* *training_data_continue*  Continues with the most recent session and saves the training and test data for the next checkpoint
	* *submit_predictions* Submits the given predictions to the LwLL website
* *--data_folder* The folder in which the *ted_talks* and *global_voices* lwll datasets have been downloaded. The download instructions are provided below
* *--pred_path* The path of the file containing the predictions on the test data. This should be a csv file with columns "id" and "text". (See the *create_sample_pred_file()* function to get a better understanding of the format)
* *--save_path* The folder in which:
	* the formatted training data and test data will be saved
	* the prediction submission response will be saved

## Step 1: Download the LWLL datasets

The first thing that needs to be done is to download the LWLL datasets from lollllz.com/wiki/display/WA/LwLL+Infrastructure+Resources using the commands:

	python download.py download_data --dataset global_voices --stage development --output DATA_FOLDER --overwrite True
	python download.py download_data --dataset ted_talks --stage development --output DATA_FOLDER --overwrite True


## Step 2: Create a new session and save formatted training and test files

Run the following command:
	
	python LwLLDataAPI.py --secret SECRET_KEY --mode training_data_new --data_folder DATA_FOLDER --save_path SAVE_PATH

This should create two files *train.npy* and *test.npy* in the SAVE_PATH folder
	* *train.npy*: This is a dictionary object saved using numpy, that also contains metadata. Each training instance contains an id, the english sentence and the arabic translation
	* *test.npy*: This is a dictionary object saved using numpy. Each test instance contains an id along with an arabic sentence

Look at the function *load_file_example()* to get a better understanding of these files

## Step 3: Create Prediction File and Submit it

First, create the prediction file according to the required format (see the *create_sample_pred_file()* function to get a better understanding)

Next, submit the prediction file using the following command:
	
	python LwLLDataAPI.py --secret SECRET_KEY --mode submit_predictions --pred_path PRED_FILE_PATH --save_path SAVE_PATH

This will create a file *Submission_Response.txt* in the SAVE_PATH folder that records the response from the LWLL server. This completes the requirement for the first checkpoint.

## Step 4: Obtain new training data and test data for the next checkpoint

Run the following command:
	
	python LwLLDataAPI.py --secret SECRET_KEY --mode training_data_continue --data_folder DATA_FOLDER --save_path SAVE_PATH

Similar to step 2, it will save a new *train.npy* file and a *test.npy* file to the SAVE_PATH folder.

Following this, repeat Step 3.

**NOTE:** The training data changes every checkpoint. However, the test data for the first 8 checkpoints is the same, and changes for the following 8 checkpoints. There are a total of 16 checkpoints. Once the prediction submissions for all 16 checkpoints are completed, the BLEU scores for all checkpoints will be sent back by the server. (saved in SAVE_PATH/Submission_Response.txt)

## Step 5: Continue receiving training and test data, and submitting predicitons

Repeat step 4 14 more times for a total of 16 checkpoints. Look at lines 325 to 330 of *LwLLDataAPI.py* for a rough sketch of the pipeline
	











