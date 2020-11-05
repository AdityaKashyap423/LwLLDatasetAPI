# LwLLDatasetAPI

## Docker Commands

__Asuuming that Docker and NVIDIA docker is installed.__, follow the following steps:

1. Download the repository and pretrained models
```
https://github.com/AdityaKashyap423/LwLLDatasetAPI
cd LwLLDatasetAPI/dockers/gpu/
git clone https://gitlab.lollllz.com/lwll/dataset_prep.git # needs user/pass for lolllz.com

cd ~/LwLLDatasetAPI
```


2. Build the docker in command line:
```bash
docker build dockers/cpu/ -t ady --no-cache
```

3. Start running the docker:

* Run this with screen since training might take a long time.
```bash
docker run --gpus all -it  ady
```

4. Run the following command for setting SECRET key, and downloading data:
```bash
git pull 
apt install wget
apt install unzip
rm -rf mt_pret*
wget https://www.seas.upenn.edu/~rasooli/mt_pret.zip
unzip mt_pret.zip  -d $SAVE_PATH/train.model
rm -rf mt_pret.zip
python dataset_prep/download.py download_data --dataset global_voices --stage development --output $DATA_FOLDER/.. --overwrite True
python dataset_prep/download.py download_data --dataset ted_talks --stage development --output $DATA_FOLDER/.. --overwrite True
export SECRET_KEY= [PUT your secret key here!]
export CUDA_VISIBLE_DEVICES=0
```

5. Run the full pipeline:
```
nohup nice python3 -u LwLLDataAPI.py --secret $SECRET_KEY --data_folder $DATA_FOLDER --save_path $SAVE_PATH &> log.txt &
```


### INSTALL DOCKER 
* Install docker
* Install NVIDIA-DOCKER
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

```




