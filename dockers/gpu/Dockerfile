FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
LABEL maintainer="Mohammad Sadegh Rasooli"
LABEL repository="lwll_mt"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip \
                   vim && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install torch==1.4.0
RUN python3 -m pip install numpy==1.18.2
RUN python3 -m pip install sacrebleu
RUN python3 -m pip install future==0.18.2
RUN python3 -m pip install pillow
RUN python3 -m pip install torchvision==0.5.0
RUN python3 -m pip install ipython
RUN python3 -m pip install transformers==2.9.0
RUN python3 -m pip install pandas
RUN python3 -m pip install pyarrow
RUN python3 -m pip install fire
RUN python3 -m pip install boto3
#RUN python3 -m pip install apex

WORKDIR /lwll
RUN cd /lwll
RUN mkdir /lwll/development
RUN mkdir /lwll/development/global_voices
RUN mkdir /lwll/development/ted_talks
RUN mkdir /lwll/models
ENV DATA_FOLDER=/lwll/development
ENV SAVE_PATH=/lwll/models
ENV LANG C.UTF-8
RUN echo 'alias python="python3"' >> ~/.bashrc

RUN cd /lwll
RUN git clone https://github.com/AdityaKashyap423/LwLLDatasetAPI

WORKDIR /lwll/LwLLDatasetAPI/
COPY dataset_prep /lwll/LwLLDatasetAPI/dataset_prep

WORKDIR /lwll/
# uninstall Apex if present, twice to make absolutely sure :)
RUN python3 -m pip uninstall -y apex || :
RUN python3 -m pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /lwll/apex/
RUN python3 -m pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

WORKDIR /lwll/LwLLDatasetAPI/

CMD ["/bin/bash"]