FROM ubuntu:18.04
LABEL maintainer="Mohammad Sadegh Rasooli"
LABEL repository="lwll_mt"

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
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

WORKDIR /lwll
RUN cd /lwll
RUN mkdir /lwll/development
RUN mkdir /lwll/development/global_voices
RUN mkdir /lwll/development/ted_talks
RUN mkdir /lwll/models
RUN export DATA_FOLDER=/lwll/development
RUN export SAVE_FOLDER=/lwll/models

RUN cd /lwll
RUN git clone https://github.com/AdityaKashyap423/LwLLDatasetAPI
WORKDIR /lwll/LwLLDatasetAPI

CMD ["/bin/bash"]