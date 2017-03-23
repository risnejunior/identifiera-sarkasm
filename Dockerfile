FROM ubuntu:16.10

ENV LANG en_US.UTF-8

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        # libpng12 \
        libzmq3-dev \
        pkg-config \
        python3.6 \
        python3-pip \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#     python get-pip.py && \
#     rm get-pip.py
RUN pip3 --no-cache-dir install \
        --upgrade pip \
        -U setuptools

RUN pip3 --no-cache-dir install \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        nltk \
        Pillow \
        tensorflow \
        tflearn

# COPY datasets /datasets
#
# COPY src /src

# TensorBoard
EXPOSE 6006

# WORKDIR "/src"
#
# CMD ["python3", "clean_tweets.py"]
# CMD ["python3", "preprocess_data.py"]
# CMD ["python3", "tflearn_rnn.py"]
