# @author Alberto Soragna (alberto dot soragna at gmail dot com)
# @2018

FROM ubuntu:16.04
LABEL maintainer="alberto dot soragna at gmail dot com"

# working directory
ENV HOME /root
WORKDIR $HOME

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# general utilities
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    vim \
    nano \
    python-dev \
    python3-pip \
    ipython

# pip
RUN pip3 install --upgrade pip

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git

#### TENSORFLOW SETUP

# install tensorflow
RUN pip install tensorflow

# general python visualization packages
RUN apt-get install -y \
  python3-tk \
  libgtk2.0-dev
RUN pip install matplotlib opencv-python

RUN apt-get install -y \
  protobuf-compiler \
  python-pil \
  python-lxml \
  python-tk \
  unzip
RUN pip install \
  Cython \
  contextlib2 \
  jupyter \
  matplotlib \
  opencv-python

RUN pip install PyOpenGL PyOpenGL-accelerate
RUN apt-get install -y python3-pil python3-pil.imagetk
RUN pip install hyperopt

#### SET ENVIRONMENT

RUN echo ' \n\
alias python="python3"' >> $HOME/.bashrc




