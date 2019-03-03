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

# pip upgrade
RUN pip3 install --upgrade pip


#### TENSORFLOW SETUP


# install tensorflow
RUN pip install tensorflow


#### ADDITIONAL PYTHON PACKAGES


RUN pip install \
    hyperopt \
    matplotlib \
    pandas

RUN apt-get install -y \
    python3-tk \
    libgtk2.0-dev


#### SET ENVIRONMENT


RUN echo ' \n\
alias python="python3"' >> $HOME/.bashrc




