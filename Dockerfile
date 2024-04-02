FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y gnupg
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
                    'libsm6'\
                    'libxext6'  -y
RUN apt-get install libglib2.0-0 -y

RUN python3 -m pip install opencv-python
RUN python3 -m pip install tensorboard
RUN python3 -m pip install matplotlib

RUN apt-get install vim -y
RUN apt install git -y

RUN python3 -m pip install joblib
RUN python3 -m pip install Pillow
RUN python3 -m pip install tqdm
RUN python3 -m pip install cmapy
RUN python3 -m pip install ray

RUN apt-get install gcc -y
RUN apt-get install --reinstall build-essential -y

RUN python3 -m pip install git+https://github.com/lucasb-eyer/pydensecrf.git

WORKDIR /