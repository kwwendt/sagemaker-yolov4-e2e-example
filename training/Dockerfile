#FROM public.ecr.aws/entelect-challenge/languages/python_pytorch:2021
FROM public.ecr.aws/lts/ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
   apt-get -y install python3-pip && \
   apt-get -y install git && \
   apt-get -y install build-essential && \
   apt-get -y install libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

RUN cd /opt \
&& git clone https://github.com/roboflow-ai/pytorch-YOLOv4.git \
&& cd pytorch-YOLOv4

RUN pip3 install numpy==1.18.2
RUN pip3 install torch==1.7.0
RUN pip3 install scikit_image==0.16.2
RUN pip3 install matplotlib
RUN pip3 install tqdm==4.43.0
RUN pip3 install easydict==1.9
RUN pip3 install Pillow==7.1.2
RUN pip3 install tensorboardX
RUN pip3 install gdown
RUN pip3 install opencv-python
RUN pip3 install opencv-contrib-python

RUN gdown https://drive.google.com/uc?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA -O /opt/pytorch-YOLOv4/cfg/yolov4.conv.137.pth
COPY train.py /opt/pytorch-YOLOv4/
COPY dataset.py /opt/pytorch-YOLOv4/

WORKDIR /opt/pytorch-YOLOv4
ENTRYPOINT ["python3", "train.py"]