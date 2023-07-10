#Deriving the latest base image
FROM jupyter/minimal-notebook:python-3.10.11
# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /scripts

COPY requirements.txt ./

USER root
RUN apt-get update \
  && apt-get install -y nano git ffmpeg libsm6 libxext6

USER jovyan
RUN pip3 install wheel setuptools \
  && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 

RUN pip3 install -r requirements.txt
 
#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

#COPY /scripts ./

CMD ["start-notebook.sh"]
#CMD [ "python3" "./main.py"]
#CMD [ "python", "./test.py"]
