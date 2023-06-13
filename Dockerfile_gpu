#Deriving the latest base image
FROM rocm/dev-ubuntu-22.04:latest
# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /scripts

COPY requirements.txt ./

RUN apt-get update \
  && apt-get install -y nano git libjpeg-dev python3-dev

RUN pip3 install wheel setuptools \
  && pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2 \
  && pip3 install -r requirements.txt
#5.4.2 soontm  
#CMD instruction should be used to run the software
#contained by your image, along with any arguments.

#COPY /scripts ./


CMD [ "python3" "./main.py"]
#CMD [ "python", "./test.py"]
