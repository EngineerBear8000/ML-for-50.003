
sudo docker build -t ocr .

docker run -it -v /home/engineerbear/Documents/OCR/data:/data -v /home/engineerbear/Documents/OCR/scripts:/scripts --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G OCR:latest bash

git clone https://github.com/pytorch/examples.git

export HSA_OVERRIDE_GFX_VERSION=10.3.0