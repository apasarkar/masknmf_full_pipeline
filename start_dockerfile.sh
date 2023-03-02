#!/bin/bash 


PROOT=$(pwd)
echo $PROOT
sudo docker run -it -p 8981:8900 --gpus=all --mount type=bind,source=$PROOT/datasets/$2,destination=/mounted_data/$2 $1
