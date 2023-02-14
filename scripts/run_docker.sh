#!/bin/bash

docker run -it --rm --name=$2 --net=host --dns=8.8.8.8 --gpus=2 -v /mnt/ssd0/damien:/mnt/ssd0/damien -v /mnt/hdd0/damien:/mnt/hdd0 damien911224/testbed:$1 /bin/bash
