#!/bin/bash
# A simple bash script to launch multiple trainings with the same parameters,
# with only the first one rendered on tensorboard (tensorboard logs taking a lot of disk space)
# will ask for :
#   - training's name, string
#   - port, if using 1env_1jet method, (integer > 1024)
#   - number of trainings

read -p "Training's name: " tn
read -p "Port to use (if multienv): " port
read -p "Number of trainings: " n
((n--))

nohup python3 train.py -t -tn $tn -p $port -tb &

counter=1
while [ $counter -le $n ]
do
    ((port++))
    tn="${tn}_"
    nohup python3 train.py -t -tn $tn -p $port &
    ((counter++)) 
done

