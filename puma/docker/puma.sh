#!/bin/bash

# arguments
# $1: simulation type

source ~/.bashrc
source /home/code/ws/devel/setup.bash
roscd puma
cd other/demos

echo the argument is $1

if [ "$1" == "planner" ]; then
    python3 uncertainty_aware_planner_demo.py
elif [ "$1" == "frame" ]; then
    python3 frame_alignment_demo.py
else
    python3 uncertainty_aware_planner_on_frame_alignment_demo.py
fi