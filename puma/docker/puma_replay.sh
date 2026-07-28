#!/bin/bash
# Replay a recorded rosbag and open rviz so you can screen-record a video.
# Usage (inside the container, called by `make replay-bag`): puma_replay.sh <bagfile>
set -e
source /opt/ros/noetic/setup.bash
source /home/code/ws/devel/setup.bash

BAG="$1"
# If the exact bag isn't there but an unfinalized "<bag>.active" is (e.g. the sim
# ended while recording), reindex it into a playable bag automatically.
if [ ! -f "$BAG" ] && [ -f "$BAG.active" ]; then
  echo "Found $BAG.active (unfinalized) -- reindexing to $BAG ..."
  rosbag reindex "$BAG.active"
  mv -f "$BAG.active" "$BAG"
  rm -f "$(dirname "$BAG")"/*.orig.active 2>/dev/null || true
fi
if [ ! -f "$BAG" ]; then
  echo "Bag not found: $BAG"
  echo "Available bags:"; ls -1 /bags 2>/dev/null
  exit 1
fi

# roscore + sim time (the bag publishes /clock via 'rosbag play --clock')
roscore &
sleep 4
rosparam set use_sim_time true

# rviz with the puma config (same view used live)
rviz -d "$(rospack find puma)/rviz_cfgs/panther.rviz" &
sleep 4

echo "================================================================"
echo "Playing: $BAG   (looping). Press Ctrl-C here to stop."
echo "Screen-record the rviz window now."
echo "================================================================"
# --clock: publish the bag's /clock so timestamps/TF line up
# --loop:  repeat so you have time to set up the recording / camera angle
rosbag play --clock --loop "$BAG"
