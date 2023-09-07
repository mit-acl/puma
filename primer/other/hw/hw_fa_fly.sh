#!/bin/bash

# this script creates multiple windows for each agent
# for now it only supports 2 agents

# session name
SESSION=hw_fa_fly
WINDOW=agents

# creates the session with a name and renames the window name
cmd="new-session -d -s $SESSION -x- -y-; rename-window $WINDOW"
tmux -2 $cmd

# window number
w=0

##
## Set the parameters
##

# agent1
agent1_voxl="nx01"
agent1_nuc="nuc1"
agent1_ip="192.168.15.2"
agent1_adhoc_ip="192.168.100.1"
agent1_quad="NX01"

# agent2
agent2_voxl="nx02"
agent2_nuc="nuc2"
agent2_ip="192.168.16.2"
agent2_adhoc_ip="192.168.100.2"
agent2_quad="NX02"

# motlee
kappa_mot=600

# split tmux into 6x3
# 1 base station (hw_base_station_fast_sam.launch) 2 agents times 5 panes (fly script, trajectory_generator onboard.launch, fastsam.launch, mapper.launch, frame_aligner.launch, ping test)
# fly script and trajectory_generator are in voxl
# fastsam, mapper, frame_aligner are in nuc

######################################
#            #           #           #   
#            #     1     #     7     #    
#            #           #           #   
#            #########################
#            #           #           #   
#            #     2     #     8     #   
#            #           #           #   
#            #########################
#            #           #           #   
#            #     3     #     9     #   
#            #           #           #   
#      0     #########################
#            #           #           #   
#            #     4     #    10     #   
#            #           #           #
#            #########################
#            #           #           #
#            #     5     #    11     #
#            #           #           #
#            #########################
#            #           #           #
#            #     6     #    12     #
#            #           #           #
######################################

# pane 0: base station
# pane 1: agent1 fly
# pane 2: agent1 trajectory_generator
# pane 3: agent1 fastsam
# pane 4: agent1 mapper
# pane 5: agent1 frame_aligner
# pane 6: agent1 ping test
# pane 7: agent2 fly
# pane 8: agent2 trajectory_generator
# pane 9: agent2 fastsam
# pane 10: agent2 mapper
# pane 11: agent2 frame_aligner
# pane 12: agent2 ping test

# create 6x3 panes
for i in 0 1
do
	tmux split-window -h
	tmux select-layout -t $SESSION:$w.$i even-horizontal
	tmux select-pane -t $SESSION:$w.$i
done

for i in 1 2 3 4 5 6 7 8 9 10 11 12
do 
	# tmux select-layout -t $SESSION:$w.$i even-vertical
	tmux select-pane -t $SESSION:$w.$i
	tmux split-window -v

done

# resize panes for voxl fly
for i in 1 7
do
	tmux resize-pane -t $SESSION:$w.$i -y 30
done

# resize panes for other panes
for i in 2 3 4 5 6 8 9 10 11 12
do
	tmux resize-pane -t $SESSION:$w.$i -y 10
done

# wait for .bashrc to load
sleep 1

# send commands to each pane
# ssh each voxl 
tmux send-keys -t $SESSION:$w.1 "ssh root@${agent1_voxl}.local" C-m
tmux send-keys -t $SESSION:$w.2 "ssh root@${agent1_voxl}.local" C-m
tmux send-keys -t $SESSION:$w.7 "ssh root@${agent2_voxl}.local" C-m
tmux send-keys -t $SESSION:$w.8 "ssh root@${agent2_voxl}.local" C-m

sleep 3

# run voxl_voxl_connection
for i in 1 2 7 8
do
	tmux send-keys -t $SESSION:$w.$i "./nuc_voxl_connection" C-m
done

# ssh each nuc
# agent 1
for i in 3 4 5 6
do 
    tmux send-keys -t $SESSION:$w.$i "ssh ${agent1_nuc}@${agent1_ip}" C-m
done

# agent 2
for i in 9 10 11 12
do 
    tmux send-keys -t $SESSION:$w.$i "ssh ${agent2_nuc}@${agent2_ip}" C-m
done

sleep 1

# nuc housekeeping 
for i in 3 4 5 6 9 10 11 12
do 
    # ntp date
    tmux send-keys -t $SESSION:$w.$i "sudo ntpdate time.nist.gov" C-m
    # ad_hoc
    tmux send-keys -t $SESSION:$w.$i "cd && ./ad_hoc_without_NM.sh" C-m
done

sleep 10

# ping each other
tmux send-keys -t $SESSION:$w.6 "cd && ping ${agent2_adhoc_ip}" C-m
tmux send-keys -t $SESSION:$w.12 "cd && ping ${agent1_adhoc_ip}" C-m

sleep 5

# run roslaunches
# base station
tmux send-keys -t $SESSION:$w.0 "cd && roslaunch --wait trajectory_generator base_station.launch" C-m
# agent1
tmux send-keys -t $SESSION:$w.1 "fly" C-m
tmux send-keys -t $SESSION:$w.2 "cd && roslaunch --wait trajectory_generator quad:={agent1_quad} onboard.launch" C-m
tmux send-keys -t $SESSION:$w.3 "cd && roslaunch --wait primer fastsam.launch quad:=${agent1_quad} is_sim:=false" C-m
tmux send-keys -t $SESSION:$w.4 "cd && roslaunch --wait motlee_ros mapper.launch quad:=${agent1_quad} kappa:=${kappa_mot}" C-m
tmux send-keys -t $SESSION:$w.5 "cd && roslaunch --wait motlee_ros frame_aligner.launch quad1:={agent1_quad} quad2:={agent2_quad}" C-m
# agent2
tmux send-keys -t $SESSION:$w.7 "fly" C-m
tmux send-keys -t $SESSION:$w.8 "cd && roslaunch --wait trajectory_generator quad:={agent2_quad} onboard.launch" C-m
tmux send-keys -t $SESSION:$w.9 "cd && roslaunch --wait primer fastsam.launch quad:=${agent2_quad} is_sim:=false" C-m
tmux send-keys -t $SESSION:$w.10 "cd && roslaunch --wait motlee_ros mapper.launch quad:=${agent2_quad} kappa:=${kappa_mot}" C-m
tmux send-keys -t $SESSION:$w.11 "cd && roslaunch --wait motlee_ros frame_aligner.launch quad1:={agent2_quad} quad2:={agent1_quad}" C-m

# attach to the session
tmux -2 attach-session -t $SESSION