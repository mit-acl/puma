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
agent1_ip="192.168.18.2"
agent1_adhoc_ip="192.168.100.8"
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
#            #     1     #     8     #    
#            #           #           #   
#            #########################
#            #           #           #   
#            #     2     #     9     #   
#            #           #           #   
#            #########################
#            #           #           #   
#            #     3     #    10     #   
#            #           #           #   
#      0     #########################
#            #           #           #   
#            #     4     #    11     #   
#            #           #           #
#            #########################
#            #           #           #
#            #     5     #    12     #
#            #           #           #
#            #########################
#            #           #           #
#            #     6     #    13     #
#            #           #           #
#            #########################
#            #           #           #
#            #     7     #    14     #
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

tmux resize-pane -t $SESSION:$w.0 -x 20
tmux resize-pane -t $SESSION:$w.1 -x 75

for i in 1 2
do 
	# tmux select-layout -t $SESSION:$w.$i even-vertical
	tmux select-pane -t $SESSION:$w.$i
	tmux split-window -v

	if [ $i=1 ] || [ $i=8 ]
	then
		tmux resize-pane -t $SESSION:$w.$i -y 150
	else
		tmux resize-pane -t $SESSION:$w.$i -y 10
	fi
done

# resize panes for voxl fly
# for i in 1 8
# do
# done

# # resize panes for other panes
# for i in 2 3 4 5 6 7 9 10 11 12 13 14
# do
# 	tmux resize-pane -t $SESSION:$w.$i -y 10
# done

# # wait for .bashrc to load
# sleep 1

# # send commands to each pane
# # ssh each voxl 
# tmux send-keys -t $SESSION:$w.1 "ssh root@${agent1_voxl}.local" C-m
# tmux send-keys -t $SESSION:$w.2 "ssh root@${agent1_voxl}.local" C-m
# tmux send-keys -t $SESSION:$w.8 "ssh root@${agent2_voxl}.local" C-m
# tmux send-keys -t $SESSION:$w.9 "ssh root@${agent2_voxl}.local" C-m

# sleep 3

# # run voxl_voxl_connection
# for i in 1 2 8 9
# do
# 	tmux send-keys -t $SESSION:$w.$i "./nuc_voxl_connection" C-m
# done

# # ssh each nuc
# # agent 1
# for i in 3 4 5 6 7
# do 
#     tmux send-keys -t $SESSION:$w.$i "ssh ${agent1_nuc}@${agent1_ip}" C-m
# done

# # agent 2
# for i in 10 11 12 13 14
# do 
#     tmux send-keys -t $SESSION:$w.$i "ssh ${agent2_nuc}@${agent2_ip}" C-m
# done

# sleep 1

# # nuc housekeeping 
# for i in 3 4 5 6 7 10 11 12 13 14
# do 
#     # ntp date
#     tmux send-keys -t $SESSION:$w.$i "sudo ntpdate time.nist.gov" C-m
#     # ad_hoc
#     tmux send-keys -t $SESSION:$w.$i "cd && ./ad_hoc_without_NM.sh" C-m
# done

# sleep 5

# # ping each other
# tmux send-keys -t $SESSION:$w.7 "cd && ping ${agent2_adhoc_ip}" C-m
# tmux send-keys -t $SESSION:$w.14 "cd && ping ${agent1_adhoc_ip}" C-m

# # run roslaunches
# # base station
# tmux send-keys -t $SESSION:$w.0 "cd && roslaunch --wait trajectory_generator base_station.launch" C-m

# # snapros for trajectory generator (which takes some time so we do this first)
# tmux send-keys -t $SESSION:$w.2 "snapros" C-m
# tmux send-keys -t $SESSION:$w.9 "snapros" C-m
# sleep 5

# # agent1
# tmux send-keys -t $SESSION:$w.1 "fly" C-m
# tmux send-keys -t $SESSION:$w.2 "cd && roslaunch --wait trajectory_generator onboard.launch quad:=${agent1_quad}" C-m
# tmux send-keys -t $SESSION:$w.3 "cd && roslaunch --wait primer fastsam.launch quad:=${agent1_quad} is_sim:=false" C-m
# tmux send-keys -t $SESSION:$w.4 "cd && roslaunch --wait motlee_ros mapper.launch quad:=${agent1_quad} kappa:=${kappa_mot}" C-m
# tmux send-keys -t $SESSION:$w.5 "cd && roslaunch --wait motlee_ros frame_aligner.launch quad1:=${agent1_quad} quad2:=[${agent2_quad}]" C-m
# tmux send-keys -t $SESSION:$w.6 "cd && roslaunch --wait primer t265.launch quad:=${agent1_quad}" C-m

# # agent2
# tmux send-keys -t $SESSION:$w.8 "fly" C-m
# tmux send-keys -t $SESSION:$w.9 "cd && roslaunch --wait trajectory_generator onboard.launch quad:=${agent2_quad}" C-m
# tmux send-keys -t $SESSION:$w.10 "cd && roslaunch --wait primer fastsam.launch quad:=${agent2_quad} is_sim:=false" C-m
# tmux send-keys -t $SESSION:$w.11 "cd && roslaunch --wait motlee_ros mapper.launch quad:=${agent2_quad} kappa:=${kappa_mot}" C-m
# tmux send-keys -t $SESSION:$w.12 "cd && roslaunch --wait motlee_ros frame_aligner.launch quad1:=${agent2_quad} quad2:=[${agent1_quad}]" C-m
# tmux send-keys -t $SESSION:$w.13 "cd && roslaunch --wait primer t265.launch quad:=${agent2_quad}" C-m

# attach to the session
tmux -2 attach-session -t $SESSION