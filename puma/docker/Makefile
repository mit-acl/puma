build:
	docker build -t puma .

build-no-cache:
	docker build --build-arg DUMMY=`date +%s` -t puma .

# Mount the local puma sources the demos actually use, so you can edit them and
# re-run without rebuilding the image. PUMA_PKG is the inner puma package dir
# (this Makefile lives in <pkg>/docker). param/ is mounted as a directory (not a
# single file) because the demo does an in-place `sed` on puma.yaml, and `sed -i`
# fails on single-file bind mounts.
PUMA_PKG := $(CURDIR)/..
DEV_MOUNTS := \
	--volume="$(PUMA_PKG)/other/demos:/home/code/ws/src/puma/puma/other/demos" \
	--volume="$(PUMA_PKG)/scripts/dynamic_corridor.py:/home/code/ws/src/puma/puma/scripts/dynamic_corridor.py" \
	--volume="$(PUMA_PKG)/param:/home/code/ws/src/puma/puma/param" \
	--volume="$(PUMA_PKG)/matlab/casadi_generated_files:/home/code/ws/src/puma/puma/matlab/casadi_generated_files"

# The GUI targets below run rviz/gazebo inside the container as root and connect to
# your host X server over the mounted /tmp/.X11-unix socket. A reboot or new login
# re-enables X access control (only localuser:<you> allowed), which silently blocks
# the container -> rviz never opens. This prerequisite re-grants local docker access
# before each GUI run. The leading '-' makes it non-fatal if there's no X server.
xhost-allow:
	-@xhost +local:root >/dev/null 2>&1 || true

run-planner: xhost-allow
	docker run  -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				$(DEV_MOUNTS) \
				--rm puma planner

# Same as run-planner, but records the whole run (from the start) to a bag that
# finalizes cleanly on its own -> other/demos/data/bags/run.bag. One terminal, no
# timing games, no ".active" bag. Replay with: make replay-bag BAG=run.bag
run-planner-record: xhost-allow
	docker run  -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				-e PUMA_RECORD_BAG=true \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				$(DEV_MOUNTS) \
				--rm puma planner

run-frame: xhost-allow
	docker run  -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				$(DEV_MOUNTS) \
				--rm puma frame

run-multiagent: xhost-allow
	docker run  -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				$(DEV_MOUNTS) \
				--rm puma multiagent

# Bags live in <pkg>/other/demos/data/bags (on the host, so they persist).
BAGS_DIR := $(PUMA_PKG)/other/demos/data/bags
BAG ?= puma_demo.bag

# Record a bag of the CURRENTLY-RUNNING sim. Run `make run-planner` in one
# terminal, then `make record-bag` in another; Ctrl-C to stop the recording.
# Records everything except camera image streams. Override the name: make record-bag BAG=myrun.bag
record-bag:
	mkdir -p $(BAGS_DIR)
	docker run -it --rm --net=host \
				--volume="$(BAGS_DIR):/bags" \
				--entrypoint bash puma -c "source /opt/ros/noetic/setup.bash; source /home/code/ws/devel/setup.bash; cd /bags; echo 'Recording to /bags/$(BAG) ... Ctrl-C to stop.'; rosbag record -a -x '(.*)(image|points|pcloud)(.*)' -O $(BAG)"

# Replay a recorded bag + rviz (for screen-recording a video). Ctrl-C to stop.
# Pick a bag: make replay-bag BAG=puma_demo.bag   (defaults to puma_demo.bag)
replay-bag: xhost-allow
	docker run -it --rm \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				--volume="$(BAGS_DIR):/bags" \
				--volume="$(PUMA_PKG)/rviz_cfgs:/home/code/ws/src/puma/puma/rviz_cfgs" \
				--volume="$(CURDIR)/puma_replay.sh:/puma_replay.sh" \
				--entrypoint bash puma /puma_replay.sh /bags/$(BAG)
