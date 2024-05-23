build: 
	docker build -f /home/jtorde/Research/puma_ws/src/puma/puma/docker/Dockerfile -t puma .

build-no-cache:
	docker build --build-arg DUMMY=`date +%s` -f /home/jtorde/Research/puma_ws/src/puma/puma/docker/Dockerfile -t puma .

run-planner:
	docker run --volume=/home/jtorde/gurobi.lic:/opt/gurobi/gurobi.lic:ro -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				--rm puma planner

run-frame:
	docker run --volume=/home/jtorde/gurobi.lic:/opt/gurobi/gurobi.lic:ro -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				--rm puma frame

run-both:
	docker run --volume=/home/jtorde/gurobi.lic:/opt/gurobi/gurobi.lic:ro -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				--rm puma both