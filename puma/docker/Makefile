build: 
	docker build -t puma .

build-no-cache:
	docker build --build-arg DUMMY=`date +%s` -t puma .

run-planner:
	docker run  -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				--rm puma planner

run-frame:
	docker run  -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				--rm puma frame

run-multiagent:
	docker run  -it \
				--privileged \
				--gpus all --env="NVIDIA_DRIVER_CAPABILITIES=all" \
				-e DISPLAY=${DISPLAY} \
				--net=host --env="DISPLAY" \
				--env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
				--rm puma multiagent