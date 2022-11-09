docker run -itd --rm --name gym_torcs --ipc=host -e DISPLAY=$DISPLAY -p 3001:3001/udp --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix zjlqwq/gym_torcs:v1.0
