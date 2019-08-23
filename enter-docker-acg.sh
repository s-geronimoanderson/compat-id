#!/usr/bin/env bash

docker_options=$''

# Before this, run X11.app and set "allow connections from the network" (in
# Preferences-Security).
# SIGA 8/3: Disabling for now.

# Find your IPv4 address (ifconfig | grep inet) and store it:
#ip=$(ifconfig | grep -e 'inet ' | grep --invert-match -e '127.0.0.1' | cut -d' ' -f2)

# Add your IP to X11's access control list:
#xhost + $ip

# Expose X11 to container software:
#docker_options+=$' '"-e DISPLAY=${ip}:0 -v /tmp/.X11-unix:/.X11-unix"

# Get ready.
achax_root="${HOME}/vcs/AChax"
achax_docker_root="${HOME}/vcs/AChax/Docker"
achax_user_name="${USER}"
achax_group_name="$(id -g -n ${USER})"

# Options for running as our desired user and group:
docker_options+=$' '"-u ${achax_user_name} -w /home/${achax_user_name}"

# Map the AChax directory:
docker_options+=$' '"--mount src=${HOME}/vcs,target=/home/${achax_user_name}/vcs,type=bind"

# Make the container interactive:
docker_options+=$' '"-i -t"

# SIGA 8/3: I think this only limits memory, so omitting gives full.
#docker_options+=$' '"-m=12g"

# Desired image and command:
docker_options+=$' '"achax:latest bash"

# Finally, run Docker with the image and options:
docker run ${docker_options}

