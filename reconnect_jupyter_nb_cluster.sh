#!/bin/bash

# function to print usage instructions
function print_usage {
        echo -e "Usage:\t start_jupyter_nb.sh CLUSTER NETHZ_USERNAME \n"
        echo -e "Arguments:\n"
        echo -e "CLUSTER\t\t\t Name of the cluster on which the jupyter notebook should be started (Euler or LeoOpen)"
        echo -e "NETHZ_USERNAME\t\tNETHZ username for which the notebook should be started"
        echo -e "./reconnect_jupyter_nb_cluster.sh LeoOpen rdanecek\n"
}

# if number of command line arguments is different from 5 or if $1==-h or $1==--help
if [ "$#" !=  2 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    print_usage
    exit
fi

CLUSTERNAME="$1"

if [ "$CLUSTERNAME" == "Euler" ]; then
    CHOSTNAME="euler.ethz.ch"
    #PCOMMAND="new gcc/4.8.2 r/3.6.0 python/3.6.1 eth_proxy"
elif [ "$CLUSTERNAME" == "LeoOpen" ]; then
    CHOSTNAME="login.leonhard.ethz.ch"
    #PCOMMAND="python_gpu/3.7.1 eth_proxy hdf5/1.10.1"
else
    echo -e "Incorrect cluster name. Please specify Euler or LeoOpen as cluster and and try again.\n"
    print_usage
    exit
fi

echo -e "\nCluster: $CLUSTERNAME"

# no need to do checks on the username. If it is wrong, the SSH commands will not work
USERNAME="$2"
echo -e "NETHZ username: $USERNAME"

remoteip=$(ssh $USERNAME@$CHOSTNAME "cat /cluster/home/$USERNAME/jnbip | grep -m1 'Remote IP' | cut -d ':' -f 2")
remoteport=$(ssh $USERNAME@$CHOSTNAME "cat /cluster/home/$USERNAME/jnbinfo | grep -m1 token | cut -d '/' -f 3 | cut -d ':' -f 2")
jnbtoken=$(ssh $USERNAME@$CHOSTNAME "cat /cluster/home/$USERNAME/jnbinfo | grep -m1 token | cut -d '=' -f 2")

if  [[ "$remoteip" == "" ]]; then
    echo -e "Error: remote ip is not defined. Terminating script."
    echo -e "Please login to the cluster and check with bjobs if the batch job is still running."
    exit 1
fi

if  [[ "$remoteport" == "" ]]; then
    echo -e "Error: remote port is not defined. Terminating script."
    echo -e "Please login to the cluster and check with bjobs if the batch job is still running."
    exit 1
fi

if  [[ "$jnbtoken" == "" ]]; then
    echo -e "Error: token for the jupyter notebook is not defined. Terminating script."
    echo -e "Please login to the cluster and check with bjobs if the batch job is still running."
    exit 1
fi

echo -e "Remote IP address: $remoteip"
echo -e "Remote port: $remoteport"
echo -e "Jupyter token: $jnbtoken"

# get a free port on local computer
echo -e "Determining free port on local computer"
PORTN=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo -e "Local port: $PORTN"

# setup SSH tunnel from local computer to compute node via login node
echo -e "Setting up SSH tunnel for connecting the browser to the jupyter notebook"
ssh $USERNAME@$CHOSTNAME -L $PORTN:$remoteip:$remoteport -N &

# SSH tunnel is started in the background, pause 5 seconds to make sure
# it is established before starting the browser
sleep 5

# save url in variable
nburl=http://localhost:$PORTN/?token=$jnbtoken
echo -e "Starting browser and connecting it to jupyter notebook"
echo -e "Connecting to url "$nburl

if [[ "$OSTYPE" == "linux-gnu" ]]; then
	xdg-open $nburl
elif [[ "$OSTYPE" == "darwin"* ]]; then
	open $nburl
else
	echo -e "Your operating system does not allow to start the browser automatically."
        echo -e "Please open $nburl in your browser."
fi
