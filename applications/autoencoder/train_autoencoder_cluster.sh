#!/bin/bash

function print_usage {
        echo -e "Usage:\t train_autoencoder_cluster.sh CLUSTER NETHZ_USERNAME NUM_CORES RUN_TIME MEM_PER_CORE\n"
        echo -e "Arguments:\n"
        echo -e "CLUSTER\t\t\t Name of the cluster on which the jupyter notebook should be started (Euler or LeoOpen)"
        echo -e "NETHZ_USERNAME\t\tNETHZ username for which the notebook should be started"
        echo -e "NUM_CORES\t\tNumber of cores to be used on the cluster (<36)"
        echo -e "RUN_TIME\t\tRun time limit for the jupyter notebook on the cluster (HH:MM)"
        echo -e "MEM_PER_CORE\t\tMemory limit in MB per core\n"
        echo -e "NUM_GPUS\t\Number  of gpus to be used on the cluster\n"
        echo -e "Example:\n"
        echo -e "./start_jupyter_nb.sh Euler sfux 4 01:20 2048 1\n"
}

# if number of command line arguments is different from 5 or if $1==-h or $1==--help
if [ "$#" !=  6 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    print_usage
    exit
fi

# Parse and check command line arguments (cluster, NETHZ username, number of cores, run time limit, memory limit per NUM_CORES)

# check on which cluster the script should run and load the proper python module
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
CONDAENV="geometric_p36"


echo -e "\nCluster: $CLUSTERNAME"

# no need to do checks on the username. If it is wrong, the SSH commands will not work
USERNAME="$2"
echo -e "NETHZ username: $USERNAME"

# number of cores to be used
NUM_CORES=$3

# check if NUM_CORES is an integer
if ! [[ "$NUM_CORES" =~ ^[0-9]+$ ]]; then
    echo -e "Incorrect format. Please specify number of cores as an integer and try again.\n"
    print_usage
    exit
fi

# check if NUM_CORES is <= 36
if [ "$NUM_CORES" -gt "36" ]; then
    echo -e "No distributed memory supported, therefore number of cores needs to be smaller or equal to 36.\n"
    print_usage
    exit
fi
echo -e "Jupyter notebook will run on $NUM_CORES cores"

# run time limit
RUN_TIME="$4"

# check if RUN_TIME is provided in HH:MM format
if ! [[ "$RUN_TIME" =~ ^[0-9][0-9]:[0-9][0-9]$ ]]; then
    echo -e "Incorrect format. Please specify runtime limit in the format HH:MM and try again\n"
    print_usage
    exit
else
    echo -e "Run time limit set to $RUN_TIME"
fi

# memory per core
MEM_PER_CORE=$5

# check if MEM_PER_CORE is an integer
if ! [[ "$MEM_PER_CORE" =~ ^[0-9]+$ ]]
    then
        echo -e "Memory limit must be an integer, please try again\n"
        print_usage
        exit
fi
echo -e "Memory per core set to $MEM_PER_CORE MB"

# number of cores to be used
NUM_GPUS=$6

# check if NUM_CORES is an integer
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
    echo -e "Incorrect format. Please specify number of gpus as an integer and try again.\n"
    print_usage
    exit
fi
echo -e "Using $NUM_GPUS GPUS\n"

ssh $USERNAME@$CHOSTNAME bsub -n $NUM_CORES -W $RUN_TIME -G ls_grossm  -R "rusage[mem=$MEM_PER_CORE,ngpus_excl_p=$NUM_GPUS]" -R 'select[gpu_model0==GeForceRTX2080Ti]' <<ENDBSUB
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDAENV
#cd ~/Repos/gdl/applications/autoencoder
cd ~/Repos/gdl
#python train_autoencoder.py --split sliced --split_term sliced
python applications/autoencoder/train_autoencoder.py --split sliced --split_term sliced
#python applications/autoencoder/train_autoencoder.py --split expression --split_term bareteeth
#python applications/autoencoder/train_autoencoder.py --split identity --split_term FaceTalk_170731_00024_TA
ENDBSUB
