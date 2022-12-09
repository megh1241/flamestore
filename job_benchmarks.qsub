#!/bin/bash
#COBALT -A VeloC
#COBALT -n 2
#COBALT -t 0:15:00
#COBALT -q default
#COBALT --mode script
#COBALT --attrs filesystems=grand

REPO_DIR=$HOME/flamestore
INIT_SCRIPT=$REPO_DIR/init-dh-environment2.sh

mapfile -t nodes_array -d '\n' < $COBALT_NODEFILE
server_node=${nodes_array[0]}
client_node=${nodes_array[1]}

cd $REPO_DIR
source $INIT_SCRIPT

workspace=./workspace
storagepath=/home/mmadhya1
storagesize=40G
backend=master-memory
protocol=ofi+verbs
fileformat=fl
filesystem=pfs
num_layers=10
total_size='4g'
variance=0

rm -r ./workspace
echo "Creating FlameStore workspace"
mkdir ${workspace}
$mpilaunch -n 1 --host $server_node bin/flamestore init  --workspace ${workspace} \
                 --backend ${backend} \
                 --protocol ${protocol}

echo "Starting FlameStore master"
$mpilaunch -n 1 --host $server_node bin/flamestore run --master --debug --workspace ${workspace} &

echo "Starting Client application"
$mpilaunch -n 1 --host $client_node $(which python) -m mpi4py examples/resnet50/client-synthetic.py ${workspace} ${fileformat} ${filesystem} ${num_layers} ${variance} ${total_size}

echo "Shutting down FlameStore"
$mpilaunch -n 1 --host $server_node bin/flamestore shutdown --workspace=${workspace} --debug

wait
