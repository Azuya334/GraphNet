#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=4 --gres=gpu:4
#SBATCH --output=slurmlogs/training_ddp_eat_ecal_lite_1reg.out  

export MODEL_NAME=ddp_eat_ecal_lite_1reg
export BASE_DIRECTORY=/home/zwan/LDMX/LDMX-scripts/GraphNet

export WORLDSIZE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1) 
export MASTER_PORT=38093

echo "MASTER_ADDR="$MASTER_ADDR
echo "Model name="$MODEL_NAME
cd $SLURM_SUBMIT_DIR

/bin/hostname
srun python -u ddp_nomp.py --optimizer ranger --start-lr 5e-3 --focal-loss-gamma 2 --network particle-net-lite --batch-size 512 --save-model-path $BASE_DIRECTORY/models/$MODEL_NAME/model --test-output-path $BASE_DIRECTORY/test_output/$MODEL_NAME/output --num-epochs 5 --num-workers 24 --num-regions 1
