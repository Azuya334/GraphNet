#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1 --gres=gpu:1
#SBATCH --output=slurmlogs/slurm_infer4.out

export MODEL_NAME=ddp_lite_eat_ecal
export BASE_DIRECTORY=/home/zwan/LDMX/LDMX-scripts/GraphNet 


cd $SLURM_SUBMIT_DIR

/bin/hostname
srun python -u infer_test.py --optimizer ranger --start-lr 5e-7 --focal-loss-gamma 2 --network particle-net-lite --batch-size 512 --save-model-path $BASE_DIRECTORY/models/$MODEL_NAME/model --test-output-path $BASE_DIRECTORY/test_output/$MODEL_NAME/output --num-epochs 20 --num-workers 24 --num-regions 2 --device 'cuda:0'
