#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurmlogs/slurm_eat_eval_lite.out

export MODEL_NAME=ddp_eat_ecal_lite_1reg
export DATA_NAME=ddp_eat_ecal_lite_1reg
export BASE_DIRECTORY=/home/zwan/LDMX/LDMX-scripts/GraphNet

cd $SLURM_SUBMIT_DIR

/bin/hostname
srun python -u evalX_new.py --save-extra --network particle-net-lite --batch-size 512 --load-model-path $BASE_DIRECTORY/models/$MODEL_NAME/model_state.pt --test-bkg '/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/pnEval/*.root' --test-sig '/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/sigEval/*.root' --test-output-path $BASE_DIRECTORY/plot_data/$DATA_NAME --num-workers 24 --device 'cuda:0' --num-regions 1 
