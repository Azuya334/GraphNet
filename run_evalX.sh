#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurmlogs/eval_full_1reg.out

export MODEL_NAME=ddp_ecal_lite_1reg
export DATA_NAME=ddp_ecal_lite_1reg
export BASE_DIRECTORY=/home/zwan/LDMX/LDMX-scripts/GraphNet

cd $SLURM_SUBMIT_DIR

/bin/hostname
srun --gres=gpu:1 python -u evalX.py --save-extra --network particle-net-lite --batch-size 512 --load-model-path $BASE_DIRECTORY/models/$MODEL_NAME/model_state_epoch-14_acc-0.9869.pt --test-bkg '/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/pn_full/bkg_eval/*.root' --test-sig '/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/sig_eval/*.root' --test-output-path $BASE_DIRECTORY/plot_data/$DATA_NAME --num-workers 16 --device 'cuda:0' --num-regions 1 
