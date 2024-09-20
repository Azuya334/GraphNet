#!/bin/bash
#SBATCH -N 1 --partition=gpu --ntasks-per-node=4 --gres=gpu:4
#SBATCH --output=slurmlogs/eval_lite_ecal_1reg_mp.out

export MODEL_NAME=ddp_ecal_1reg
export DATA_NAME=ddp_ecal_1reg_mpeval
export BASE_DIRECTORY=/home/zwan/LDMX/LDMX-scripts/GraphNet
export WORLDSIZE=4

cd $SLURM_SUBMIT_DIR

/bin/hostname
mount $BASE_DIRECTORY/plot_data
#srun python -u mp_eval.py --device 'cuda:0' --save-extra --network particle-net-lite --batch-size 512 --load-model-path $BASE_DIRECTORY/models/$MODEL_NAME/model_state.pt --test-bkg '/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/pnEval/*.root' --test-sig '/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/sigEval/*.root' --test-output-path $BASE_DIRECTORY/plot_data/$DATA_NAME --num-workers 24 --num-regions 2
srun python -u mp_eval.py --device 'cuda:0' --save-extra --network particle-net-lite --batch-size 512 --load-model-path $BASE_DIRECTORY/models/$MODEL_NAME/model_state.pt --test-bkg '/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/pn_full/bkg_eval/*.root' --test-sig '/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/sig_eval/*.root' --test-output-path $BASE_DIRECTORY/plot_data/$DATA_NAME --num-workers 24 --num-regions 1
