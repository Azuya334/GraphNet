from __future__ import print_function

import psutil


import numpy as np
import torch
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float64)

#DDP modules
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group



import tqdm
import os
import sys
import datetime
import argparse
import gc

from utils.ParticleNetX import ParticleNetX
from datasetX import XCalHitsDataset
from datasetX import collate_wrapper as collate_fn
from utils.SplitNetX import SplitNetX


parser = argparse.ArgumentParser()
parser.add_argument('--demo', action='store_true', default=False,
                    help='quickly test the setup by running over only a small number of events')
#parser.add_argument('--coord-ref', type=str, default='none', choices=['none', 'ecal_sp', 'target_sp', 'ecal_centroid'],
#                    help='refernce points for the x-y coordinates')
parser.add_argument('--network', type=str, default='particle-net-lite', choices=['particle-net', 'particle-net-lite', 'particle-net-k5', 'particle-net-k7'],
                    help='network architecture')
parser.add_argument('--focal-loss-gamma', type=float, default=2,
                    help='value of the gamma parameter if focal loss is used; when setting to 0, will use simple cross-entropy loss')
parser.add_argument('--save-model-path', type=str, default='models/particle_net_model',
                    help='path to save the model during training')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='number of epochs')
parser.add_argument('--optimizer', type=str, default='ranger', choices=['adam', 'ranger'],
                    help='optimizer for the training')
parser.add_argument('--start-lr', type=float, default=5e-3,
                    help='start learning rate')
parser.add_argument('--lr-steps', type=str, default='10,20',
                    help='steps to reduce the lr')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='device for the training')
parser.add_argument('--num-workers', type=int, default=2,
                    help='number of threads to load the dataset')

parser.add_argument('--predict', action='store_true', default=False,
                    help='run prediction instead of training')
parser.add_argument('--load-model-path', type=str, default='',
                    help='path to the model for prediction')
parser.add_argument('--test-sig', type=str, default='',
                    help='signal sample to be used for testing')
parser.add_argument('--test-bkg', type=str, default='',
                    help='background sample to be used for testing')
parser.add_argument('--save-extra', action='store_true', default=False,
                    help='save extra information defined in `obs_branches` and `veto_branches` to the prediction output')
parser.add_argument('--test-output-path', type=str, default='test-outputs/particle_net_output',
                    help='path to save the prediction output')
parser.add_argument('--num-regions', type=int, default=1,
                    help='Number of regions for SplitNet')
parser.add_argument('-X', '--extended', action='store_true', default=False,
                    help='Use extended ParticleNet (ECal + HCal)')

print(sys.argv)
args = parser.parse_args()


training_mode = True

###### locations of the signal and background files ######
# NOTE:  These must be output files produced by file_processor.py, not unprocessed ldmx-sw ROOT files.

#v14 8gev

# bkglist = {
#     # (filepath, num_events_for_training)
#     0: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*pn*.root', -1)
#     }

# siglist = {
#     # (filepath, num_events_for_training)
#     1:    ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.001*.root', 200000),
#     10:   ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.01*.root',  200000),
#     100:  ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.1*.root',   200000),
#     1000: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*1.0*.root',   200000),
#     }
# presel_eff = {1: 0.9952855229150378, 10: 0.9976172400798192, 100: 0.9979411114121182, 1000: 0.9981519444725636, 0: 0.04734728725337247}

#v14 8gev
# bkglist = {
#     # (filepath, num_events_for_training)
#     0: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/pn_full/*pn*.root', -1)
#     }

# siglist = {
#     # (filepath, num_events_for_training)
#     1:    ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.001*.root', 200000),
#     10:   ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.01*.root',  200000),
#     100:  ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.1*.root',   200000),
#     1000: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*1.0*.root',   200000),
#     }
# presel_eff = {1: 0.9952855229150378, 10: 0.9976172400798192, 100: 0.9979411114121182, 1000: 0.9981519444725636, 0: 0.03282988102560554}

#EaT
bkglist = {
    # (filepath, num_events_for_training)
    0: ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/pnTrain/*pn*.root', -1)
    }
siglist = {
    # (filepath, num_events_for_training)
    1:    ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/sigTrain/*0.001*.root', 40000),
    10:   ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/sigTrain/*0.01*.root',  40000),
    100:  ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/sigTrain/*0.1*.root',   40000),
    1000: ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/sigTrain/*1.0*.root',   40000),
    }
presel_eff={1: 0.9988621022179364, 10: 0.999275930896827, 100: 0.9991549148445952, 1000: 0.9991183067080328, 0: 0.05252190640746514}

if args.demo:
    bkglist = {
        # (filepath, num_events_for_training)
        0: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*pn*.root', 8000)
        }

    siglist = {
        # (filepath, num_events_for_training)
        1:    ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.001*.root', 2000),
        10:   ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.01*.root',  2000),
        100:  ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.1*.root',   2000),
        1000: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*1.0*.root',   2000),
        }



#########################################################

###### `observer` variables to be saved in the prediction output ######
# NOTE:  Now unnecessary; only required branches are saved in the input files.
# -> have to modify+rerun preprocessing script if new vars are required

obs_branches = []
#veto_branches = []
if args.save_extra:
    # List of extra branches to save for plotting information
    # Should match everything in plot_ldmx_nn.ipynb
    # EXCEPT for ParticleNet_extra_label and ParticleNet_disc, which are computed after training
    obs_branches = [
        'discValue_',
        'recoilX_',
        'recoilY_',
        'TargetSPRecoilE_pt',
        'maxPE'
        ]


#########################################################
#########################################################



# model parameter
if args.network == 'particle-net':
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]
elif args.network == 'particle-net-lite':
    conv_params = [
        (7, (32, 32, 32)),
        (7, (64, 64, 64))
        ]
    fc_params = [(128, 0.1)]
elif args.network == 'particle-net-k5':
    conv_params = [
        (5, (64, 64, 64)),
        (5, (128, 128, 128)),
        (5, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]
elif args.network == 'particle-net-k7':
    conv_params = [
        (7, (64, 64, 64)),
        (7, (128, 128, 128)),
        (7, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]

print('conv_params: %s' % conv_params)
print('fc_params: %s' % fc_params)


def loader_prep(rank,worldsize):
    # load data
    if training_mode:
        train_data = XCalHitsDataset(siglist=siglist, bkglist=bkglist, load_range=(0.2, 1), nRegions=args.num_regions, extended=args.extended)
        train_sampler=DistributedSampler(train_data,num_replicas=worldsize,rank=rank) #New distributed sampler that automatically splits data among allocated gpus
        train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True,sampler=train_sampler) #Dataloader split data using sampler
        
        val_data = XCalHitsDataset(siglist=siglist, bkglist=bkglist, load_range=(0, 0.2), nRegions=args.num_regions, extended=args.extended)
        #val_sampler=DistributedSampler(val_data,num_replicas=worldsize,rank=rank)
        val_loader = DataLoader(val_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True) #same copy of val_loader on each gpu
        print('Train: %d events, Val: %d events' % (len(train_data), len(val_data)))
    input_dims = train_data.num_features
    return train_loader, val_data,val_loader ,input_dims

def model_prep(dev,input_dims):
    # model
    print("Initializing model")
    # Create the SplitNet model here.  This is the "real" ParticleNet.
    model = SplitNetX(input_dims=input_dims, num_classes=2,
                    conv_params=conv_params,
                    fc_params=fc_params,
                    use_fusion=True,
                    nRegions=args.num_regions)
    model = model.to(dev)
    print("GPU number: ",dev)
    ddp_model=DDP(model,device_ids=[dev],output_device=dev,find_unused_parameters=True) #wraps DDP model around splitnet
    return ddp_model


def train(model, opt, scheduler, train_loader, dev, loss_func):
    model.train()

    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    with tqdm.tqdm(train_loader) as tq:
        for batch in tq:
            label = batch.label
            num_examples = label.shape[0]
            label = label.to(dev).squeeze().long()
            opt.zero_grad()
            logits = model(batch.coordinates.to(dev), batch.features.to(dev))
            loss = loss_func(logits, label)
            loss.backward()
            opt.step()

            _, preds = logits.max(1)

            num_batches += 1
            count += num_examples
            loss = loss.item()
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'Loss': '%.7f' % loss,
                'AvgLoss': '%.7f' % (total_loss / num_batches),
                'Acc': '%.7f' % (correct / num_examples),
                'AvgAcc': '%.7f' % (total_correct / count)})

            avgloss = (total_loss / num_batches)
            avgacc = (total_correct / count)

    scheduler.step()
    return avgloss, avgacc


def evaluate(model, test_loader, dev, return_scores=False):
    model.eval()

    total_correct = 0
    count = 0
    scores = []
    i=0
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch in tq:
                label = batch.label
                num_examples = label.shape[0]
                label = label.to(dev).squeeze().long()
                logits = model(batch.coordinates.to(dev), batch.features.to(dev))
                _, preds = logits.max(1)

                if(i<5):
                    print(f"label: {label}, shape: {label.shape}")
                    print(f"logit shape {logits.shape}")
                    print(f"logit: {logits}")
                    print(f"preds: {preds}, shape: {preds.shape}")

                if return_scores:
                    #log_scores = torch.nn.functional.log_softmax(logits, dim=1).cpu().detach().numpy()
                    #scores.append(np.exp(np.longdouble(log_scores)))
                    log_scores = torch.nn.functional.log_softmax(logits, dim=1)
                    scores.append(torch.exp(log_scores).cpu().detach().numpy())
                    #scores.append(torch.softmax(logits, dim=1).cpu().detach().numpy())

                correct = (preds == label).sum().item()
                total_correct += correct
                count += num_examples

                tq.set_postfix({
                    'Acc': '%.7f' % (correct / num_examples),
                    'AvgAcc': '%.7f' % (total_correct / count)})

    if return_scores:
        return np.concatenate(scores), (total_correct / count)
    else:
        return total_correct / count

def run_epochs(rank,epochs,model,tloader,test_data,vloader,local_rank):
    # optimizer & learning rate
    if args.optimizer == 'adam':
        opt = torch.optim.Adam(model.module.parameters(), lr=args.start_lr)
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_steps, gamma=0.1)
    else:
        from utils.ranger import Ranger
        opt = Ranger(model.parameters(), lr=args.start_lr)
        lr_decay_epochs = int(args.num_epochs * 0.3)
        lr_decay_rate = 0.01 ** (1. / lr_decay_epochs)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(range(args.num_epochs - lr_decay_epochs, args.num_epochs)), gamma=lr_decay_rate)
    # loss function
    if args.focal_loss_gamma > 0:
        print('Using focal loss w/ gamma=%s' % args.focal_loss_gamma)
        from utils.focal_loss import FocalLoss
        loss_func = FocalLoss(gamma=args.focal_loss_gamma)
    else:
        loss_func = torch.nn.CrossEntropyLoss()
    # training loop
    best_train_acc = 0
    best_valid_acc = 0
    train_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(epochs):
        print(f'Epoch {epoch} Training | Batchsize={tloader.batch_size}')
        train_loss, train_acc = train(model, opt, scheduler, tloader, local_rank, loss_func) #Any func using to(dev) needs to use local_rank to match gpu id
        print(f'Epoch {epoch} Validating | Batchsize={vloader.batch_size}')
        disc_arr, valid_acc = evaluate(model, vloader, local_rank, return_scores=True) #same here
        if train_acc > best_train_acc:
            best_train_acc = train_acc

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            if rank==1 and args.save_model_path:
                dirname = os.path.dirname(args.save_model_path)
                if dirname and not os.path.exists(dirname):
                    os.makedirs(dirname)
                torch.save(model.module.state_dict(), args.save_model_path + '_state.pt')
                torch.save(model.module, args.save_model_path + '_full.pt')

        if(rank==1): #saves model from gpu 1 and 5 (check if they are the same)
            torch.save(model.module.state_dict(), args.save_model_path + '_state_epoch-%d-acc-%.6f-gpu-%d.pt' % (epoch,valid_acc,rank))
        print('Current train loss: %.7f' % (train_loss))
        print('Current train acc: %.7f (best: %.7f)' % (train_acc, best_train_acc))
        print('Current validation acc: %.7f (best: %.7f)' % (valid_acc, best_valid_acc))
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        # check for saturated scores
        disc_arr = disc_arr[:,1]
        total_scores = len(disc_arr)
        sat_one = len(disc_arr[disc_arr==1])
        sat_zero = len(disc_arr[disc_arr<1e-9])
        print(f"Fraction of scores saturated at one: {sat_one/total_scores}")
        print(f"Fraction of scores saturated below 1e-9: {sat_zero/total_scores}")

        gc.collect()
        torch.cuda.empty_cache()
        #print(torch.cuda.list_gpu_processes()) #Not working for some reason
    if(rank==1):
        model_path = args.load_model_path if args.predict else args.save_model_path
        if not model_path.endswith('.pt'):
            model_path += '_state.pt'
        print('Loading model %s for eval' % model_path)
        model.load_state_dict(torch.load(model_path),strict=False)

        # evaluate model on test dataset
        path, name = os.path.split(args.test_output_path)
        if path and not os.path.exists(path):
            os.makedirs(path)

        test_preds, _ = evaluate(model, vloader, local_rank, return_scores=True)
        test_labels = test_data.label
        #test_labels=np.array([np.array([1,0]) if i==0 else [0,1] for i in test_labels])
        test_extra_labels = test_data.extra_labels

        info_dict = {'model_name':args.network,
                    'model_params': {'conv_params':conv_params, 'fc_params':fc_params},
                    'initial LR': args.start_lr,
                    'batch size': args.batch_size,
                    'date': str(datetime.date.today()),
                    'model_path': args.load_model_path,
                    'siglist': siglist,
                    'bkglist': bkglist,
                    'presel_eff': presel_eff,  #test_data.presel_eff,
                    }
        if training_mode:
            info_dict.update({'model_path': args.save_model_path})
        
        # auto generated plots for ROC, train/val acc, and train loss
        from utils.plot_utils import plotROC, get_signal_effs, plot_acc, plot_loss
        for k in siglist:
            if k > 0:
                mass = '%d MeV' % k
                fpr, tpr, auc, acc = plotROC(test_preds, test_labels, sample_weight=np.logical_or(test_extra_labels == 0, test_extra_labels == k),
                                            sig_eff=presel_eff[k], bkg_eff=presel_eff[0],
                                            output=os.path.splitext(args.test_output_path)[0] + 'ROC_%s.pdf' % mass, label=mass, xlim=[1e-7, .01], ylim=[0, 1], logx=True)
                info_dict[mass] = {'auc-presel': auc,
                                'acc-presel': acc,
                                'effs': get_signal_effs(fpr, tpr)
                                }

        if training_mode:
            plot_acc(tacc=train_acc_list, vacc=valid_acc_list, epoch=epochs,output=os.path.splitext(args.test_output_path)[0] + 'acc_plot.pdf')
            plot_loss(tloss=train_loss_list, epoch=epochs,output=os.path.splitext(args.test_output_path)[0] + 'loss_plot.pdf')

            # save train accuracy, validation accuracy, and train loss in a pickle file (can custom plot later)
            import pickle
            df_dict = {'train_acc': train_acc_list, 'val_acc': valid_acc_list, 'train_loss': train_loss_list}
            with open(os.path.splitext(args.test_output_path)[0] + 'df.pkl', 'wb') as pklf:
                pickle.dump(df_dict, pklf)

        print(' === Summary ===')
        for k in info_dict:
            print('%s: %s' % (k, info_dict[k]))

        info_file = os.path.splitext(args.test_output_path)[0] + '_INFO.txt'
        with open(info_file, 'w') as f:
            for k in info_dict:
                f.write('%s: %s\n' % (k, info_dict[k]))

        print("SAVING OUTPUT")
        # save prediction output
        import awkward
        pred_file = os.path.splitext(args.test_output_path)[0] + '_OUTPUT'
        out_data = test_data.get_obs_data()  #test_data.obs_data
        out_data['ParticleNet_extra_label'] = test_extra_labels
        out_data['ParticleNet_disc'] = test_preds[:, 1].astype(np.float64)
        # OUTDATED:
        # awkward.save(pred_file, out_data, mode='w')
        #import pyarrow.parquet as pq
        out_data = awkward.copy(awkward.Array(out_data))  # NOW trying a direct conversion from dict...
        # The copy may make the memory continguous...
        # Confirm that recoilX is nonzero...
        print("Sending to parquet")
        awkward.to_parquet(out_data, pred_file+'.parquet')


from socket import gethostname
def main(epochs):
    world_size=int(os.environ["WORLDSIZE"]) #total number of processes (gpus)
    rank=int(os.environ["SLURM_PROCID"]) #slurm assigns process id e.g. (0-4)
    gpu_per_node=torch.cuda.device_count() 
    local_rank=rank-gpu_per_node*(rank//gpu_per_node) #gpu id on node (0-1 for 2 node with 2 gpus each)
    print(f"Hello from rank {rank} and local rank {local_rank} of {world_size} on {gethostname()}", flush=True)

    init_process_group(backend="nccl",rank=rank,world_size=world_size) #Process group for message passing
    trainloader,valdata,valloader,f_dims=loader_prep(rank,world_size) #Initiate dataloaders
    model=model_prep(local_rank,f_dims) #Inititate ddp model
    run_epochs(rank,epochs,model,trainloader,valdata,valloader,local_rank) #Main train loop
    destroy_process_group() #end process group

########
if __name__ == '__main__':
    #This function will be ran once on each gpu
    start=datetime.datetime.now()
    epoch_tot=args.num_epochs
    main(epoch_tot)
    end=datetime.datetime.now()
    print(f"Started {start}")
    print(f"Finished {end}")
    print(f"Duration {end-start}")
