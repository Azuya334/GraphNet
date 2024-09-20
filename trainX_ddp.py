from __future__ import print_function

import psutil

# print("Importing ROOT")
# import ROOT as r
# print("ROOT imported")


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
bkglist = {
    # (filepath, num_events_for_training)
    0: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*pn*.root', -1)
    }

# was processed/*pn*, *0.001*, etc.

siglist = {
    # (filepath, num_events_for_training)
    1:    ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.001*.root', 200000),
    10:   ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.01*.root',  200000),
    100:  ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.1*.root',   200000),
    1000: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*1.0*.root',   200000),
    }

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

# v14 8gev:
presel_eff = {1: 0.9952855229150378, 10: 0.9976172400798192, 100: 0.9979411114121182, 1000: 0.9981519444725636, 0: 0.04734728725337247}

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


def loader_prep(rank):
    # load data
    if training_mode:
        train_data = XCalHitsDataset(siglist=siglist, bkglist=bkglist, load_range=(0, 1), nRegions=args.num_regions, extended=args.extended)
        #val_data = XCalHitsDataset(siglist=siglist, bkglist=bkglist, load_range=(0, 0.2), nRegions=args.num_regions, extended=args.extended)
        train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True,sampler=DistributedSampler(train_data,rank=rank))
        #val_loader = DataLoader(val_data, num_workers=args.num_workers, batch_size=args.batch_size,
        #                    collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)
        print('Train: %d events, Val: %d events' % (len(train_data), 0))
    input_dims = train_data.num_features
    return train_loader,input_dims

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
    model=DDP(model,device_ids=[dev],find_unused_parameters=False)
    print(next(model.parameters()).get_device())
    return model


def train(model, opt, scheduler, train_loader, dev, loss_func):
    model.train()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    count = 0
    for batch in train_loader:
        label = batch.label
        num_examples = label.shape[0]
        print("batch number:",num_batches)
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

        # tq.set_postfix({
        #     'Loss': '%.7f' % loss,
        #     'AvgLoss': '%.7f' % (total_loss / num_batches),
        #     'Acc': '%.7f' % (correct / num_examples),
        #     'AvgAcc': '%.7f' % (total_correct / count)})

        avgloss = (total_loss / num_batches)
        avgacc = (total_correct / count)

    scheduler.step()
    return avgloss, avgacc


def evaluate(model, test_loader, dev, return_scores=False):
    model.eval()

    total_correct = 0
    count = 0
    scores = []

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for batch in tq:
                label = batch.label
                num_examples = label.shape[0]
                label = label.to(dev).squeeze().long()
                logits = model(batch.coordinates.to(dev), batch.features.to(dev))
                _, preds = logits.max(1)

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

def run_epochs(rank,epochs,model,tloader):
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
        train_loss, train_acc = train(model, opt, scheduler, tloader, rank, loss_func)
        #print(f'Epoch {epoch} Validating | Batchsize={vloader.batch_size}')
        #disc_arr, valid_acc = evaluate(model, vloader, rank, return_scores=True)
        if train_acc > best_train_acc:
            best_train_acc = train_acc

        # if valid_acc > best_valid_acc:
        #     best_valid_acc = valid_acc
        #     if args.save_model_path:
        #         dirname = os.path.dirname(args.save_model_path)
        #         if dirname and not os.path.exists(dirname):
        #             os.makedirs(dirname)
        #         torch.save(model.module.state_dict(), args.save_model_path + '_state.pt')
        #         torch.save(model.module, args.save_model_path + '_full.pt')

        if(rank==1):
            torch.save(model.module.state_dict(), args.save_model_path + '_state_epoch-%d-acc-%.6f-gpu-%d.pt' % (epoch,train_acc,rank))
        print('Current train loss: %.7f' % (train_loss))
        print('Current train acc: %.7f (best: %.7f)' % (train_acc, best_train_acc))
        #print('Current validation acc: %.7f (best: %.7f)' % (valid_acc, best_valid_acc))
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        #valid_acc_list.append(valid_acc)

        # check for saturated scores
        # disc_arr = disc_arr[:,1]
        # total_scores = len(disc_arr)
        # sat_one = len(disc_arr[disc_arr==1])
        # sat_zero = len(disc_arr[disc_arr<1e-9])
        # print(f"Fraction of scores saturated at one: {sat_one/total_scores}")
        # print(f"Fraction of scores saturated below 1e-9: {sat_zero/total_scores}")

        gc.collect()
        torch.cuda.empty_cache()
        print(torch.cuda.list_gpu_processes())



def main(rank,world_size,epochs):
    init_process_group(backend="nccl",rank=rank,world_size=world_size)
    trainloader,f_dims=loader_prep(rank)
    model=model_prep(rank,f_dims)
    run_epochs(rank,epochs,model,trainloader)
    destroy_process_group()

########
if __name__ == '__main__':
    world_size=torch.cuda.device_count()
    print("World size: ",world_size)
    epoch_tot=args.num_epochs
    mp.spawn(main,args=(world_size,epoch_tot),nprocs=world_size)