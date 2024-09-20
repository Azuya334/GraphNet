import numpy as np
import torch
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float64)

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


print(sys.argv)
args = parser.parse_args()

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

#EaT
bkglist = {
    # (filepath, num_events_for_training)
    0: ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/pnTrain/*pn*.root', -1)
    }
siglist = {
    # (filepath, num_events_for_training)
    1:    ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/*0.001*.root', 40000),
    10:   ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/*0.01*.root',  40000),
    100:  ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/*0.1*.root',   40000),
    1000: ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/*1.0*.root',   40000),
    }
presel_eff={1: 0.9988621022179364, 10: 0.999275930896827, 100: 0.9991549148445952, 1000: 0.9991183067080328, 0: 0.05252190640746514}

training_mode=True
def loader_prep():
    # load data
    val_data = XCalHitsDataset(siglist=siglist, bkglist=bkglist, load_range=(0, 0.2), nRegions=args.num_regions, extended=args.extended)
    val_loader = DataLoader(val_data, num_workers=args.num_workers, batch_size=args.batch_size,
                            collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True) #same copy of val_loader on each gpu
    input_dims = val_data.num_features
    return val_data,val_loader,input_dims

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
    return model

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

                #if(i<5):
                    #print(f"label: {label}, shape: {label.shape}")
                    #print(f"logit shape {logits.shape}")
                    #print(f"logit: {logits}")
                    #print(f"preds: {preds}, shape: {preds.shape}")

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

def main():
    dev = torch.device(args.device)
    test_data,test_loader,dims=loader_prep()
    model=model_prep(dev,dims)

    # load saved model
    model_path = args.save_model_path+'_state.pt'

    print('Loading model %s for eval' % model_path)
    model.load_state_dict(torch.load(model_path,map_location='cuda:0'),strict=False)

    # evaluate model on test dataset

    test_preds, _ = evaluate(model, test_loader, dev, return_scores=True)
    test_labels = test_data.label
    print(test_preds)
    print(test_labels,test_labels.dtype)
    test_labels=np.array([np.array([1,0]) if i==0 else [0,1] for i in test_labels])
    print(test_labels)
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
    from utils.plot_utils import plotROC, get_signal_effs
    for k in siglist:
        if k > 0:
            mass = '%d MeV' % k
            print(f"test_preds: {test_preds} shape {test_preds.shape}")
            print(f"test_labels: {test_labels} shape {test_labels.shape}")
            print(mass)
            fpr, tpr, auc, acc = plotROC(test_preds, test_labels, sample_weight=np.logical_or(test_extra_labels == 0, test_extra_labels == k),
                                            sig_eff=presel_eff[k], bkg_eff=presel_eff[0],
                                            output=os.path.splitext(args.test_output_path)[0] + 'ROC_%s.pdf' % mass, label=mass, xlim=[1e-7, .01], ylim=[0, 1], logx=True)
            info_dict[mass] = {'auc-presel': auc,
                                'acc-presel': acc,
                                'effs': get_signal_effs(fpr, tpr)
                                }

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

    print("DONE")

main()
