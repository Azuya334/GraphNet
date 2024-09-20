import awkward

import numpy as np
import torch
from torch.utils.data import DataLoader
torch.set_default_dtype(torch.float64)
from datasetX import XCalHitsDataset


obs_branches = []
#v14 8gev
# bkglist = {
#     # (filepath, num_events_for_training)
#     0: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/pn_full/*pn*.root', 8000)
#     }

# siglist = {
#     # (filepath, num_events_for_training)
#     1:    ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.001*.root', 2000),
#     10:   ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.01*.root',  2000),
#     100:  ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*0.1*.root',   2000),
#     1000: ('/home/duncansw/GraphNet_input/v14/8gev/v3_tskim/XCal_total/*1.0*.root',   2000),
#     }
# presel_eff = {1: 0.9952855229150378, 10: 0.9976172400798192, 100: 0.9979411114121182, 1000: 0.9981519444725636, 0: 0.03282988102560554}
#EaT
bkglist = {
    # (filepath, num_events_for_training)
    0: ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/pnTrain/*pn*.root', 8000)
    }
siglist = {
    # (filepath, num_events_for_training)
    1:    ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/*0.001*.root', 2000),
    10:   ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/*0.01*.root',  2000),
    100:  ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/*0.1*.root',   2000),
    1000: ('/home/zwan/LDMX/LDMX-scripts/GraphNet/input/eat/*1.0*.root',   2000),
    }
presel_eff={1: 0.9988621022179364, 10: 0.999275930896827, 100: 0.9991549148445952, 1000: 0.9991183067080328, 0: 0.05252190640746514}




test_data = XCalHitsDataset(siglist=siglist, bkglist=bkglist, load_range=(0, 1), obs_branches=obs_branches, nRegions=1, extended=False)

test_labels = test_data.label
test_extra_labels = test_data.extra_labels

print(test_labels.shape)
print(test_labels)