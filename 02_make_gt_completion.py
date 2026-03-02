
from train_dataloader import EdgesDataset
import yaml
import torch

# just run this, it will process dummy_completion/incomplete.pkl to get xyz ground truth, 
# saved to "gt_xyz.pt" to bypass prediction in input partial mesh. 

with open("configs/tmgpt.yaml","r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

TRAIN_PATH = "dummy_completion"
VAL_PATH = "dummy_completion"

quant_bit = config["quant_bit"]

train_dataset = EdgesDataset(TRAIN_PATH, quant_bit=quant_bit)
val_dataset = EdgesDataset(VAL_PATH, quant_bit=quant_bit)


gt_ind = train_dataset[0]['gt_ind'].unsqueeze(0)
edges = train_dataset[0]['edges'].unsqueeze(0)
vertices = train_dataset[0]['vertices'].unsqueeze(0)

# construct mask  
##########################      
gt_ind[gt_ind==-2] = -1

indices = (gt_ind != -1).cumsum(dim=1).eq((gt_ind != -1).sum(dim=1, keepdim=True)) & (gt_ind != -1)
non_zeros = indices.nonzero()

#if non_zeros.size(0) == 0:
#    a = self.sos_emb * 0
#    return a.sum(), a.sum()

last_indices = non_zeros[:, 1].view(gt_ind.size(0))

# assume batch = 1
eos_ind = int(last_indices+1)        
gt_ind = gt_ind[:,:eos_ind+1]
gt_ind[:, -1] = -2
B, N = gt_ind.shape
edges = edges[:,:N]
range_tensor = torch.arange(N, device=edges.device).unsqueeze(0).expand(B, N)
mask = range_tensor <= N

# gather gt, split into gt_x, gt_y, gt_z
gt_ind_copy = gt_ind.clone()
gt_ind_copy[gt_ind_copy == -1] = 0
gt_ind_copy[gt_ind_copy == -2] = 0
_, M, C = vertices.shape  # C = 3 (the last dimension)
expanded_indices = gt_ind_copy.unsqueeze(-1).expand(B, N, C)
gt = torch.gather(vertices, dim=1, index=expanded_indices)
gt[gt_ind==-1] = 2**quant_bit
gt[gt_ind==-2] = 2**quant_bit + 1

gt_x = gt[:,:,0].clamp(max = 2**quant_bit-1)
gt_y = gt[:,:,1].clamp(max = 2**quant_bit-1)
gt_z = gt[:,:,2]

gt_xyz = torch.cat((gt_x[:, None], gt_y[:, None], gt_z[:, None]), dim=1)

torch.save(gt_xyz, "gt_xyz.pt")