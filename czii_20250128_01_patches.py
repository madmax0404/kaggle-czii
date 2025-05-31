#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import polars as pl
import os
import gc
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import zarr
#import napari
from scipy.spatial import KDTree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.cuda.amp as amp  # ✅ Import automatic mixed precision (AMP)
import copy

gc.enable()

pd.options.display.max_columns = None
#pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)

#pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(10000)


# In[2]:


import sys
sys.path.append("/home/max1024/projects/MedicalNet")  # Adjust path as needed

# Test import
from model import generate_model
print("✅ MedicalNet imported successfully!")


# In[3]:


help(generate_model)


# In[4]:


from models import resnet


# In[5]:


path = '/media/max1024/Extreme SSD1/Kaggle/czii-cryo-et-object-identification/'


# In[6]:


train_data_experiment_folders_path = path + 'train/static/ExperimentRuns/'
train_data_experiment_folders_path


# In[7]:


test_data_experiment_folders_path = path + 'test/static/ExperimentRuns/'
test_data_experiment_folders_path


# In[8]:


train_data_experiments = os.listdir(train_data_experiment_folders_path)
train_data_experiments


# In[9]:


test_data_experiments = os.listdir(test_data_experiment_folders_path)
test_data_experiments


# In[10]:


data_dict = {}
for experiment in tqdm(train_data_experiments):
    image_types_dict = {}    
    image_types_dict['denoised'] = zarr.open(train_data_experiment_folders_path + f'{experiment}/VoxelSpacing10.000/denoised.zarr', mode='r')
    image_types_dict['iso'] = zarr.open(train_data_experiment_folders_path + f'{experiment}/VoxelSpacing10.000/isonetcorrected.zarr', mode='r')
    image_types_dict['dcon'] = zarr.open(train_data_experiment_folders_path + f'{experiment}/VoxelSpacing10.000/ctfdeconvolved.zarr', mode='r')
    image_types_dict['wbp'] = zarr.open(train_data_experiment_folders_path + f'{experiment}/VoxelSpacing10.000/wbp.zarr', mode='r')
    data_dict[experiment] = image_types_dict


# In[11]:


data_dict


# In[12]:


test_data_dict = {}
for experiment in tqdm(test_data_experiments):
    image_types_dict = {}    
    image_types_dict['denoised'] = zarr.open(test_data_experiment_folders_path + f'{experiment}/VoxelSpacing10.000/denoised.zarr', mode='r')
    test_data_dict[experiment] = image_types_dict


# In[13]:


test_data_dict


# In[14]:


train_label_experiment_folders_path = path + 'train/overlay/ExperimentRuns/'
train_label_experiment_folders_path


# In[15]:


train_label_experiments = os.listdir(train_label_experiment_folders_path)
train_label_experiments


# In[16]:


labels_dict = {}
for experiment in tqdm(train_label_experiments):
    particle_types_dict = {}
    
    with open(f'{train_label_experiment_folders_path}{experiment}/Picks/apo-ferritin.json') as f:
        loaded_json = json.loads(f.read())
    particle_types_dict['apo-ferritin'] = loaded_json

    with open(f'{train_label_experiment_folders_path}{experiment}/Picks/beta-amylase.json') as f:
        loaded_json = json.loads(f.read())
    particle_types_dict['beta-amylase'] = loaded_json

    with open(f'{train_label_experiment_folders_path}{experiment}/Picks/beta-galactosidase.json') as f:
        loaded_json = json.loads(f.read())
    particle_types_dict['beta-galactosidase'] = loaded_json

    with open(f'{train_label_experiment_folders_path}{experiment}/Picks/ribosome.json') as f:
        loaded_json = json.loads(f.read())
    particle_types_dict['ribosome'] = loaded_json

    with open(f'{train_label_experiment_folders_path}{experiment}/Picks/thyroglobulin.json') as f:
        loaded_json = json.loads(f.read())
    particle_types_dict['thyroglobulin'] = loaded_json

    with open(f'{train_label_experiment_folders_path}{experiment}/Picks/virus-like-particle.json') as f:
        loaded_json = json.loads(f.read())
    particle_types_dict['virus-like-particle'] = loaded_json

    labels_dict[experiment] = particle_types_dict


# In[17]:


labels_dict


# In[18]:


particle_radius = {
    'apo-ferritin': 60,
    'beta-amylase': 65,
    'beta-galactosidase': 90,
    'ribosome': 150,
    'thyroglobulin': 130,
    'virus-like-particle': 135,
}


# In[19]:


class_ids = {
    'apo-ferritin': 0,
    'beta-amylase': 1,
    'beta-galactosidase': 2,
    'ribosome': 3,
    'thyroglobulin': 4,
    'virus-like-particle': 5,
}


# In[20]:


weights_dict = {
    'apo-ferritin': 1,
    'beta-amylase': 0,
    'beta-galactosidase': 2,
    'ribosome': 1,
    'thyroglobulin': 2,
    'virus-like-particle': 1,
}


# In[21]:


weights_list = [1, 0, 2, 1, 2, 1]  # Order must match class indices
class_weights = torch.tensor(weights_list, dtype=torch.float32).to('cuda')


# In[22]:


experiment_list = []
particle_type_list = []
x_list = []
y_list = []
z_list = []
r_list = []
class_id_list = []
for experiment in tqdm(train_data_experiments):
    #print(experiment)
    #print(len(labels_dict[experiment]['apo-ferritin']['points']))
    #print(type(labels_dict[experiment]['apo-ferritin']['points']))
    #print(labels_dict[experiment]['apo-ferritin']['points'][0])

    for key in labels_dict[experiment].keys():
        #print(labels_dict[experiment][key])
        #print(labels_dict[experiment][key]['pickable_object_name'])
        for i in range(len(labels_dict[experiment][key]['points'])):
            experiment_list.append(labels_dict[experiment][key]['run_name'])
            particle_type_list.append(labels_dict[experiment][key]['pickable_object_name'])
            x_list.append(labels_dict[experiment][key]['points'][i]['location']['x']/10.012444537618887)
            y_list.append(labels_dict[experiment][key]['points'][i]['location']['y']/10.012444196428572)
            z_list.append(labels_dict[experiment][key]['points'][i]['location']['z']/10.012444196428572)
            r_list.append(particle_radius[key]/10)
            class_id_list.append(class_ids[key])


# In[23]:


labels_df = pd.DataFrame({'experiment':experiment_list, 'particle_type':particle_type_list, 'x':x_list, 'y':y_list, 'z':z_list, 'r':r_list, 'class_id':class_id_list})


# In[24]:


for k, v in class_ids.items():
    labels_df['class_id_' + str(v)] = (labels_df['class_id'] == v).astype(int)
labels_df = labels_df.drop(columns=['class_id'])
print(labels_df.shape)
labels_df


# In[25]:


labels_df.to_csv(path + 'labels.csv', index=False)


# In[26]:


labels_df['experiment'].unique()


# In[27]:


labels_df['experiment'].nunique()


# In[28]:


labels_df['particle_type'].unique()


# In[29]:


labels_df['particle_type'].nunique()


# In[30]:


train_experiments = list(data_dict.keys())[:5]
train_experiments


# In[31]:


valid_experiments = list(data_dict.keys())[5:]
valid_experiments


# In[32]:


data_dict[train_experiments[0]].keys()


# In[33]:


train_labels_df = labels_df[labels_df['experiment'].isin(train_experiments)].reset_index(drop=True).reset_index().rename(columns={'index':'id'})
train_labels_df


# In[34]:


valid_labels_df = labels_df[labels_df['experiment'].isin(valid_experiments)].reset_index(drop=True).reset_index().rename(columns={'index':'id'})
valid_labels_df


# In[35]:


data_dict['TS_5_4']['denoised'][0].shape


# In[36]:


plt.imshow(data_dict['TS_5_4']['denoised'][0][0])


# In[37]:


1 <= 2 <= 3


# In[38]:


valid_labels_df


# In[39]:


a = torch.ones(0, 300)
b = torch.ones(2, 300)
c = torch.ones(19, 300)
d = torch.ones(19, 300)
test_pad = pad_sequence([a, b, c, d])
test_pad.size()


# In[40]:


class YOLO3DDataset_Multi(Dataset):
    def __init__(self, dataset_labels_df, crop_size=(64, 128, 128), stride=(32, 64, 64), transform=None):
        self.dataset_labels_df = dataset_labels_df.copy()
        self.crop_size = crop_size
        self.stride = stride
        self.transform = transform
        
        self.patches = []
        self.patch_bboxes = []
        self.patch_experiment_ids = []
        self.patch_offsets = []  # ✅ Initialize offsets

        self.process_all_images()

    def process_all_images(self):
        """ Process all images and extract patches with corresponding bounding spheres. """
        
        i = 0        
        for experiment in tqdm(self.dataset_labels_df['experiment'].unique()):
            for v in data_dict[experiment].keys():
                image = data_dict[experiment][v]['0']

                # ✅ Ensure consistent normalization
                image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

                image_labels = self.dataset_labels_df[self.dataset_labels_df['experiment'] == experiment].drop(columns=['id']).to_numpy()

                patches, bboxes, offsets = self.create_crops_with_labels(image, image_labels)
                self.patches.extend(patches)
                self.patch_bboxes.extend(bboxes)
                self.patch_experiment_ids.extend([experiment] * len(patches))
                self.patch_offsets.extend(offsets)
                
                i += 1
                
        print(f"✅ Processed {i} images into {len(self.patches)} patches.")

    def create_crops_with_labels(self, image, labels):
        """ Extract overlapping 3D patches and assign bounding spheres to them efficiently. """
        d, h, w = image.shape
        crops, bboxes, offsets = [], [], []
    
        crop_size = self.crop_size
        stride = self.stride

        padding_value = -1 #np.mean(image)  # Use mean intensity for padding

        for z in range(0, d - stride[0], stride[0]):  
            for y in range(0, h - stride[1], stride[1]):  
                for x in range(0, w - stride[2], stride[2]):
                    
                    # ✅ Extract cropped patch
                    cropped_patch = image[z:z+crop_size[0], y:y+crop_size[1], x:x+crop_size[2]]

                    # ✅ Apply padding if necessary
                    if cropped_patch.shape != (64, 128, 128):
                        padded_image = np.full((64, 128, 128), padding_value, dtype=image.dtype)
                        padded_image[:cropped_patch.shape[0], :cropped_patch.shape[1], :cropped_patch.shape[2]] = cropped_patch
                        cropped_patch = padded_image.copy()
                        
                    cropped_bboxes = []
    
                    # ✅ Find bounding spheres within this patch (considering partial overlap)
                    for bbox in labels:
                        exp_id, particle_name, x_center, y_center, z_center, radius, *class_vector = bbox

                        # ✅ Compute the nearest stride-aligned grid position
                        z_grid = (z_center // stride[0]) * stride[0]
                        y_grid = (y_center // stride[1]) * stride[1]
                        x_grid = (x_center // stride[2]) * stride[2]
                        
                        # ✅ Assign bounding sphere only once to the closest aligned grid patch
                        if (z == z_grid and y == y_grid and x == x_grid):
                            z_new = z_center - z
                            y_new = y_center - y
                            x_new = x_center - x
                            cropped_bboxes.append([z_new, y_new, x_new, radius] + class_vector)
    
                    # ✅ Append patches and bounding spheres
                    crops.append(torch.tensor(cropped_patch, dtype=torch.float32))
                    bboxes.append(torch.tensor(cropped_bboxes, dtype=torch.float32) if cropped_bboxes else torch.empty((0, 4 + 6)))
                    offsets.append([z, y, x])  # ✅ Store offset for each patch
    
        # ✅ Stack patches into a single tensor for efficient storage
        crops_tensor = torch.stack(crops)
    
        # ✅ Pad bounding sphere tensors for batch consistency
        bboxes_tensor = pad_sequence(bboxes, batch_first=True, padding_value=-1)

        offsets_tensor = torch.tensor(offsets, dtype=torch.float32)
    
        return crops_tensor, bboxes_tensor, offsets_tensor

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = self.patches[idx].unsqueeze(0)  # Add channel dimension (1, D, H, W)
        y = self.patch_bboxes[idx]
        experiment_id = self.patch_experiment_ids[idx]
        crop_offset = self.patch_offsets[idx]  # ✅ New: Store crop offsets

        if self.transform:
            x, y = self.transform(x, y)

        return x, y, experiment_id, crop_offset


# In[41]:


# ✅ Example Usage
train_dataset = YOLO3DDataset_Multi(train_labels_df)


# In[42]:


valid_dataset = YOLO3DDataset_Multi(valid_labels_df)


# In[43]:


def find_max_anchors(dataset):
    max_anchors = 0

    for _, targets, _, _ in dataset:
        num_spheres = (targets[..., 0] != -1).sum().item()  # Count valid bounding spheres
        max_anchors = max(max_anchors, num_spheres)

    return max_anchors

# Example usage
max_anchors = find_max_anchors(train_dataset)
print(f"Optimal Number of Anchors: {max_anchors}")


# In[44]:


val_max_anchors = find_max_anchors(valid_dataset)
val_max_anchors


# In[45]:


class YOLO3D_MedicalNet(nn.Module):
    def __init__(self, num_classes=6, num_anchors=5, pretrain_path=None):
        super(YOLO3D_MedicalNet, self).__init__()

        # ✅ Load MedicalNet (Pretrained ResNet-18 3D)
        self.backbone = resnet.resnet18(
            sample_input_W=128,
            sample_input_H=128,
            sample_input_D=64,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=num_classes
        )

        # ✅ Load Pretrained Weights
        if pretrain_path:
            print(f"✅ Loading Pretrained MedicalNet Weights from: {pretrain_path}")
            pretrain = torch.load(pretrain_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
            self.backbone.load_state_dict(pretrain['state_dict'], strict=False)

        # ✅ Define YOLO3D Detection Head
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # ✅ Calculate Feature Size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 128, 128)  # [batch, channels, depth, height, width]
            dummy_output = self.backbone(dummy_input)  # Feature map after backbone
            feature_map_size = dummy_output.shape[2:]  # Extract (D, H, W)
            feature_depth = dummy_output.shape[1]  # Extract feature depth (channels)
        
        # ✅ 1×1×1 Convolution to Reduce Feature Depth
        self.conv1x1x1 = nn.Conv3d(feature_depth, 64, kernel_size=1, stride=1, padding=0)

        # ✅ Fully Connected Layer for YOLO3D
        self.fc = nn.Linear(64 * feature_map_size[0] * feature_map_size[1] * feature_map_size[2], self.num_anchors * (4 + self.num_classes))

    def forward(self, x):
        x = self.backbone(x)  # Extract Features using MedicalNet
        x = self.conv1x1x1(x)  # ✅ Reduce depth while keeping spatial structure
        x = torch.flatten(x, start_dim=1)  # Flatten for FC Layer
        x = self.fc(x)  # Predict Bounding Spheres

        # ✅ Dynamically compute number of anchors
        batch_size = x.shape[0]
        num_anchors = self.num_anchors  # Use predefined anchors

        # ✅ Reshape safely
        x = x.view(batch_size, num_anchors, 4 + self.num_classes)  

        # ✅ Apply softmax to class probabilities
        x_class = torch.softmax(x[..., 4:], dim=-1)  # ✅ Safe
        x = torch.cat([x[..., :4], x_class], dim=-1)  # ✅ No In-Place Ops

        return x


# In[46]:


model = YOLO3D_MedicalNet(num_anchors=max_anchors, pretrain_path="/home/max1024/models/MedicalNet/resnet_18_23dataset.pth")


# In[47]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        """Compute focal loss across all anchors."""

        # ✅ Adjust for Multiple Anchors: Shape (batch_size, num_anchors, 4 + num_classes)
        pred_cls = predictions[..., 4:].clone()  # Extract class predictions
        target_cls = targets[..., 4:].clone()  # Extract target class labels

        pred_cls = torch.clamp(pred_cls, min=-20, max=20)  # Prevent instability
        target_cls = torch.clamp(target_cls, min=0, max=1)

        ce_loss = F.binary_cross_entropy_with_logits(pred_cls, target_cls, reduction="none")

        # Compute focal weight
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        # 🔹 Reduce Focal Loss weight to allow learning
        focal_loss *= 0.5  

        # ✅ Average across anchors
        focal_loss = focal_loss.mean(dim=1)  # Average over `num_anchors`

        # ✅ Apply per-class weights
        if self.alpha is not None:
            focal_loss *= self.alpha.to(predictions.device)

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class SoftFBetaLoss(nn.Module):
    def __init__(self, beta=4, eps=1e-6, alpha=None):
        super(SoftFBetaLoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.alpha = alpha  # Class weights

    def forward(self, predictions, targets):
        """Compute F-beta loss across multiple anchors."""

        # ✅ Ensure Predictions and Targets have the same number of anchors
        min_size = min(predictions.shape[1], targets.shape[1])  # Align sizes
        predictions = predictions[:, :min_size, :].clone()
        targets = targets[:, :min_size, :].clone()

        target_cls = targets[..., 4:]
        pred_cls = predictions[..., 4:]

        pred_cls = torch.clamp(pred_cls, min=-10, max=10)  
        pred_cls = torch.sigmoid(pred_cls)  

        # Compute Precision and Recall
        tp = torch.sum(pred_cls * target_cls, dim=0)
        fp = torch.sum(pred_cls * (1 - target_cls), dim=0)
        fn = torch.sum((1 - pred_cls) * target_cls, dim=0)

        # ✅ Fix: Use `.sum()` instead of direct comparison
        tp_fp_sum = tp + fp
        tp_fn_sum = tp + fn

        precision = tp / (tp_fp_sum + self.eps)  
        recall = tp / (tp_fn_sum + self.eps)  

        fbeta = (1 + self.beta ** 2) * precision * recall / ((self.beta ** 2) * precision + recall + self.eps)

        if self.alpha is not None:
            fbeta *= self.alpha.to(pred_cls.device)

        return 1 - fbeta.mean()

class RegressionLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(RegressionLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, predictions, targets):
        pred_reg = predictions[..., :4]
        target_reg = targets[..., :4]

        # Normalize regression targets (avoid large-scale errors)
        max_range = 128.0  # Adjust based on dataset scale
        pred_reg /= max_range
        target_reg /= max_range

        # Compute Smooth L1 loss
        reg_loss = self.smooth_l1(pred_reg, target_reg)

        # ✅ Ensure correct reduction
        if reg_loss.dim() > 1:
            reg_loss = reg_loss.mean(dim=-1)  # Average across outputs (x, y, z, r)

        return reg_loss.mean() if self.smooth_l1.reduction == "mean" else reg_loss.sum()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, beta=4, lambda_focal=0.5, lambda_fbeta=0.5, lambda_reg=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.fbeta_loss = SoftFBetaLoss(beta=beta, alpha=alpha)
        self.regression_loss = RegressionLoss()

        self.lambda_focal = lambda_focal
        self.lambda_fbeta = lambda_fbeta
        self.lambda_reg = lambda_reg  # New term for regression loss

    def forward(self, predictions, targets):
        """Compute combined loss over multiple anchors."""

        focal = self.focal_loss(predictions, targets)
        fbeta = self.fbeta_loss(predictions, targets)
        reg_loss = self.regression_loss(predictions, targets)

        total_loss = self.lambda_focal * focal + self.lambda_fbeta * fbeta + self.lambda_reg * reg_loss

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"❌ NaN/Inf detected in loss function! Focal Loss: {focal}, F_beta Loss: {fbeta}, Regression Loss: {reg_loss}")

        return total_loss


# In[48]:


# ✅ Initialize Loss Function
loss_fn = CombinedLoss(alpha=class_weights, lambda_focal=0.5, lambda_fbeta=0.5, lambda_reg=0.5)
#loss_fn = CombinedLoss(lambda_focal=0.5, lambda_fbeta=0.5, lambda_reg=0.5)


# In[49]:


def collate_fn(batch):
    images, targets, experiment_ids, crop_offsets = zip(*batch)  
    images = torch.stack(images, dim=0)  # Stack images into a batch tensor

    max_boxes = max(target.shape[0] for target in targets)

    padded_targets = torch.full((len(targets), max_boxes, 10), -1.0)  # Use -1 for padding
    for i, target in enumerate(targets):
        padded_targets[i, :target.shape[0], :] = target  # Copy actual values

    # ✅ Fix: Avoid redundant tensor conversion
    if isinstance(crop_offsets[0], torch.Tensor):  
        crop_offsets_tensor = torch.stack(crop_offsets)  # Already tensors, just stack
    else:
        crop_offsets_tensor = torch.stack([offset.clone().detach() for offset in crop_offsets])

    return images, padded_targets, list(experiment_ids), crop_offsets_tensor


# In[50]:


train_loader = DataLoader(
    train_dataset, 
    batch_size=8,  # ✅ Increase batch size for efficiency
    shuffle=True,  
    num_workers=8,  # ✅ Start with 2, increase if stable
    pin_memory=True,  # ✅ Optimizes GPU memory transfer
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=8,  # ✅ Keep batch size consistent
    shuffle=False,  # ✅ No need for shuffling in validation
    num_workers=8,  # ✅ Start with a low value
    pin_memory=True,  
    collate_fn=collate_fn
)


# In[51]:


def compute_metrics(reference_points, reference_radius, candidate_points):
    num_reference_particles = len(reference_points)
    num_candidate_particles = len(candidate_points)

    if len(reference_points) == 0:
        return 0, num_candidate_particles, 0

    if len(candidate_points) == 0:
        return 0, 0, num_reference_particles

    ref_tree = KDTree(reference_points)
    candidate_tree = KDTree(candidate_points)
    raw_matches = candidate_tree.query_ball_tree(ref_tree, r=reference_radius)
    matches_within_threshold = []
    for match in raw_matches:
        matches_within_threshold.extend(match)
    # Prevent submitting multiple matches per particle.
    # This won't be be strictly correct in the (extremely rare) case where true particles
    # are very close to each other.
    matches_within_threshold = set(matches_within_threshold)
    tp = int(len(matches_within_threshold))
    fp = int(num_candidate_particles - tp)
    fn = int(num_reference_particles - tp)
    return tp, fp, fn


def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        distance_multiplier: float,
        beta: int) -> float:
    '''
    F_beta
      - a true positive occurs when
         - (a) the predicted location is within a threshold of the particle radius, and
         - (b) the correct `particle_type` is specified
      - raw results (TP, FP, FN) are aggregated across all experiments for each particle type
      - f_beta is calculated for each particle type
      - individual f_beta scores are weighted by particle type for final score
    '''

    particle_radius = {
        'apo-ferritin': 60,
        'beta-amylase': 65,
        'beta-galactosidase': 90,
        'ribosome': 150,
        'thyroglobulin': 130,
        'virus-like-particle': 135,
    }

    weights = {
        'apo-ferritin': 1,
        'beta-amylase': 0,
        'beta-galactosidase': 2,
        'ribosome': 1,
        'thyroglobulin': 2,
        'virus-like-particle': 1,
    }

    particle_radius = {k: v * distance_multiplier for k, v in particle_radius.items()}

    # Filter submission to only contain experiments found in the solution split
    split_experiments = set(solution['experiment'].unique())
    submission = submission.loc[submission['experiment'].isin(split_experiments)]

    # Only allow known particle types
    if not set(submission['particle_type'].unique()).issubset(set(weights.keys())):
        raise ParticipantVisibleError('Unrecognized `particle_type`.')

    assert solution.duplicated(subset=['experiment', 'x', 'y', 'z']).sum() == 0
    assert particle_radius.keys() == weights.keys()

    results = {}
    for particle_type in solution['particle_type'].unique():
        results[particle_type] = {
            'total_tp': 0,
            'total_fp': 0,
            'total_fn': 0,
        }

    for experiment in split_experiments:
        for particle_type in solution['particle_type'].unique():
            reference_radius = particle_radius[particle_type]
            select = (solution['experiment'] == experiment) & (solution['particle_type'] == particle_type)
            reference_points = solution.loc[select, ['x', 'y', 'z']].values

            select = (submission['experiment'] == experiment) & (submission['particle_type'] == particle_type)
            candidate_points = submission.loc[select, ['x', 'y', 'z']].values

            if len(reference_points) == 0:
                reference_points = np.array([])
                reference_radius = 1

            if len(candidate_points) == 0:
                candidate_points = np.array([])

            tp, fp, fn = compute_metrics(reference_points, reference_radius, candidate_points)

            results[particle_type]['total_tp'] += tp
            results[particle_type]['total_fp'] += fp
            results[particle_type]['total_fn'] += fn

    aggregate_fbeta = 0.0
    for particle_type, totals in results.items():
        tp = totals['total_tp']
        fp = totals['total_fp']
        fn = totals['total_fn']

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        fbeta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        aggregate_fbeta += fbeta * weights.get(particle_type, 1.0)

    if weights:
        aggregate_fbeta = aggregate_fbeta / sum(weights.values())
    else:
        aggregate_fbeta = aggregate_fbeta / len(results)
    return aggregate_fbeta


# In[52]:


def extract_predictions(predictions, experiment_ids, crop_offsets, threshold=0.2):
    pred_list = []
    particle_classes = ['apo-ferritin', 'beta-amylase', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']

    #print(f"🚨 Debug: Predictions shape: {predictions.shape}")  # Should be (batch, num_anchors, 10)
    #print(f"🚨 Debug: Experiment IDs: {experiment_ids}")

    for i in range(len(predictions)):  # Loop over batch
        pred_batch = predictions[i].detach().cpu()  # (num_anchors, 10)

        #print(f"🚨 Debug: Batch {i} predictions shape: {pred_batch.shape}")  # Check anchor count

        experiment_id = experiment_ids[i]
        offset_z, offset_y, offset_x = map(float, crop_offsets[i])  # ✅ Convert offsets to floats

        # ✅ Loop over anchors within this batch item
        for anchor_idx in range(pred_batch.shape[0]):  
            pred = pred_batch[anchor_idx]  # Extract anchor predictions

            # ✅ Convert relative coordinates to absolute
            x_pred = pred[2] + offset_x
            y_pred = pred[1] + offset_y
            z_pred = pred[0] + offset_z
            radius = pred[3]

            # ✅ Extract class probabilities and ensure validity
            class_probs = pred[4:].numpy()

            #print(class_probs.shape)

            if class_probs.size == 0:
                continue
            
            #print(f"🚨 Debug: Anchor {anchor_idx} class_probs shape: {class_probs.shape}")

            if len(class_probs) != len(particle_classes):
                print(f"⚠️ Warning: class_probs shape mismatch. Expected {len(particle_classes)}, got {len(class_probs)}.")
                print(predictions.shape)
                continue  # Skip this anchor

            class_id = np.argmax(class_probs)
            confidence = np.max(class_probs)

            # ✅ Ensure `class_id` is within range
            if 0 <= class_id < len(particle_classes) and confidence > threshold:
                particle_type = particle_classes[class_id]

                # ✅ Store as DataFrame
                pred_df = pd.DataFrame({
                    "experiment": [experiment_id],
                    "particle_type": [particle_type],
                    "x": [x_pred],
                    "y": [y_pred],
                    "z": [z_pred]
                })
                pred_list.append(pred_df)

    return pd.concat(pred_list, ignore_index=True) if pred_list else pd.DataFrame()


# In[53]:


def train_yolo3d_medicalnet(
    model, 
    train_dataloader, 
    val_dataloader,  
    loss_fn, 
    epochs=5, 
    lr=0.0001, 
    fine_tune=True, 
    beta=4,  
    distance_multiplier=1.0,
    backbone_lr_multiplier=1
):
    """
    Trains YOLO3D with multiple anchors instead of a single bounding sphere per image.
    Uses validation set and computes `F_beta` score for competition ranking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model)
    
    # Enable TensorFloat32 acceleration (for NVIDIA Ampere+ GPUs)
    torch.set_float32_matmul_precision('high')  

    # Mixed Precision Components
    scaler = torch.amp.GradScaler()  
    autocast_dtype = torch.float16  # Use FP16 for mixed precision
    
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": lr * backbone_lr_multiplier},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": lr}
    ]) 

    best_fbeta = 0  
    best_model_state = None  
    best_epoch = 0

    print(f"🚀 Training on {device} | Mixed Precision: Enabled")
    print(f"📝 Fine-tuning: {fine_tune} | Backbone LR: {lr * backbone_lr_multiplier} | New Layers LR: {lr}")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for batch_idx, (x, y, experiment_ids, crop_offsets) in enumerate(train_dataloader):
            if x is None:  
                continue
                
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)

            if torch.isnan(y).any() or torch.isinf(y).any():
                print(f"❌ NaN/Inf detected in labels at epoch {epoch}, batch {batch_idx}")
                print(f"🛠️ Labels Stats - Min: {y.min()}, Max: {y.max()}")
                continue  

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):  
                predictions = model(x)

                #extract_predictions(predictions, experiment_ids, crop_offsets)

                # **✅ Ensure loss handles multiple anchors**
                num_anchors = predictions.shape[1]

                # **✅ Ensure padding aligns properly**
                max_size = max(predictions.shape[1], y.shape[1])  
                padded_predictions = torch.full((predictions.shape[0], max_size, predictions.shape[2]), -1.0, device=predictions.device)
                padded_targets = torch.full((y.shape[0], max_size, y.shape[2]), -1.0, device=y.device)
                
                # Copy actual values
                padded_predictions[:, :predictions.shape[1], :] = predictions
                padded_targets[:, :y.shape[1], :] = y

                loss = loss_fn(padded_predictions, padded_targets)

            scaler.scale(loss).backward()  
            scaler.step(optimizer)  
            scaler.update()  

            total_train_loss += loss.item()

        # **🔹 Run Validation**
        model.eval()
        total_val_loss = 0.0
        val_preds, val_gts = [], []

        with torch.no_grad():
            for batch_idx, (x, y, experiment_ids, crop_offsets) in enumerate(val_dataloader):
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, dtype=torch.float32, non_blocking=True)

                with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):  
                    predictions = model(x)

                    # **✅ Ensure loss accounts for multiple anchors**
                    num_anchors = predictions.shape[1]
                    max_size = max(predictions.shape[1], y.shape[1])  

                    # **✅ Padding to ensure correct loss computation**
                    padded_predictions = torch.full((predictions.shape[0], max_size, predictions.shape[2]), -1.0, device=predictions.device)
                    padded_targets = torch.full((y.shape[0], max_size, y.shape[2]), -1.0, device=y.device)
                    
                    # Copy actual values
                    padded_predictions[:, :predictions.shape[1], :] = predictions
                    padded_targets[:, :y.shape[1], :] = y

                    val_loss = loss_fn(padded_predictions, padded_targets)
                    total_val_loss += val_loss.item()

                    # **✅ Extract Predictions Per Anchor**
                    val_preds.append(extract_predictions(predictions, experiment_ids, crop_offsets))  

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)

        # **🔹 Compute F_beta score**
        submission_df = pd.concat(val_preds, ignore_index=True).drop_duplicates(
            subset=['experiment', 'particle_type', 'x', 'y', 'z']
        ).sort_values(by=['experiment', 'particle_type']).reset_index(drop=True).reset_index().rename(
            columns={'index': 'id'}
        ) if val_preds else pd.DataFrame()
        
        val_fbeta = score(valid_labels_df[['id', 'experiment', 'particle_type', 'x', 'y', 'z']], submission_df, "id", 0.1, 4) if not submission_df.empty else 0.0

        # **🔹 Track Best Model Based on Validation F_beta Score**
        if val_fbeta > best_fbeta:
            best_fbeta = val_fbeta
            best_model_state = copy.deepcopy(model)  
            best_epoch = epoch
            torch.save(best_model_state, f"best_yolo3d_model_{best_fbeta:.4f}.pth")
            print(f"💾 Best Model Saved with F_beta: {best_fbeta:.4f}")

        print(f"✅ Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | 📊 Val F_beta: {val_fbeta:.4f} | Best Epoch: {best_epoch+1} | Best Val F_beta: {best_fbeta:.4f}")

    print("🎯 Training Complete!")


# In[ ]:


# ✅ Run training
train_yolo3d_medicalnet(model, train_loader, valid_loader, loss_fn, epochs=10, lr=0.01, fine_tune=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # References
# 
# 1. https://www.kaggle.com/code/davidlist/experiment-ts-6-4-visualization
# 2. https://www.kaggle.com/code/nk35jk/3d-visualization-of-particles

# In[ ]:




