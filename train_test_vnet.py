# -*- coding: utf-8 -*-
import sys
import numpy as np
import torch

print('Python:', sys.version)
print('Numpy:', np.__version__)
print('torch', torch.__version__)

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))


import os
import glob
import time
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import einops
import torch.nn as nn
import monai
import nibabel as nib
from monai.utils import first, set_determinism
from monai.transforms import (AsDiscrete, AsDiscreted, EnsureChannelFirstd, Compose, CropForegroundd, LoadImaged, Orientationd, RandFlipd,RandShiftIntensityd,Activations,
                              RandCropByPosNegLabeld, SaveImaged, ScaleIntensityRanged, Spacingd, Invertd, DataStatsd, Resized, RandRotate90d, ToTensord, RandAffined, RandGaussianSmoothd,RandGaussianNoised,RandZoomd,RandHistogramShiftd,RandAdjustContrastd, SpatialPadd)
from monai.networks.nets import UNet, VNet, AttentionUnet, BasicUNetPlusPlus, UNETR, SwinUNETR
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss, TverskyLoss, GeneralizedDiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU
from monai.utils import set_determinism
from monai.data import DataLoader, decollate_batch, CacheDataset, Dataset
from monai.handlers.utils import from_engine
from torchvision.utils import save_image
from monai.config import print_config
from utils import *
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

pd.set_option('max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

set_determinism(seed=1142)

data_dir= "thigh_3folds/fold1/"
train_data_dir= os.path.join(data_dir, "train/")
test_data_dir= os.path.join(data_dir, "test/")
## -------------------------------------------------------------------
train_img_dir= os.path.join(train_data_dir, "images/")
train_mask_dir=  os.path.join(train_data_dir, "masks/")

test_img_dir= os.path.join(test_data_dir, "images/")
test_mask_dir=  os.path.join(test_data_dir, "masks/")
## ------------------------------------------------------------------
train_img_paths = sorted(glob.glob(train_img_dir + "*.nii"))
train_mask_paths = sorted(glob.glob(train_mask_dir + "*.nii"))

test_images = sorted(glob.glob(test_img_dir + "*.nii"))
test_masks = sorted(glob.glob(test_mask_dir + "*.nii"))

## ------------------------------------------------------------------
'''--------------------split data into train and validation and test sets--------------------'''

train_images, valid_images, train_masks, valid_masks = train_test_split(train_img_paths, train_mask_paths, test_size=0.2)

print(f'#training samples: {len(train_images)}, #validation samples: {len(valid_images)}, #test samples: {len(test_images)}')


train_files = [{"image": image_name, "label": label_name}
               for image_name, label_name in zip(train_images, train_masks)]

val_files = [{"image": image_name, "label": label_name}
               for image_name, label_name in zip(valid_images, valid_masks)]

test_files = [{"image": image_name, "label": label_name}
               for image_name, label_name in zip(test_images, test_masks)]

patch_size= (64, 64, 64)    ## roi_size
# patch_size=(96, 96, 64)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        RandCropByPosNegLabeld(keys=["image", "label"],label_key="label", spatial_size = patch_size,pos=1, neg=1,
            num_samples=6,image_key="image",image_threshold=0,),   
        ToTensord(keys=["image", "label"]),
    ])       

val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ToTensord(keys=["image", "label"]),])


num_gpus = torch.cuda.device_count()  
batch_size = 1 * num_gpus

# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

# val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
# val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

#################################################
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=True)

#################################################

print(f"num_gpus: {num_gpus}")
print(f"batch_size: {batch_size}")


num_classes= 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VNet(spatial_dims=3,
             in_channels=1,
             out_channels=num_classes)

num_params = torch.nn.utils.parameters_to_vector(model.parameters()).numel()
num_params_millions = num_params / 1_000_000

formatted_num_params = "{:,.0f}".format(num_params_millions * 1_000_000)
print('Total number of parameters (millions):', formatted_num_params)

total_layers = sum(p.requires_grad for p in model.parameters())
print('Total number of layers:', total_layers)

model = nn.DataParallel(model)
model = model.to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=True, reduction="mean")
iou_metric= MeanIoU(include_background=True, reduction="mean", get_not_nans=False)


model_name= model.module.__class__.__name__
model_name= str(model_name)
model_path = str(model_name)+"_fold1"


max_epochs = 1000
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
post_label = Compose([AsDiscrete(to_onehot=num_classes)])

max_iterations = max_epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iterations)

start_time = time.time()  # start time for total training

for epoch in range(max_epochs):
    print("--" * 50)
    model.train()
    epoch_loss = 0
    step = 0

    epoch_start_time = time.time()  # start time for the current epoch

    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step the scheduler
        epoch_loss += loss.item()

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    epoch_end_time = time.time()  # end time for the current epoch
    epoch_duration = epoch_end_time - epoch_start_time  # duration for the current epoch

    print(f"epoch {epoch + 1}/{max_epochs} - average loss: {epoch_loss:.4f} - time: {epoch_duration:.2f} seconds (Model: {model.module.__class__.__name__})")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = patch_size
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_path, "best_metric_model.pth"))
            print(f"Current mean dice: {metric:.4f} | best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

end_time = time.time()  # end time for total training
total_duration = end_time - start_time  # total duration
print(f"train completed, best_metric: {best_metric:.4f} " f"at iteration: {best_metric_epoch}")


print(f"Total training time: {total_duration:.2f} seconds")

# Calculate hours, minutes, and seconds
hours = int(total_duration / 3600)
minutes = int((total_duration % 3600) / 60)

# Print the result
print(f"Total training time: {hours} hours, {minutes} minutes")

plt.figure(figsize=(14,6))

# Plotting Loss over Epochs
plt.plot(range(max_epochs), epoch_loss_values, color='red', label='Loss')

# Plotting Dice Coefficient over Epochs
plt.plot(range(0, max_epochs, val_interval), metric_values, color='green', label='Dice Coefficient')

# Find index of the lowest loss and highest Dice coefficient
lowest_loss_index = epoch_loss_values.index(min(epoch_loss_values))
highest_dice_index = metric_values.index(max(metric_values))

# Annotate lowest loss
plt.scatter(lowest_loss_index, epoch_loss_values[lowest_loss_index], 
            color='red', label=f"Lowest Loss: {min(epoch_loss_values):.4f} at epoch: {lowest_loss_index}")

# Annotate highest Dice coefficient
plt.scatter(highest_dice_index*val_interval, metric_values[highest_dice_index], 
            color='yellow', label=f"Highest Dice: {max(metric_values):.4f} at epoch: {highest_dice_index*val_interval}")

plt.title('Loss and Dice Coefficient over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.legend()

plt.show()


out_dir= f"{str(model_path)}/{str(model_path)}_predicted_dir/"
print(out_dir)
os.makedirs(out_dir, exist_ok=True)


test_org_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])

test_org_ds = Dataset(data=test_files, transform=test_org_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=1)

post_transforms = Compose([
    Invertd(
        keys="pred",
        transform=test_org_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
        device="cpu",
    ),
    AsDiscreted(keys="pred", argmax=True),
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=out_dir, output_ext=".nii", output_postfix="seg", separate_folder=False, resample=False),
])


total_inference_time = 0

model.load_state_dict(torch.load(os.path.join(model_path, "best_metric_model.pth")))
model.eval()

with torch.no_grad():
  for test_data in test_org_loader:
    test_inputs = test_data["image"].to(device)
    roi_size = roi_size
    sw_batch_size = 4
    
    start_time = time.time()
    test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
    inference_time = time.time() - start_time
    
      
    test_data = [post_transforms(i) for i in decollate_batch(test_data)]
    test_outputs, test_labels = from_engine(["pred", "label"])(test_data)
    
    total_inference_time += inference_time
    inference_time_str = format_inference_time(inference_time)
    print(f"Inference time: {inference_time_str}")
    print("===================================================")

formatted_total_inference_time_str = format_inference_time(total_inference_time)
print(f"Total Inference time: {formatted_total_inference_time_str}")

pred_files= glob.glob(out_dir+"*.nii")
pred_files= [fname.replace('_seg', '') for fname in pred_files]
test_pred_files= [os.path.basename(fname) for fname in pred_files]
test_truth_files = sorted(test_masks)


prediction_results= []
for test_truth, test_pred in zip(sorted(test_truth_files), sorted(test_pred_files)):
#     print(test_truth, test_pred)
    y_true_path= os.path.join(data_dir, 'masks/',test_truth)
    y_true_path= test_truth
#     print("****", test_fname)  
    y_true = nib.load(y_true_path).get_fdata()
    
    y_pred_path=os.path.join(out_dir,test_pred.replace(".nii", "_seg.nii"))
#     print("****", y_pred_path)
    y_pred= nib.load(y_pred_path).get_fdata()
    print(y_true.shape, y_pred.shape)
    
    jaccard_score = jaccard_index(y_true, y_pred)
    dice_score = dice_coef(y_true, y_pred)
    print("image: {} -- jaccard index: {} -- dice score: {}".format(test_truth, jaccard_score, dice_score))
    prediction_results.append([test_truth, jaccard_score, dice_score])
    print("======"* 15)
    
    
df= pd.DataFrame(prediction_results, columns=['file', 'jaccard_score', 'dice_score'])
ds_col= df['dice_score'].values
js_col= df['jaccard_score'].values
print("============" * 12)
print("Dice score Mean: {}, +- {} std, Median: {}".format(np.mean(ds_col), np.std(ds_col), np.median(ds_col)))
print("Jaccard score Mean: {}, +- {} std, Median: {}".format(np.mean(js_col), np.std(js_col), np.median(js_col)))
print("============" * 12)
save_path = os.path.join(model_path,"dice_jaccard_results.csv")
df.to_csv(save_path, sep=',')    



ds_col= df['dice_score'].values
js_col= df['jaccard_score'].values
print("============" * 12)
print("Dice score Mean: {}, +- {} std, Median: {}".format(np.mean(ds_col), np.std(ds_col), np.median(ds_col)))
print("Jaccard score Mean: {}, +- {} std, Median: {}".format(np.mean(js_col), np.std(js_col), np.median(js_col)))
print("=========" * 12)


prediction_results= []
class_names = {0: "background", 
               1: "bone", 
               2: "inter_fat",
               3: "intra_fat", 
               4: "sat",
               5: "muscle", 
               6: "gluteus",              
              }
for test_truth, test_pred in zip(sorted(test_truth_files), sorted(test_pred_files)):
    y_true = nib.load(test_truth).get_fdata()
    y_pred= nib.load(os.path.join(out_dir,test_pred.replace(".nii", "_seg.nii"))).get_fdata()

    for class_id, class_name in class_names.items():
        jaccard_score = jaccard_index_multiclass(y_true, y_pred, class_id)
        dice_score = dice_coef_multiclass(y_true, y_pred, class_id)
        print("image: {} class: {} -- jaccard index: {} -- dice score: {}".format(test_truth, class_name, jaccard_score, dice_score))
        prediction_results.append([test_truth, class_name, jaccard_score, dice_score])

df= pd.DataFrame(prediction_results, columns=['file', 'class', 'jaccard_score', 'dice_score'])
df.to_csv(os.path.join(model_path,"dice_jaccard_class_results.csv"))

class_results = {}


for class_id, class_name in class_names.items():
    # Filter the dataframe by class
    class_df = df[df['class'] == class_name]

    dice_scores = class_df['dice_score'].values
    jaccard_scores = class_df['jaccard_score'].values

    # Calculate mean, std, min, max and median for dice scores and jaccard scores
    dice_stats = {
        "mean": np.mean(dice_scores),
        "std": np.std(dice_scores),
        "min": np.min(dice_scores),
        "max": np.max(dice_scores),
        "median": np.median(dice_scores),
    }
    jaccard_stats = {
        "mean": np.mean(jaccard_scores),
        "std": np.std(jaccard_scores),
        "min": np.min(jaccard_scores),
        "max": np.max(jaccard_scores),
        "median": np.median(jaccard_scores),
    }

    class_results[class_name] = {
        "dice": dice_stats,
        "jaccard": jaccard_stats,
    }

# Print the results for each class
for class_name, results in class_results.items():
    print("Class: {}".format(class_name))
    print("Dice Score - Mean: {:.4f}, Std: {:.4f}, Min: {:.4f}, Max: {:.4f}, Median: {:.4f}".format(
        results["dice"]["mean"],
        results["dice"]["std"],
        results["dice"]["min"],
        results["dice"]["max"],
        results["dice"]["median"],
    ))
    print("Jaccard Score - Mean: {:.4f}, Std: {:.4f}, Min: {:.4f}, Max: {:.4f}, Median: {:.4f}".format(
        results["jaccard"]["mean"],
        results["jaccard"]["std"],
        results["jaccard"]["min"],
        results["jaccard"]["max"],
        results["jaccard"]["median"],
    ))
    print("-------------------------")







