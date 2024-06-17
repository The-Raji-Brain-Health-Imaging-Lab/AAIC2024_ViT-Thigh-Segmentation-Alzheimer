# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
from PIL import Image
import cv2

import matplotlib.pyplot as plt

cases_dir_bmp = '03_mask_cases_dir'
bmp_files = os.listdir(os.path.join(cases_dir_bmp, os.listdir(cases_dir_bmp)[0]))

out_dir="05_masks_dir_final"
os.makedirs(out_dir, exist_ok=True)
img_dir= "00_FS_img_dir"


for img_file in sorted(os.listdir(img_dir)):
    if img_file.endswith('.img'):
        case_name = img_file.split('-')[1]
        bmp_dir = os.path.join(cases_dir_bmp, f'AT-{case_name}')
        img_path = os.path.join(img_dir, img_file)
        img_img = nib.load(img_path)
        img_data = img_img.get_fdata()
        img_data = np.squeeze(img_data) # remove the extra dimension
        img_header = img_img.header
        width, height, num_slices = img_data.shape
        
        bmp_files = sorted([f for f in os.listdir(bmp_dir) if f.endswith('.bmp')])

        # Read the BMP files and stack them into a 3D numpy array
        stacked_slices = np.stack([np.array(cv2.imread(os.path.join(bmp_dir, bmp_file), 0)).T for bmp_file in bmp_files])
        swapped_stacked_slices = np.transpose(stacked_slices, (1,2,0))
        swapped_stacked_slices_rotated = np.flip(swapped_stacked_slices, axis=1)

        print(img_file, "==>", swapped_stacked_slices_rotated.shape, "||", img_data.shape)
        
        ## encode mask
        swapped_stacked_slices_rotated[swapped_stacked_slices_rotated == 255] = 1 ##  bone
        swapped_stacked_slices_rotated[swapped_stacked_slices_rotated == 150] = 2 ##  Intermuscular fat
        swapped_stacked_slices_rotated[swapped_stacked_slices_rotated == 25] =  3 ## intramuscular  fat
        swapped_stacked_slices_rotated[swapped_stacked_slices_rotated == 226] = 4 ## SAT
        swapped_stacked_slices_rotated[swapped_stacked_slices_rotated == 76] =  5 ## Muscle
        swapped_stacked_slices_rotated[swapped_stacked_slices_rotated == 105] = 6 ## Gluteus
        # Write a new .img file using the stacked slices and the original header
        mask_volume = nib.Nifti1Image(swapped_stacked_slices_rotated, img_img.affine, img_img.header)
        nib.save(mask_volume, os.path.join(out_dir, f'{img_file.replace(".img", ".nii")}'))