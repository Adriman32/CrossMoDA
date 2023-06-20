import time
import os
from PIL import Image
import nibabel as nib
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def load_data(filepath, loader = 'unet', resize = 'default', task = 'segmentation', verbose = 0):

    load_start = time.time()
    img_list = []
    label_list = []

    if loader == 'unet':
        img_list, label_list =  unet_loader(filepath, resize, task, verbose)
    else:
        raise Exception("Invalid data loader. Please select a valid loader: unet")
        
    if verbose:
        print(f'Data loaded in {((time.time()-load_start)/60):.2f} minutes')

    return img_list, label_list

def unet_loader(filepath, resize, task='segmentation', verbose=0):
    if verbose:
        print('Initiating U-Net Loader')
    img_list = []
    label_list = []

    if resize == 'default':
        resize = (572, 572)
    mask_crop_size = (resize[0] // 16, resize[1] // 16)

    entries = sorted(os.scandir(filepath), key=lambda e: e.name)

    for entry_index, entry in enumerate(entries):
        if entry.is_file():
            img_filename = entry.name
            label = img_filename.split('_')[0]
            if task == 'classification':
                if img_filename.endswith(('.jpg', '.jpeg', '.png')):
                    img = Image.open(entry.path).resize(resize)
                    img_list.append(img)
                    label_list.append(label)
                elif img_filename.endswith('.nii.gz'):
                    img_data = nib.load(entry.path).get_fdata()
                    num_slices = img_data.shape[-1]
                    for slice_idx in range(num_slices):
                        slice_data = img_data[..., slice_idx]
                        img = Image.fromarray(slice_data.astype(np.uint8))
                        if resize is not None and img.size != resize:
                            img = img.resize(resize)
                        img_list.append(np.asarray(img))
                        label_list.append(label)
            elif task == 'segmentation':
                if entry_index % 2 == 0:
                    if img_filename.endswith(('.jpg', '.jpeg', '.png')):
                        mask_entry = entries[entry_index + 1]
                        mask_filename = mask_entry.name
                        img = Image.open(mask_entry.path).resize(resize) # Swapped because this dataset has mask then img
                        mask = Image.open(entry.path)
                        mask_width, mask_height = mask.size
                        crop_left = (mask_width - mask_crop_size[0]) // 2
                        crop_top = (mask_height - mask_crop_size[1]) // 2
                        mask = mask.crop((crop_left, crop_top, crop_left + mask_crop_size[0], crop_top + mask_crop_size[1]))
                        img_list.append(np.asarray(img))
                        label_list.append(np.asarray(mask))
                    elif img_filename.endswith('.nii.gz'):
                        mask_entry = entries[entry_index + 1]
                        mask_filename = mask_entry.name
                        mask_data = nib.load(entry.path).get_fdata()        # Swapped because this dataset has mask then img
                        img_data = nib.load(mask_entry.path).get_fdata()
                        num_slices = img_data.shape[-1]
                        for slice_idx in range(num_slices):
                            img_slice = img_data[..., slice_idx]
                            mask_slice = mask_data[..., slice_idx]
                            img = Image.fromarray(img_slice.astype(np.uint8))
                            mask = Image.fromarray(mask_slice.astype(np.uint8))
                            if resize is not None and img.size != resize:
                                img = img.resize(resize)
                            mask_width, mask_height = mask.size
                            crop_left = (mask_width - mask_crop_size[0]) // 2
                            crop_top = (mask_height - mask_crop_size[1]) // 2
                            mask = mask.crop((crop_left, crop_top, crop_left + mask_crop_size[0], crop_top + mask_crop_size[1]))
                            img_list.append(np.asarray(img))
                            label_list.append(np.asarray(mask))
            if verbose == 2:
                print(f'Loaded image: {label}')

    return img_list, label_list

def display_images(img_list, label_list=None, num_images=None, random_sampling=False):
    num_images_total = len(img_list)

    if random_sampling:
        indices = random.sample(range(num_images_total), num_images_total)
        img_list = [img_list[i] for i in indices]
        if label_list is not None:
            label_list = [label_list[i] for i in indices]

    if num_images is not None:
        img_list = img_list[:num_images]
        if label_list is not None:
            label_list = label_list[:num_images]

    num_images = len(img_list)

    if label_list is not None:
        num_cols = 2 if any(not isinstance(label, str) for label in label_list) else 1
        num_rows = num_images

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))

        for i, (img, label) in enumerate(zip(img_list, label_list)):
            if isinstance(label, str):  # Text label
                axs[i].imshow(img)
                axs[i].axis('off')
                axs[i].set_title(label)
            else:  # Image label
                if num_cols == 2:
                    axs[i, 0].imshow(img)
                    axs[i, 0].axis('off')
                    axs[i, 0].set_title('Original')

                    axs[i, 1].imshow(label)
                    axs[i, 1].axis('off')
                    axs[i, 1].set_title('Mask')
                else:
                    axs[i].imshow(img)
                    axs[i].axis('off')
                    axs[i].set_title('Original')

        plt.tight_layout()
        plt.show()
    else:
        num_cols = 1
        num_rows = num_images
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5, 5 * num_rows))

        for i, img in enumerate(img_list):
            axs[i].imshow(img)
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()
