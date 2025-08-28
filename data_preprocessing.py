import os
import math
import random
from PIL import Image
import blobfile as bf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, ToPILImage, Compose
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage.measure import label, regionprops
import re 

from params import *
args = parse_arguments()

def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])
    return 0


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir), key=extract_number):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def calculate_mask_properties(image_array, size_threshold=args.st):
    mask_positions = image_array == 255
    labeled_array = label(mask_positions)

    properties = []
    for region in regionprops(labeled_array):
        if region.area >= size_threshold: 
            region_area = region.area  
            region_center = region.centroid  
            properties.append((region_area, region_center))

    return properties

def max_count(directory_path):
    counts = []
    for filename in sorted(os.listdir(directory_path), key=lambda x: int(re.findall(r'\d+', x)[0]) if x.startswith('mask') and (x.endswith('.png') or x.endswith('.jpg')) else float('inf')):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(directory_path, filename)
            image = cv2.resize(cv2.imread(image_path), (args.image_size, args.image_size))[:,:,1]
            image_array = np.array(image)
            mask_properties = calculate_mask_properties(image_array)
            counts.append(len(mask_properties))
    return max(counts)

def process_directory(directory_path, size_threshold=args.st):
    results = {}
    images = []
    size_label = []
    location_label = []
    count_label = []
    max_masks = max_count(directory_path)

    for filename in sorted(os.listdir(directory_path), key=lambda x: int(re.findall(r'\d+', x)[0]) if x.startswith('mask') and (x.endswith('.png') or x.endswith('.jpg')) else float('inf')):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(directory_path, filename)
            image = cv2.resize(cv2.imread(image_path), (args.image_size, args.image_size))[:,:,1]
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            image_array = np.array(image)

            mask_properties = calculate_mask_properties(image_array, size_threshold)

            for i, prop in enumerate(mask_properties):
                size_label.append(prop[0])
                location_label.append([prop[1][1],prop[1][0]])  

            count_label.append(len(mask_properties))

            image = np.expand_dims(image, axis=-1)
            images.append(image)

            results[filename] = mask_properties

    return images, size_label, location_label, count_label

def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def max_normalize(data):
    max_val = max(data)
    normalized_data = [x / max_val for x in data]
    return normalized_data

def decimal_scaling_normalize(data):
    max_abs_val = max(abs(x) for x in data)
    scale = len(str(int(max_abs_val))) 
    scale_factor = 10 ** scale
    normalized_data = [x / scale_factor for x in data]
    return normalized_data

class NormMaskDataset(Dataset):
    def __init__(self, images, sizes, locations, transform=None):
        self.images = images
        self.sizes = sizes
        self.locations = locations
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        size = self.sizes[idx]
        location = self.locations[idx]

        image = ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor([size] + list(location), dtype=torch.float32)

        return image, label