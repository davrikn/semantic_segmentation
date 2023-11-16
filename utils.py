from os import listdir, makedirs
from os.path import isdir
from shutil import copyfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from random import random, randint
import torch

mappings = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 31: 16, 32: 17, 33: 18}

train_mapping_to_rgb = {0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70], 3: [102, 102, 156], 4: [190, 153, 153], 5: [153, 153, 153],
                        6: [250, 170, 30], 7: [220, 220, 0], 8: [107, 142, 35], 9: [152, 251, 152], 10: [70, 130, 180], 11: [220, 20, 60],
                        12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70], 15: [0, 60, 100], 16: [0, 80, 100], 17: [0, 0, 230], 19: [119, 11, 32]}

val_imgroot = './data2/val/img'
val_labelroot = './data2/val/labels'
imgroot = './data2/train/img'
labelroot = './data2/train/labels'

def project_out(out, colored=False):
    projected = np.zeros(shape=(out.shape[0], out.shape[2], out.shape[3]))
    for i in range(len(out)):
        for w in range(out.shape[2]):
            for h in range(out.shape[3]):
                max = -999
                _ = -1
                for c in range(out.shape[1]):
                    if out[i][c][w][h] > max:
                        max = out[i][c][w][h]
                        _ = c
                if colored:
                    _ = train_mapping_to_rgb[_]
                projected[i][w][h] = _
    return projected

def load_images(image_names, validation=False, device="cuda", filter=None):
    if filter is None:
        return np.array([np.array(Image.open(f"{imgroot if not validation else val_imgroot}/{x}")) for x in image_names])
    else:
        return np.array([np.array(Image.open(f"{imgroot if not validation else val_imgroot}/{x}").filter(filter)) for x in image_names])
        

def load_labels(image_names, validation=False):
    return np.array([np.array(Image.open(f"{labelroot if not validation else val_labelroot}/{x}")) for x in image_names])

def invert_channels(images: np.ndarray):
    #new = np.zeros((image.shape[0], image.shape[2], image.shape[3], image.shape[1]))
    #for i in range(image.shape[0]):
    #    new[i] = np.moveaxis(image[i], 0, -1)
    #return new
    return np.moveaxis(images, -1, 1)

def to_tensor(arr: np.ndarray, device="cuda", dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype).to(device)


def augment_pairs(images: np.ndarray, labels: np.ndarray, flip_likelihood=0.5):
    for i in range(len(images)):
        if random() < flip_likelihood:
            if random() > 0.5:
                images[i] = np.flip(images[i], axis=0)
                labels[i] = np.flip(labels[i], axis=0)
            else:
                images[i] = np.flip(images[i], axis=1)
                labels[i] = np.flip(labels[i], axis=1)
    
    cropsizes = [randint(150, 255), randint(350, 511)] 
    cropoffset = [randint(0, 256-cropsizes[0]), randint(0, 512-cropsizes[1])]

    h_start = cropoffset[0]
    h_end = cropoffset[0] + cropsizes[0]

    w_start = cropoffset[1]
    w_end = cropoffset[1] + cropsizes[1]


    images = images[:, h_start:h_end, w_start:w_end]
    labels = labels[:, h_start:h_end, w_start:w_end]   

    if (images.shape[:3] != labels.shape):
        raise Exception("Image-label mismatch")

    return images, labels


def verify_files(img, label):
    img_city, img_1, img_2 = img.split("_")[0:3]
    label_city, label_1, label_2 = label.split("_")[0:3]
    if img_city != label_city or img_1 != label_1 or img_2 != label_2:
        print("Img mismatch")
        exit(1)


def id_to_rgb(input: np.ndarray):
    out = np.zeros((input.shape[0], input.shape[1], 3))
    for i in range(len(input)):
        for j in range(len(input[i])):
            _class = input[i][j]
            try:
                rgb = train_mapping_to_rgb[_class]
            except:
                rgb = [0,0,0]
            out[i][j] = rgb
    return out


def encode_labels(label: Image):
    ar = np.array(label)
    ignore = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    for i in ignore:
        ar[ar == i] = 255

    for (k, v) in mappings.items():
        ar[ar == k] = v
    return ar

def preprocess(mode: str = "train"):
    if mode == "train":
        subfolder = "train"
    elif mode == "val":
        subfolder = "val"


    if not isdir(f"data2/{subfolder}/img") or not isdir(f"data2/{subfolder}/labels"):
        makedirs(f"./data2/{subfolder}/img")
        makedirs(f"./data2/{subfolder}/labels")


    labelroot = f'data/gtFine/{subfolder}'
    imgroot = f'data/leftImg8bit/{subfolder}'
    cities = listdir(imgroot)
    cities.sort()

    for city in cities:
        images = listdir(f'{imgroot}/{city}')
        images.sort()
        labels = listdir(f'{labelroot}/{city}')
        labels.sort()

        bar = tqdm(range(len(images)))
        bar.set_description(city)
        for i in bar:
            img = images[i]

            img_prefix = f"{img[0:len(img)-16]}.png"

            loadedimg = Image.open(f"{imgroot}/{city}/{img}")
            loadedimg = loadedimg.resize((512, 256), resample=Image.Resampling.NEAREST)
            loadedimg.save(f"data2/{subfolder}/img/{img_prefix}")

            if mode == "train" or mode == "val":
                label = labels[i*4+2]
                verify_files(img, label)
                label_prefix = f"{label[0:len(label)-20]}.png"

                processed_labels = encode_labels(Image.open(f"{labelroot}/{city}/{label}"))
                processed_labels = Image.fromarray(processed_labels)
                processed_labels = processed_labels.resize((512, 256), resample=Image.Resampling.NEAREST)
                processed_labels.save(f"data2/{subfolder}/labels/{label_prefix}")


if __name__ == "__main__":
    preprocess("val")