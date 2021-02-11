
import h5py
from tqdm import tqdm
import json
import os
import torch
import clip
import sys
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

assert len(sys.argv) > 2, "Must specify json path and then path with images"

def create_and_save(json_path, image_root):

    embed_size = 512
    print('Reading json...')
    json_data = json.load(open(json_path, 'r'))

    train_embeds = np.zeros((len(json_data['unique_img_train']), embed_size), dtype=np.float32)

    for idx, path in tqdm(enumerate(json_data['unique_img_train'])):
        filename = os.path.join(image_root, path)
        image = preprocess(Image.open(filename)).unsqueeze(0).to(device)

        with torch.no_grad(): 
            image_features = model.encode_image(image)
            train_embeds[idx] = image_features.cpu()

    val_embeds = np.zeros((len(json_data['unique_img_val']), embed_size), dtype=np.float32)

    for idx, path in tqdm(enumerate(json_data['unique_img_val'])):
        filename = os.path.join(image_root, path)
        image = preprocess(Image.open(filename)).unsqueeze(0).to(device)

        with torch.no_grad(): 
            image_features = model.encode_image(image)
            val_embeds[idx] = image_features.cpu()

    test_embeds = np.zeros((len(json_data['unique_img_test']), embed_size), dtype=np.float32)

    for idx, path in tqdm(enumerate(json_data['unique_img_test'])):
        filename = os.path.join(image_root, path)
        image = preprocess(Image.open(filename)).unsqueeze(0).to(device)

        with torch.no_grad(): 
            image_features = model.encode_image(image)
            test_embeds[idx] = image_features.cpu()

    output_h5 = "./data_img.h5"
    print('Saving hdf5 to %s...' % output_h5)
    f = h5py.File(output_h5, 'w') 
    f.create_dataset("images" + '_train', data=train_embeds)
    f.create_dataset("images" + '_val', data=val_embeds)
    f.create_dataset("images" + '_test', data=test_embeds)
    f.close()

    return 


create_and_save(sys.argv[1], sys.argv[2])
