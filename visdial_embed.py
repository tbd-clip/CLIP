
import h5py
from tqdm import tqdm
import json
import os
import torch
import clip
import sys
from PIL import Image
import numpy as np

def create_and_save(clip_model, clip_preprocess, image_paths, image_root):

    embed_size = 512

    def get_embeds(json_data):
        image_embeds = np.zeros((len(json_data), embed_size), dtype=np.float32)

        for idx, path in tqdm(enumerate(json_data)):
            filename = os.path.join(image_root, path)
            image = clip_preprocess(Image.open(filename)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                image_embeds[idx] = image_features.cpu()

        return image_embeds

    train_embeds = get_embeds(image_paths['unique_img_train'])
    val_embeds = get_embeds(image_paths['unique_img_val'])
    test_embeds = get_embeds(image_paths['unique_img_test'])

    output_h5 = "./data_img.h5"
    print('Saving hdf5 to %s...' % output_h5)
    f = h5py.File(output_h5, 'w')
    f.create_dataset("images" + '_train', data=train_embeds)
    f.create_dataset("images" + '_val', data=val_embeds)
    f.create_dataset("images" + '_test', data=test_embeds)
    f.close()

    return 

if __name__ == "__main__":
    assert len(sys.argv) > 2, "Must specify json path and then path with images"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    image_paths_json, image_root = sys.argv[1], sys.argv[2]
    image_paths = json.load(open(image_paths_json, 'r')) # read json file

    create_and_save(clip_model, clip_preprocess, image_paths, image_root)