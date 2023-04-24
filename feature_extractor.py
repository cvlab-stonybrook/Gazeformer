from models import ResNetCOCO
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from sentence_transformers import SentenceTransformer
import PIL
import os
from os.path import join, isdir, isfile
import numpy as np
import argparse
        
def image_data(dataset_path, device = 'cuda:0', overwrite = False):
    resize_dim = (320 * 2, 512 * 2)
    src_path = join(dataset_path, 'images/')
    target_path = join(dataset_path, 'image_features/')
    resize = T.Resize(resize_dim)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    folders = [i for i in os.listdir(src_path) if isdir(join(src_path, i))]

    bbone = ResNetCOCO(device = device).to(device)
    for folder in folders:
        if not (os.path.exists(join(target_path, folder)) and os.path.isdir(join(target_path, folder))):
            os.mkdir(join(target_path, folder))
        files =  [i for i in os.listdir(join(src_path, folder)) if isfile(join(src_path, folder, i)) and i.endswith('.jpg')]
        for f in files:
           if overwrite == False and os.path.exists(join(target_path, folder, f.replace('jpg', 'pth'))):
               continue
           PIL_image = PIL.Image.open(join(src_path, folder, f))
           tensor_image = normalize(resize(T.functional.to_tensor(PIL_image))).unsqueeze(0)

           features = bbone(tensor_image).squeeze().detach().cpu()
           torch.save(features, join(target_path, folder, f.replace('jpg', 'pth')))
           
           
def text_data(dataset_path, device = 'cuda:0', lm_model = 'sentence-transformers/stsb-roberta-base-v2'):
    src_path = join(dataset_path, 'images/')
    tasks = [' '.join(i.split('_')) for i in os.listdir(src_path) if isdir(join(src_path, i))]

    lm = SentenceTransformer(lm_model, device=device).eval()
    embed_dict = {}
    for task in tasks:
       embed_dict[task] = lm.encode(task)
    with open(join(dataset_path,'embeddings.npy'), 'wb') as f:
        np.save(f, embed_dict, allow_pickle = True)
        f.close()
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Gazeformer Feature Extractor Utils', add_help=False)
    parser.add_argument('--dataset_path', default= '/home/sounakm/dataset', type=str)
    parser.add_argument('--lm_model', default= 'sentence-transformers/stsb-roberta-base-v2', type=str)
    parser.add_argument('--cuda', default=0, type=int)
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.cuda))
    image_data(dataset_path = args.dataset_path, device = device, overwrite = True)
    text_data(dataset_path = args.dataset_path, device = device, lm_model = args.lm_model)

