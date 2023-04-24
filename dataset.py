from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from os.path import join
import numpy as np


class fixation_dataset(Dataset):
    def __init__(self, fixs, img_ftrs_dir):
        self.fixs = fixs
        self.img_ftrs_dir = img_ftrs_dir

        
    def __len__(self):
        return len(self.fixs)
        
    def __getitem__(self, idx):
        fixation = self.fixs[idx]

        image_ftrs = torch.load(join(self.img_ftrs_dir, fixation['task'].replace(' ', '_'), fixation['img_name'].replace('jpg', 'pth'))).unsqueeze(0)

        
        return {'task': fixation['task'], 'tgt_y': fixation['tgt_seq_y'].float(), 'tgt_x': fixation['tgt_seq_x'].float(), 'tgt_t': fixation['tgt_seq_t'].float(),'src_img': image_ftrs }
        

class COCOSearch18Collator(object):
    def __init__(self, embedding_dict, max_len, im_h, im_w, patch_size):
        self.embedding_dict = embedding_dict
        self.max_len = max_len
        self.im_h = im_h
        self.im_w = im_w
        self.patch_size = patch_size
        self.PAD = [-3, -3, -3]

    def __call__(self, batch):
        batch_tgt_y = []
        batch_tgt_x = []
        batch_tgt_t = []
        batch_imgs = []
        batch_tasks = []
        
        for t in batch:
            batch_tgt_y.append(t['tgt_y'])
            batch_tgt_x.append(t['tgt_x'])
            batch_tgt_t.append(t['tgt_t'])
            batch_imgs.append(t['src_img'])
            batch_tasks.append(self.embedding_dict[t['task']])
        
        batch_tgt_y.append(torch.zeros(self.max_len))
        batch_tgt_x.append(torch.zeros(self.max_len))
        batch_tgt_t.append(torch.zeros(self.max_len))
        batch_tgt_y = pad_sequence(batch_tgt_y, padding_value=self.PAD[0])[:, :-1].unsqueeze(-1)
        batch_tgt_x = pad_sequence(batch_tgt_x, padding_value=self.PAD[1])[:, :-1].unsqueeze(-1)
        batch_tgt_t = pad_sequence(batch_tgt_t, padding_value=self.PAD[2])[:, :-1].unsqueeze(-1)
        
        batch_imgs = torch.cat(batch_imgs, dim = 0)
        batch_tgt = torch.cat([batch_tgt_y, batch_tgt_x, batch_tgt_t], dim = -1).long().permute(1, 0, 2)
        batch_firstfix = torch.tensor([(self.im_h//2)*self.patch_size, (self.im_w//2)*self.patch_size]).unsqueeze(0).repeat(batch_imgs.size(0), 1)
        batch_tgt_padding_mask = batch_tgt[:, :, 0] == self.PAD[0]
        
        
        return batch_imgs, batch_tgt, batch_tgt_padding_mask, torch.tensor(batch_tasks), batch_firstfix

        
        
