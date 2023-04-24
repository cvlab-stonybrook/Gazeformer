from typing import Optional, List
from timeit import default_timer as timer
import argparse
from datetime import datetime
import os
from os.path import join
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from models import Transformer
from gazeformer import gazeformer
from utils import seed_everything, fixations2seq, get_args_parser_train, save_model_train
from dataset import fixation_dataset, COCOSearch18Collator

torch.autograd.set_detect_anomaly(True)
   
    
def train(epoch, args, model, SlowOpt, MidOpt, FastOpt, loss_fn_token, loss_fn_y, loss_fn_x, loss_fn_t, train_dataloader, model_dir, model_name, device = 'cuda:0', im_h=20, im_w=32, patch_size=16):
    model.train()
    token_losses = 0
    reg_losses = 0
    t_losses = 0

    with tqdm(train_dataloader, unit="batch") as tepoch:
        minibatch = 0
        for batch_imgs, batch_tgt, batch_tgt_padding_mask, batch_tasks, batch_firstfix in tepoch:
            out_token, out_y, out_x, out_t = model(src = batch_imgs, tgt = batch_firstfix, task = batch_tasks)
            out_y, out_x = torch.clamp(out_y, min=0, max=im_h * patch_size - 2), torch.clamp(out_x, min=0, max=im_w * patch_size - 2)

            SlowOpt.zero_grad()
            MidOpt.zero_grad()
            FastOpt.zero_grad()

            tgt_out = batch_tgt.to(device)
            batch_tgt_padding_mask = batch_tgt_padding_mask.to(device)
            token_gt = batch_tgt_padding_mask.long()
            fixation_mask = torch.logical_not(batch_tgt_padding_mask).float()
            #predict padding or valid fixation
            token_loss = loss_fn_token(out_token.permute(1,2,0), token_gt)
            out_y = out_y.squeeze(-1).permute(1,0) * fixation_mask
            out_x = out_x.squeeze(-1).permute(1,0) * fixation_mask
            out_t = out_t.squeeze(-1).permute(1,0) * fixation_mask
            #calculate regression L1 losses for only valid ground truth fixations
            reg_loss = (loss_fn_y(out_y.float(), tgt_out[:, :, 0] * fixation_mask).sum(-1)/fixation_mask.sum(-1) + loss_fn_x(out_x.float(), tgt_out[:, :, 1]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            t_loss = (loss_fn_t(out_t.float(), tgt_out[:, :, 2]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            loss = token_loss + reg_loss + t_loss
            loss.backward()
            token_losses += token_loss.item()
            reg_losses += reg_loss.item()
            t_losses += t_loss.item()

            SlowOpt.step()
            MidOpt.step()
            FastOpt.step()
            
            minibatch += 1.
            tepoch.set_postfix(token_loss=token_losses/minibatch, reg_loss=reg_losses/minibatch, t_loss=t_losses/minibatch)
    save_model_train(epoch, args, model, SlowOpt, MidOpt, FastOpt, model_dir, model_name)
    return token_losses / len(train_dataloader),  reg_losses / len(train_dataloader), t_losses / len(train_dataloader)
    

def evaluate(model, loss_fn_token, loss_fn_y, loss_fn_x, loss_fn_t, valid_dataloader, device = 'cuda:0', im_h=20, im_w=32, patch_size=16):
    model.eval()
    token_losses = 0
    reg_losses = 0
    t_losses = 0

    with tqdm(valid_dataloader, unit="batch") as tepoch:
        minibatch = 0

        for batch_imgs, batch_tgt, batch_tgt_padding_mask, batch_tasks, batch_firstfix in tepoch:
            with torch.no_grad():
                out_token, out_y, out_x,out_t = model(src = batch_imgs, tgt = batch_firstfix, task = batch_tasks)
            out_y, out_x = torch.clamp(out_y, min=0, max=im_h *patch_size -2), torch.clamp(out_x, min=0, max=im_w *patch_size -2)

            tgt_out = batch_tgt.to(device)
            batch_tgt_padding_mask = batch_tgt_padding_mask.to(device)
            token_gt = batch_tgt_padding_mask.long()
            fixation_mask = torch.logical_not(batch_tgt_padding_mask).float()
            token_loss = loss_fn_token(out_token.permute(1,2,0), token_gt)
            out_y = out_y.squeeze(-1).permute(1,0) * fixation_mask
            out_x = out_x.squeeze(-1).permute(1,0) * fixation_mask
            out_t = out_t.squeeze(-1).permute(1,0) * fixation_mask
            reg_loss = (loss_fn_y(out_y.float(), tgt_out[:, :, 0] * fixation_mask).sum(-1)/fixation_mask.sum(-1) + loss_fn_x(out_x.float(), tgt_out[:, :, 1]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            t_loss = (loss_fn_t(out_t.float(), tgt_out[:, :, 2]*fixation_mask).sum(-1)/fixation_mask.sum(-1)).mean()
            
            token_losses += token_loss.item()
            reg_losses += reg_loss.item()
            t_losses += t_loss.item()
            minibatch += 1.
            tepoch.set_postfix(token_loss=token_losses/minibatch, reg_loss=reg_losses/minibatch, t_loss=t_losses/minibatch)
    return token_losses / len(valid_dataloader),  reg_losses / len(valid_dataloader), t_losses/len(valid_dataloader)
    
    
def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.cuda))
    device_id = args.cuda
    retraining = args.retraining
    last_checkpoint = args.last_checkpoint
    if retraining:
        model_dir = '/'.join(args.last_checkpoint.split('/')[:-1])
        args = argparse.Namespace(**json.load(open(join(model_dir, 'config.json'))))
        logfile = 'logs/output_' + last_checkpoint.split('/')[-2].split('_')[-1]+'.txt'
        args.cuda = device_id
    else:
        timenow = datetime.now().strftime("%d-%m-%Y-%H-%M-%S") 
        logfile = 'logs/output_' + timenow + '.txt'
        model_dir = join(args.model_root, 'train_' + timenow)
        os.mkdir(model_dir)
        
        open(logfile, 'w').close()
        with open(logfile, "a") as myfile:
            myfile.write(str(vars(args)) + '\n\n')
            myfile.close()
    print(str(vars(args)) + '\n\n')
    with open(join(model_dir, 'config.json'), "w") as outfile:
        json.dump(vars(args), outfile)
        outfile.close()


    model_name = 'gazeformer_'+str(args.num_encoder)+'E_'+str(args.num_decoder)+'D_'+str(args.batch_size)+'_'+str(args.hidden_dim)+'d'
    dataset_root = args.dataset_dir
    train_file = args.train_file
    valid_file = args.valid_file
    with open(join(dataset_root,
                   train_file)) as json_file:
        fixations_train = json.load(json_file)
    with open(join(dataset_root,
                   valid_file)) as json_file:
        fixations_valid = json.load(json_file)

        
    seq_train = fixations2seq(fixations =fixations_train, max_len = args.max_len)
            
    seq_valid = fixations2seq(fixations = fixations_valid, max_len = args.max_len)

    train_dataset = fixation_dataset(seq_train, img_ftrs_dir = args.img_ftrs_dir)
    valid_dataset = fixation_dataset(seq_valid, img_ftrs_dir = args.img_ftrs_dir)

    #target embeddings
    embedding_dict = np.load(open(join(dataset_root, 'embeddings.npy'), mode='rb'), allow_pickle = True).item()

    collate_fn = COCOSearch18Collator(embedding_dict, args.max_len, args.im_h, args.im_w, args.patch_size)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=6, collate_fn = collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=False, num_workers=6, collate_fn = collate_fn)

    transformer = Transformer(num_encoder_layers=args.num_encoder, nhead = args.nhead, d_model = args.hidden_dim, 
    num_decoder_layers=args.num_decoder, encoder_dropout = args.encoder_dropout, decoder_dropout = args.decoder_dropout, dim_feedforward = args.hidden_dim, 
    img_hidden_dim = args.img_hidden_dim, lm_dmodel = args.lm_hidden_dim, device = device).to(device)

    model = gazeformer(transformer, spatial_dim = (args.im_h, args.im_w), dropout=args.cls_dropout, max_len = args.max_len, device = device).to(device)

    loss_fn_token = torch.nn.NLLLoss()
    loss_fn_y = nn.L1Loss(reduction='none')
    loss_fn_x = nn.L1Loss(reduction='none')
    loss_fn_t = nn.L1Loss(reduction='none')

    #Disjoint optimization
    head_params = list(model.transformer.encoder.parameters()) + list(model.token_predictor.parameters())  
    SlowOpt = torch.optim.AdamW( head_params, lr=args.head_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    belly_params = list(model.generator_t_mu.parameters()) + list(model.generator_t_logvar.parameters()) 
    MidOpt = torch.optim.AdamW(belly_params, lr=args.belly_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    tail_params = list(model.transformer.decoder.parameters())  + list(model.generator_y_mu.parameters()) + list(model.generator_x_mu.parameters()) + list(model.generator_y_logvar.parameters()) + list(model.generator_x_logvar.parameters()) + list(model.querypos_embed.parameters()) + list(model.firstfix_linear.parameters()) 
    FastOpt = torch.optim.AdamW(tail_params, lr=args.tail_lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)

    start_epoch = 1
    if retraining:
        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        SlowOpt.load_state_dict(checkpoint['optim_slow'])
        MidOpt.load_state_dict(checkpoint['optim_mid'])
        FastOpt.load_state_dict(checkpoint['optim_fast'])
        start_epoch = checkpoint['epoch'] + 1
        print("Retraining from", start_epoch)
    for epoch in range(start_epoch, args.epochs+1):
        start_time = timer()
        train_token_loss, train_reg_loss, train_t_loss = train(epoch = epoch, args = args, model = model, SlowOpt = SlowOpt, FastOpt = FastOpt, MidOpt = MidOpt, loss_fn_token = loss_fn_token, loss_fn_y = loss_fn_y, loss_fn_x = loss_fn_x, loss_fn_t = loss_fn_t, train_dataloader = train_dataloader, model_dir = model_dir, model_name = model_name, device = device)
        end_time = timer()
        
        valid_token_loss, valid_reg_loss, valid_t_loss = evaluate(model = model, loss_fn_token = loss_fn_token, loss_fn_y = loss_fn_y, loss_fn_x = loss_fn_x, loss_fn_t=loss_fn_t, valid_dataloader = valid_dataloader, device = device)
        output_str = f"Epoch: {epoch}, Train token loss: {train_token_loss:.3f}, Train reg loss: {train_reg_loss:.3f}, Train T loss: {train_t_loss:.3f}, Val token loss: {valid_token_loss:.3f},  Val reg loss: {valid_reg_loss:.3f}, Valid T loss: {valid_t_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s, Saved to {model_dir+'/'+model_name}\n"
        print(output_str)
        with open(logfile, "a") as myfile:
            myfile.write(output_str)
            myfile.close()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gazeformer Train', parents=[get_args_parser_train()])
    args = parser.parse_args()
    main(args)
    
