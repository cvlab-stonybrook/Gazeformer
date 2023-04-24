import argparse
from os.path import join
import json
import numpy as np
import torch

from models import Transformer
from gazeformer import gazeformer
from utils import seed_everything, get_args_parser_test
from metrics import postprocessScanpaths, get_seq_score, get_seq_score_time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



def run_model(model, src, task, device = "cuda:0", im_h=20, im_w=32, patch_size = 16, num_samples = 1):
    src = src.to(device).repeat(num_samples, 1, 1)
    task = torch.tensor(task.astype(np.float32)).to(device).unsqueeze(0).repeat(num_samples, 1)
    firstfix = torch.tensor([(im_h//2)*patch_size, (im_w//2)*patch_size]).unsqueeze(0).repeat(num_samples, 1)
    with torch.no_grad():
        token_prob, ys, xs, ts = model(src = src, tgt = firstfix, task = task)
    token_prob = token_prob.detach().cpu().numpy()
    ys = ys.cpu().detach().numpy()
    xs = xs.cpu().detach().numpy()
    ts = ts.cpu().detach().numpy()
    scanpaths = []
    for i in range(num_samples):
        ys_i = [(im_h//2) * patch_size] + list(ys[:, i, 0])[1:]
        xs_i = [(im_w//2) * patch_size] + list(xs[:, i, 0])[1:]
        ts_i = list(ts[:, i, 0])
        token_type = [0] + list(np.argmax(token_prob[:, i, :], axis=-1))[1:]
        scanpath = []
        for tok, y, x, t in zip(token_type, ys_i, xs_i, ts_i):
            if tok == 0:
                scanpath.append([min(im_h * patch_size - 2, y),min(im_w * patch_size - 2, x), t])
            else:
                break
        scanpaths.append(np.array(scanpath))
    return scanpaths
    
    
def test(args):
    trained_model = args.trained_model
    device = torch.device('cuda:{}'.format(args.cuda))
    transformer = Transformer(num_encoder_layers=args.num_encoder, nhead = args.nhead, d_model = args.hidden_dim, num_decoder_layers=args.num_decoder, dim_feedforward = args.hidden_dim, img_hidden_dim = args.img_hidden_dim, lm_dmodel = args.lm_hidden_dim, device = device).to(device)
    model = gazeformer(transformer = transformer, spatial_dim = (args.im_h, args.im_w), max_len = args.max_len, device = device).to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device)['model'])
    model.eval()
    dataset_root = args.dataset_dir
    img_ftrs_dir = args.img_ftrs_dir
    max_len = args.max_len
    fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
    if args.condition == 'absent':
        fixation_path = join(dataset_root, 'coco_search18_fixations_TA_test.json')
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(filter(lambda x: x['split'] == 'test' and x['condition']==args.condition, human_scanpaths))
    if args.zerogaze:
        test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
        print("Zero Gaze on", args.task.replace('_', ' '))
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                     traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])
    
    test_task_img_pairs = np.unique([traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
    embedding_dict = np.load(open(join(dataset_root, 'embeddings.npy'), mode='rb'), allow_pickle = True).item()
    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))
    for target_traj in tqdm(test_task_img_pairs):
        task_name, name, condition = target_traj.split('_')
        image_ftrs = torch.load(join(img_ftrs_dir, task_name.replace(' ', '_'), name.replace('jpg', 'pth'))).unsqueeze(0)
        task_emb = embedding_dict[task_name]

        scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples)
        for idx, scanpath in enumerate(scanpaths):
            pred_list.append((task_name, name, condition, idx+1, scanpath))

    predictions = postprocessScanpaths(pred_list)
    fix_clusters = np.load(join('./data', 'clusters.npy'), allow_pickle=True).item()
    
    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)
    print("Calculating Sequence Score with Duration...")
    seq_score_t = get_seq_score_time(predictions, fix_clusters, max_len, t_dict)
    return seq_score, seq_score_t
    
def main(args):
    seed_everything(args.seed)
    seq_score, seq_score_t = test(args)
    print('Sequence Score : {:.3f}, Sequence Score with Duration : {:.3f}'.format(seq_score, seq_score_t))
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gaze Transformer Test', parents=[get_args_parser_test()])
    args = parser.parse_args()
    main(args)
    
