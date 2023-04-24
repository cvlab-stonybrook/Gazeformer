import numpy as np
import gzip
from os.path import join
from multimatch import docomparison
from saliency_metrics import cc, nss
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0


def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))
    
def scanpath2clusters(meanshift, scanpath):
    string = []
    xs = scanpath['X']
    ys = scanpath['Y']
    for i in range(len(xs)):
        symbol = meanshift.predict([[xs[i], ys[i]]])[0]
        string.append(symbol)
    return string
    
    
def postprocessScanpaths(trajs):
    # convert actions to scanpaths
    scanpaths = []
    for traj in trajs:
        task_name, img_name, condition, subject, fixs = traj
        scanpaths.append({
            'X': fixs[:, 1],
            'Y': fixs[:, 0],
            'T': fixs[:, 2],
            'subject':subject,
            'name': img_name,
            'task': task_name,
            'condition': condition
        })
    return scanpaths
    
# compute sequence score
def compute_SS(preds, clusters, truncate, reduce='mean', print_clusters = False):
    results = []
    for scanpath in tqdm(preds):
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                     scanpath['name'][:-4])
        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']

        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        for gt in strings.values():
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                gt = gt[:truncate] if len(gt) > truncate else gt
                if print_clusters:
                    print(pred, gt)
                score = nw_matching(pred, gt)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results
    
# compute sequence score
def compute_SS_Time(preds, clusters, truncate, time_dict, reduce='mean', print_clusters = False, tempbin = 50):
    results = []
    for scanpath in tqdm(preds):
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                     scanpath['name'][:-4])
        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']
        

        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        for subj, gt in strings.items():
            if len(gt) > 0:
                time_string = time_dict[key+'-'+str(subj)]
                pred = pred[:truncate] if len(pred) > truncate else pred
                gtime_string = time_string[:truncate] if len(time_string) > truncate else time_string
                ptime_string = scanpath['T'][:truncate] if len(scanpath['T']) > truncate else scanpath['T']
                gt = gt[:truncate] if len(gt) > truncate else gt
                if print_clusters:
                    print(pred, gt)
                pred_time = []
                gt_time = []
                for p, t_p in zip(pred, ptime_string):
                    pred_time.extend([p for _ in range(int(t_p/tempbin))])
                for g, t_g in zip(gt, gtime_string):
                    gt_time.extend([g for _ in range(int(t_g/tempbin))])
                
                score = nw_matching(pred_time, gt_time)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results


def get_seq_score(preds, clusters, max_step, tasks=None, print_clusters = False):
    results = compute_SS(preds, clusters, truncate=max_step, print_clusters=print_clusters)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))
 
def get_seq_score_time(preds, clusters, max_step, time_dict, tasks=None, print_clusters = False):
    results = compute_SS_Time(preds, clusters, truncate=max_step, time_dict = time_dict, print_clusters=print_clusters)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))
        
def scanpath2categories(seg_map, scanpath):
    string = []
    xs = scanpath['X']
    ys = scanpath['Y']
    ts = scanpath['T']
    for x,y,t in zip(xs, ys, ts):
        symbol = str(int(seg_map[int(y), int(x)]))
        string.append((symbol, t))
    return string

# compute semantic sequence score
def compute_SSS(preds, fixations, truncate, segmentation_map_dir, reduce='mean'):
    results = []
    for scanpath in preds:
        #print(len(results), '/', len(preds), end='\r')
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                     scanpath['name'][:-4])
        strings = list(fixations[key])
        with gzip.GzipFile(join(segmentation_map_dir, scanpath['name'][:-3]+'npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        pred = scanpath2categories(segmentation_map, scanpath)
        scores = []
        human_scores = []
        pred = pred[:truncate] if len(pred) > truncate else pred
        pred_noT = [i[0] for i in pred]
        for gt in strings:
            if len(gt) > 0:
                gt = gt[:truncate] if len(gt) > truncate else gt
                gt_noT = [i[0] for i in gt]
                score = nw_matching(pred_noT, gt_noT)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results
    
# compute semantic sequence score
def compute_SSS_time(preds, fixations, truncate, segmentation_map_dir, reduce='mean', tempbin=50):
    results = []
    for scanpath in preds:
        #print(len(results), '/', len(preds), end='\r')
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                     scanpath['name'][:-4])
        strings = list(fixations[key])
        with gzip.GzipFile(join(segmentation_map_dir, scanpath['name'][:-3]+'npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        pred = scanpath2categories(segmentation_map, scanpath)
        scores = []
        human_scores = []
        pred_T = []
        
        pred = pred[:truncate] if len(pred) > truncate else pred
        for p in pred:
            pred_T.extend([p[0] for _ in range(int(p[1]/tempbin))])
        for gt in strings:
            gt_T = []
            if len(gt) > 0:
                gt = gt[:truncate] if len(gt) > truncate else gt
                for g in gt:
                    gt_T.extend([g[0] for _ in range(int(g[1]/tempbin))])
                
                score = nw_matching(pred_T, gt_T)
                scores.append(score)
                del gt_T
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results


def get_semantic_seq_score(preds, fixations, max_step, segmentation_map_dir, tasks=None):
    results = compute_SSS(preds, fixations, truncate=max_step, segmentation_map_dir = segmentation_map_dir)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))
        
def get_semantic_seq_score_time(preds, fixations, max_step, segmentation_map_dir, tasks=None):
    results = compute_SSS_time(preds, fixations, truncate=max_step, segmentation_map_dir = segmentation_map_dir)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))
        
        
        
def multimatch(s1, s2, im_size):
    s1x = s1['X']
    s1y = s1['Y']
    s1t = s1['T']
    l1 = len(s1x)
    if l1 < 3:
        scanpath1 = np.ones((3, 3), dtype=np.float32)
        scanpath1[:l1, 0] = s1x
        scanpath1[:l1, 1] = s1y
        scanpath1[:l1, 2] = s1t[:l1]
    else:
        scanpath1 = np.ones((l1, 3), dtype=np.float32)
        scanpath1[:, 0] = s1x
        scanpath1[:, 1] = s1y
        scanpath1[:, 2] = s1t[:l1]
    s2x = s2['X']
    s2y = s2['Y']
    s2t = s2['T']
    l2 = len(s2x)
    if l2 < 3:
        scanpath2 = np.ones((3, 3), dtype=np.float32)
        scanpath2[:l2, 0] = s2x
        scanpath2[:l2, 1] = s2y
        scanpath2[:l2, 2] = s2t[:l2]
    else:
        scanpath2 = np.ones((l2, 3), dtype=np.float32)
        scanpath2[:, 0] = s2x
        scanpath2[:, 1] = s2y
        scanpath2[:, 2] = s2t[:l2]
    mm = docomparison(scanpath1, scanpath2, sz=im_size)
    return mm[0]


def compute_mm(human_trajs, model_trajs, im_w, im_h, tasks=None):
    """
    compute scanpath similarity using multimatch
    """
    all_mm_scores = []
    for traj in model_trajs:
        img_name = traj['name']
        task = traj['task']
        gt_trajs = list(
            filter(lambda x: x['name'] == img_name and x['task'] == task,
                   human_trajs))
        all_mm_scores.append((task,
                              np.mean([
                                  multimatch(traj, gt_traj, (im_w, im_h))#[:4]
                                  for gt_traj in gt_trajs
                              ],
                                      axis=0)))

    if tasks is not None:
        mm_tasks = {}
        for task in tasks:
            mm = np.array([x[1] for x in all_mm_scores if x[0] == task])
            mm_tasks[task] = np.mean(mm, axis=0)
        return mm_tasks
    else:
        return np.mean([x[1] for x in all_mm_scores], axis=0)
        
        
        
def _Levenshtein_Dmatrix_initializer(len1, len2):
    Dmatrix = []

    for i in range(len1):
        Dmatrix.append([0] * len2)

    for i in range(len1):
        Dmatrix[i][0] = i

    for j in range(len2):
        Dmatrix[0][j] = j

    return Dmatrix


def _Levenshtein_cost_step(Dmatrix, string_1, string_2, i, j, substitution_cost=1):
    char_1 = string_1[i - 1]
    char_2 = string_2[j - 1]

    # insertion
    insertion = Dmatrix[i - 1][j] + 1
    # deletion
    deletion = Dmatrix[i][j - 1] + 1
    # substitution
    substitution = Dmatrix[i - 1][j - 1] + substitution_cost * (char_1 != char_2)

    # pick the cheapest
    Dmatrix[i][j] = min(insertion, deletion, substitution)


def _Levenshtein(string_1, string_2, substitution_cost=1):
    # get strings lengths and initialize Distances-matrix
    len1 = len(string_1)
    len2 = len(string_2)
    Dmatrix = _Levenshtein_Dmatrix_initializer(len1 + 1, len2 + 1)

    # compute cost for each step in dynamic programming
    for i in range(len1):
        for j in range(len2):
            _Levenshtein_cost_step(Dmatrix,
                                   string_1, string_2,
                                   i + 1, j + 1,
                                   substitution_cost=substitution_cost)

    if substitution_cost == 1:
        max_dist = max(len1, len2)
    elif substitution_cost == 2:
        max_dist = len1 + len2

    return Dmatrix[len1][len2]
    
    
def compute_ED(preds, clusters, truncate, reduce='mean', print_clusters = False):
    results = []
    for scanpath in preds:
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                     scanpath['name'][:-4])
        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']

        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        for gt in strings.values():
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                gt = gt[:truncate] if len(gt) > truncate else gt
                if print_clusters:
                    print(pred, gt)
                score = _Levenshtein(pred, gt)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results

def compute_ED_Time(preds, clusters, truncate, time_dict, reduce='mean', print_clusters = False, tempbin = 50):
    results = []
    for scanpath in preds:
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                     scanpath['name'][:-4])
        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']
        

        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        for subj, gt in strings.items():
            if len(gt) > 0:
                time_string = time_dict[key+'-'+str(subj)]
                pred = pred[:truncate] if len(pred) > truncate else pred
                gtime_string = time_string[:truncate] if len(time_string) > truncate else time_string
                ptime_string = scanpath['T'][:truncate] if len(scanpath['T']) > truncate else scanpath['T']
                gt = gt[:truncate] if len(gt) > truncate else gt
                if print_clusters:
                    print(pred, gt)
                pred_time = []
                gt_time = []
                for p, t_p in zip(pred, ptime_string):
                    pred_time.extend([p for _ in range(int(t_p/tempbin))])
                for g, t_g in zip(gt, gtime_string):
                    gt_time.extend([g for _ in range(int(t_g/tempbin))])
                
                score = _Levenshtein(pred_time, gt_time)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results
    
def get_ed(preds, clusters, max_step, tasks=None, print_clusters = False):
    results = compute_ED(preds, clusters, truncate=max_step, print_clusters=print_clusters)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))
        
def get_ed_time(preds, clusters, max_step, time_dict, tasks=None, print_clusters = False):
    results = compute_ED_Time(preds, clusters, truncate=max_step, time_dict = time_dict, print_clusters=print_clusters)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))


# compute semantic sequence score
def compute_SED(preds, fixations, truncate, segmentation_map_dir, reduce='mean'):
    results = []
    for scanpath in preds:
        #print(len(results), '/', len(preds), end='\r')
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                     scanpath['name'][:-4])
        strings = list(fixations[key])
        with gzip.GzipFile(join(segmentation_map_dir, scanpath['name'][:-3]+'npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        pred = scanpath2categories(segmentation_map, scanpath)
        scores = []
        human_scores = []
        pred = pred[:truncate] if len(pred) > truncate else pred
        pred_noT = [i[0] for i in pred]
        for gt in strings:
            if len(gt) > 0:
                gt = gt[:truncate] if len(gt) > truncate else gt
                gt_noT = [i[0] for i in gt]
                score = _Levenshtein(pred_noT, gt_noT)
                scores.append(score)
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results
    
# compute semantic sequence score
def compute_SED_time(preds, fixations, truncate, segmentation_map_dir, reduce='mean', tempbin=50):
    results = []
    for scanpath in preds:
        #print(len(results), '/', len(preds), end='\r')
        key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                     scanpath['name'][:-4])
        strings = list(fixations[key])
        with gzip.GzipFile(join(segmentation_map_dir, scanpath['name'][:-3]+'npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        pred = scanpath2categories(segmentation_map, scanpath)
        scores = []
        human_scores = []
        pred_T = []
        
        pred = pred[:truncate] if len(pred) > truncate else pred
        for p in pred:
            pred_T.extend([p[0] for _ in range(int(p[1]/tempbin))])
        for gt in strings:
            gt_T = []
            if len(gt) > 0:
                gt = gt[:truncate] if len(gt) > truncate else gt
                for g in gt:
                    gt_T.extend([g[0] for _ in range(int(g[1]/tempbin))])
                
                score = _Levenshtein(pred_T, gt_T)
                scores.append(score)
                del gt_T
        result = {}
        result['condition'] = scanpath['condition']
        result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results
    
def get_semantic_ed(preds, fixations, max_step, segmentation_map_dir, tasks=None):
    results = compute_SED(preds, fixations, truncate=max_step, segmentation_map_dir = segmentation_map_dir)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))
        
def get_semantic_ed_time(preds, fixations, max_step, segmentation_map_dir, tasks=None):
    results = compute_SED_time(preds, fixations, truncate=max_step, segmentation_map_dir = segmentation_map_dir)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))
    
    
def get_cc(pred_dict, gt_dict):
    cc_res = []
    for key in gt_dict.keys():
        gt_list = gt_dict[key]
        pred_list = pred_dict[key]
        for g in gt_list:
            gt_map = cv2.imread(g,0)
            res = []
            for p in pred_list:
                pred_map = cv2.imread(p,0)
                res.append(cc(pred_map, gt_map))
        cc_res.append(np.mean(res))
    return np.mean(cc_res)


def get_nss(pred_dict, gt_dict):
    nss_res = []
    for key in gt_dict.keys():
        gt_list = gt_dict[key]
        pred_list = pred_dict[key]
        for g in gt_list:
            gt_map = cv2.imread(g,0)
            res = []
            for p in pred_list:
                pred_map = cv2.imread(p,0)
                res.append(nss(pred_map, gt_map))
        nss_res.append(np.mean(res))
    return np.mean(nss_res)
