import logging
import sys
import os
import cv2
import numpy as np
import math
from multiprocessing import Process, Queue
import threading
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity
import importlib

from tpu_perf.infer import SGInfer
from tpu_perf.harness import harness


def split_img(img_data, net_in_size, overlap, info_dict={}):
    # logging.debug(f'split_img: {img_data.shape}, {net_in_size}, {overlap}, {info_dict}')
    hk, wk = net_in_size
    h_lap, w_lap = overlap
    hs, ws = hk - h_lap, wk - w_lap
    n, c, h_, w_ = img_data.shape
    h, w = max(h_, hk), max(w_, wk)

    # if input is too small...
    if h_ < h or w_ < w:
        dh, dw = h - h_, w - w_
        img_data_ = np.zeros((n, c, h, w))
        for i in range(n):
            img = img_data[i].transpose((1,2,0))
            img = cv2.copyMakeBorder(img, 0, dh, 0, dw, cv2.BORDER_REFLECT)
            img_data_[i] = img.transpose((2,0,1))
        img_data = img_data_

    h_num = math.ceil((h - hk) / hs) + 1
    w_num = math.ceil((w - wk) / ws) + 1

    lr_list = []
    info_list = []
    i = 0
    for hn in range(h_num):
        for wn in range(w_num):
            if hn == h_num - 1:
                top, bottom = max(0, h - hk), h
            else:
                top, bottom = hn * hs, hn * hs + hk
            if wn == w_num - 1:
                left, right = max(0, w - wk), w
            else:
                left, right = wn * ws, wn * ws + wk
            lr = img_data[..., top:bottom, left:right].copy()
            info_dict.update(index=i, num=h_num*w_num)
            lr_list.append(lr)
            info_list.append(info_dict.copy())
            i += 1
            # logging.debug('{:5d}, {:5d}, {:5d}, {:5d} - shape={}'.format(top, bottom, left, right, lr.shape))
            # logging.debug(f'split_img: info_dict={info_dict}')

    return lr_list, info_list


def stitch_img(data, target_size, overlap, blend=False):

    sr_list = data['output']
    sr_index = data['index']
    # logging.debug(f'stitch_img: sr_list={len(sr_list)}, sr_index={sorted(sr_index)}, full_size={target_size}, overlap={overlap}, blend={blend}')

    h_, w_ = target_size
    n, c, hk, wk = sr_list[0].shape
    h_lap, w_lap = overlap
    hs, ws = hk - h_lap, wk - w_lap
    h, w = max(h_, hk), max(w_, wk)     # if input is too small...

    h_num = math.ceil((h - hk) / hs) + 1
    w_num = math.ceil((w - wk) / ws) + 1

    i = 0
    if blend:
        output = np.zeros((n, c, h, w), dtype=np.float32)
        num_map = output.copy()
        for hn in range(h_num):
            for wn in range(w_num):
                if hn == h_num - 1:
                    top, bottom = h - hk, h
                else:
                    top, bottom = hn * hs, hn * hs + hk
                if wn == w_num - 1:
                    left, right = w - wk, w
                else:
                    left, right = wn * ws, wn * ws + wk
                sr = sr_list[sr_index.index(i)]
                output[..., top:bottom, left:right] += sr
                num_map[..., top:bottom, left:right] += 1   # counter
                i += 1
        assert not (num_map == 0).any()
        output = output / num_map
    else:
        output = np.full((n, c, h, w), float('nan'), dtype=np.float32)
        for hn in range(h_num):
            for wn in range(w_num):
                if hn == 0:
                    if h_num == 1:
                        top_shave, bottom_shave = 0, 0
                    else:
                        top_shave, bottom_shave = 0, int(h_lap / 2)
                    top, bottom = top_shave, hn * hs + hk - bottom_shave
                elif hn == h_num - 1:
                    top_shave, bottom_shave = hk - h + hn * hs + int(h_lap / 2), 0
                    top, bottom = h - hk + top_shave, h - bottom_shave
                else:
                    top_shave, bottom_shave = int(h_lap / 2), int(h_lap / 2)
                    top, bottom = hn * hs + top_shave, hn * hs + hk - bottom_shave
                if wn == 0:
                    if w_num == 1:
                        left_shave, right_shave = 0, 0
                    else:
                        left_shave, right_shave = 0, int(w_lap / 2)
                    left, right = left_shave, wn * ws + wk - right_shave
                elif wn == w_num - 1:
                    left_shave, right_shave = wk - w + wn * ws + int(w_lap / 2), 0
                    left, right = w - wk + left_shave, w - right_shave
                else:
                    left_shave, right_shave = int(w_lap / 2), int(w_lap / 2)
                    left, right = wn * ws + left_shave, wn * ws + wk - right_shave
                # logging.info('{:5d}, {:5d}, {:5d}, {:5d} <- {:5d}, {:5d}, {:5d}, {:5d}'.format(top, bottom, left, right, top_shave, hk - bottom_shave, left_shave, wk - right_shave))
                sr = sr_list[sr_index.index(i)]
                output[..., top:bottom, left:right] = sr[..., top_shave:hk - bottom_shave, left_shave:wk - right_shave]
                i += 1
        assert not np.isnan(output).all(axis=(0,1)).any()

    output = output[..., :h_, :w_]

    return output


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.

    # gray img
    if img.ndim == 2:
        img = np.stack([img,img,img], 2)
    elif img.shape[2] == 1:
        img = np.concatenate([img,img,img], 2)

    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return structural_similarity(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(structural_similarity(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return structural_similarity(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_lpips(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2 and img2.ndim == 2:
        img1 = np.stack([img1,img1,img1], 2)
        img2 = np.stack([img2,img2,img2], 2)
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border, ...]
    img2 = img2[border:h-border, border:w-border, ...]

    img1 = lpips.im2tensor(img1)    # torch.Tensor
    img2 = lpips.im2tensor(img2)

    # Compute distance
    lpips_distance = loss_fn.forward(img1, img2)
    lpips_distance = lpips_distance.item()

    return lpips_distance


def evaluate(sr_img, hr_img, criterion, scale):
    logging.debug(f'evaluate: sr_img={sr_img.shape} {sr_img.dtype}, hr_img={hr_img.shape} {hr_img.dtype}')
    if hr_img.shape != sr_img.shape:
        hr_h, hr_w, _ = hr_img.shape
        sr_h, sr_w, _ = sr_img.shape
        if hr_w - sr_w >= scale or hr_h - sr_h >= scale:
            logging.warning(f'evaluate: hr_img is larger than expected')
        elif hr_w - sr_w < 0 or hr_h - sr_h < 0:
            logging.warning(f'evaluate: hr_img is smaller than expected')
        w, h = min(hr_w, sr_w), min(hr_h, sr_h)
        if w != hr_w or h != hr_h:
            hr_img = hr_img[:h, :w, :]
            logging.debug(f'evaluate: cropped_hr_img={hr_img.shape} {hr_img.dtype}')
        if w != sr_w or h != sr_h:
            sr_img = sr_img[:h, :w, :]
            logging.debug(f'evaluate: cropped_sr_img={sr_img.shape} {sr_img.dtype}')
    sr_img_y = rgb2ycbcr(sr_img, only_y=True)
    hr_img_y = rgb2ycbcr(hr_img, only_y=True)
    psnr, ssim, lpi = None, None, None
    if 'psnr' in criterion:
        psnr = calculate_psnr(sr_img_y, hr_img_y, border=scale)
    if 'ssim' in criterion:
        ssim = calculate_ssim(sr_img_y, hr_img_y, border=scale)
    if 'lpips' in criterion:
        lpi = calculate_lpips(sr_img, hr_img, border=scale)
    return dict(psnr=psnr, ssim=ssim, lpips=lpi)


def read_image_list(lr_dir, hr_dir):
    imlist = []
    assert os.path.exists(lr_dir) and os.path.exists(hr_dir), 'lr_dir or hr_dir does not exist'
    for lr_name in os.listdir(lr_dir):
        hr_name = lr_name.split('x')[0] + os.path.splitext(lr_name)[-1]
        lr_path = os.path.join(lr_dir, lr_name)
        hr_path = os.path.join(hr_dir, hr_name)
        if os.path.exists(hr_path):
            imlist.append((lr_path, hr_path))
        else:
            logging.debug(f'{hr_path} not found')
    assert len(imlist), 'no matched images in lr_dir and hr_dir'
    return imlist


def sever(l, n):
    step = math.ceil(len(l) / n)
    return [l[i * step : (i + 1) * step] for i in range(0, n)]


class Runner:
    def __init__(self, bmodel, devices, lr_dir, hr_dir, config, threads):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.config = config
        self.model = SGInfer(bmodel, devices=devices)
        self.input_info = self.model.get_input_info()
        self.output_info = self.model.get_output_info()
        logging.info(f'input_info: {self.input_info}')
        logging.info(f'output_info: {self.output_info}')
        self.input_info = next(iter(self.input_info.values()))
        self.output_info = next(iter(self.output_info.values()))
        self.input_shape = self.input_info['shape']
        self.input_scale = self.input_info['scale']
        self.output_shape = self.output_info['shape']
        self.scale = int(self.output_shape[-1] / self.input_shape[-1])

        self.pairs = None       # list of (lr_path, hr_path)
        self.task_info = dict()
        self.outputs = {}       # dict of hr_path->outputs,infos
        self.scores = {}        # dict of hr_path->score
        self.stats = dict(psnr=0, ssim=0, lpips=0)

        self.pairs = read_image_list(lr_dir, hr_dir)
        self.num = len(self.pairs)
        logging.info(f'****** total num: {self.num} ******')

        parts = sever(self.pairs, threads)
        self.pre_procs = []
        self.q = Queue(maxsize=threads * 2)
        for part in parts:
            p = Process(target=self.preprocess, args=(part,), daemon=True)   # evenly distributed to precesses
            self.pre_procs.append(p)
            p.start()

        self.relay = threading.Thread(target=self.relay)
        self.relay.start()

        for _, hr_path in self.pairs:
            self.scores[hr_path] = {}.fromkeys(['psnr', 'ssim', 'lpips'])

        self.post = threading.Thread(target=self.postprocess)
        self.post.start()

    def preprocess(self, part):
        try:
            self._preprocess(part)
        except Exception as err:
            logging.error(f'Preprocess failed, {err}')
            raise

    def _preprocess(self, part):
        logging.debug(f'[PID={os.getpid()}] preprocess process started')

        batch_size = self.input_shape[0]
        input_size = self.input_shape[2:]
        input_scale = self.input_scale
        is_fp32 = input_scale == 1

        bulk_info = []
        bulk = []
        def enqueue():
            nonlocal bulk_info, bulk
            if not bulk:
                return
            # logging.debug(f'[PID={os.getpid()}] self.q.put: len={len(bulk)}, bulk_info={bulk_info}')
            self.q.put((np.concatenate(bulk, axis=0), bulk_info))
            bulk = []
            bulk_info = []
        for lr_path, hr_path in part:
            ori_img = cv2.imread(lr_path)
            logging.debug(f'[PID={os.getpid()}] preprocessing {lr_path:12s}, shape={ori_img.shape}')

            # convert color
            if ori_img.ndim != 3:   # if gray image
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_GRAY2BGR)
            if self.config.get('bgr2rgb'):
                ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

            # normalize
            data = ori_img.astype(np.float32)
            if 'mean' in self.config:
                data -= self.config['mean']
            if 'scale' in self.config:
                data *= self.config['scale']

            data = data.transpose([2, 0, 1])[None, ...] # NCHW

            # self_ensemble method
            enhanced_data = [data]
            lr_list, info_list = [], []
            if self.config['self_ensemble']:
                enhanced_data.append(data[:, :, :, ::-1].copy())
                enhanced_data.append(data[:, :, ::-1, :].copy())
                enhanced_data.append(data.transpose((0, 1, 3, 2)).copy())
            # split image
            for i,d in enumerate(enhanced_data):
                info_dict={'hr_path': hr_path, 'lr_size': d.shape[2:],
                           'self_ensemble_i': i, 'self_ensemble_n': len(enhanced_data)}
                _lr_list, _info_list = split_img(d, input_size, self.config['overlap'], info_dict=info_dict)
                lr_list += _lr_list     # list of CHW
                info_list += _info_list

            # add to task queue
            # logging.debug(f'[PID={os.getpid()}] preprocessed: len={len(lr_list)}, info_list={info_list}')
            for lr, info in zip(lr_list, info_list):
                dtype = np.float32
                if not is_fp32:
                    lr *= input_scale
                    dtype = np.int8

                logging.debug(f'preprocess: info={info}')
                bulk.append(lr.astype(dtype))
                bulk_info.append(info)
                if len(bulk) < batch_size:
                    continue
                enqueue()
            enqueue()
        enqueue()

    def relay(self):
        try:
            while True:
                task = self.q.get()
                if task is None:
                    break
                self._relay(task)
        except Exception as err:
            logging.error(f'Relay task failed, {err}')
            raise

    def _relay(self, task):
        data, infos = task
        logging.debug(f'relay: data={data.shape}, len={len(infos)}')
        task_id = self.model.put(data)
        self.task_info[task_id] = infos    # list of dict(hr_path, lr_size, self_ensemble_i, self_ensemble_n, index, num)
        # logging.debug(f'relay: task_id={task_id}, infos={infos}')

    def postprocess(self):
        try:
            self._postprocess()
        except Exception as err:
            logging.error(f'Task postprocess failed, {err}')
            raise

    def _postprocess(self):
        n = 0
        while True:
            task_id, results, valid = self.model.get()
            if task_id == 0:
                break
            output = results[0] # NCHW
            ## get network outputs
            infos = self.task_info[task_id]
            batch_size = self.input_shape[0]
            output = np.split(output, batch_size, axis=0)
            assert len(infos) == len(output)
            logging.debug(f'postprocess: task_id={task_id}, valid={valid}, len(output)={len(output)}')
            hr_path = infos[0]['hr_path']
            lr_size = infos[0]['lr_size']
            self_ensemble_i = infos[0]['self_ensemble_i']
            self_ensemble_n = infos[0]['self_ensemble_n']
            index = [info['index'] for info in infos]
            num = infos[0]['num']
            if hr_path not in self.outputs.keys():
                self.outputs[hr_path] = {}
                self.outputs[hr_path]['self_ensemble_n'] = self_ensemble_n
                for i in range(self_ensemble_n):
                    self.outputs[hr_path][f'self_ensemble_{i}'] = {}
                    self.outputs[hr_path][f'self_ensemble_{i}']['output'] = []
                    self.outputs[hr_path][f'self_ensemble_{i}']['index'] = []
                    self.outputs[hr_path][f'self_ensemble_{i}']['num'] = num
                    self.outputs[hr_path][f'self_ensemble_{i}']['lr_size'] = lr_size[::-1] if i == 3 else lr_size
                logging.debug(f'postprocess: self.outputs["{hr_path}"]={self.outputs[hr_path][f"self_ensemble_{i}"]}')
            k = f'self_ensemble_{self_ensemble_i}'
            self.outputs[hr_path][k]['output'] += output
            self.outputs[hr_path][k]['index'] += index
            logging.debug(f'postprocess: self.outputs["{hr_path}"]["{k}"]: add len={len(output)}, index={index}')
            ## stich image pieces together
            n1 = len(self.outputs[hr_path][k]['index'])
            n2 = self.outputs[hr_path][k]['num']
            if n1 >= n2:
                assert n1 == n2
                data = self.outputs[hr_path][k]
                lr_size = self.outputs[hr_path][k]['lr_size']
                sr_size = [s*self.scale for s in lr_size]
                sr_overlap = [o*self.scale for o in self.config['overlap']]
                im = stitch_img(data, sr_size, sr_overlap, self.config['blend'])
                self.outputs[hr_path][k] = im    # NCHW
                logging.debug(f'postprocess: self.outputs["{hr_path}"]["{k}"]: completed, shape={im.shape}')
                ## check if the SR images is complete
                completed = 0
                for i in range(self_ensemble_n):
                    if type(self.outputs[hr_path][f'self_ensemble_{i}']) != dict:
                        completed += 1
                if completed == self_ensemble_n:
                    ## self_ensemble method
                    imlist = []
                    for i in range(self_ensemble_n):
                        imlist.append(self.outputs[hr_path][f'self_ensemble_{i}'])
                    if self.config['self_ensemble']:
                        imlist[1] = imlist[1][:, :, :, ::-1]
                        imlist[2] = imlist[2][:, :, ::-1, :]
                        imlist[3] = imlist[3].transpose((0, 1, 3, 2))
                    else:
                        assert len(imlist) == 1
                    sr_img = np.stack(imlist).mean(axis=0)  # NCHW
                    self.outputs.pop(hr_path)   # delete
                    logging.debug(f'postprocess: self.outputs["{hr_path}"]: completed, shape={sr_img.shape}')
                    ## evaluate
                    sr_img = sr_img[0].transpose(1,2,0)     # RGB
                    sr_img = np.clip(sr_img, 0, 255).astype(np.uint8)
                    criterion = self.config['criterion']
                    hr_img = cv2.imread(hr_path)
                    if hr_img.ndim != 3:   # if gray image
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_GRAY2RGB)
                    else:
                        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
                    score = evaluate(sr_img, hr_img, criterion, self.scale)
                    n += 1
                    logging.info(f'- [{n:03d}/{self.num:03d}] '
                                 f'{os.path.split(hr_path)[-1]}: '
                                 f'psnr={score["psnr"]:.4f} db, '
                                 f'ssim={score["ssim"]:.4f}, '
                                 f'lpips={score["lpips"]:.4f}')
                    self.scores[hr_path] = score
                    for k in ['psnr', 'ssim', 'lpips']:
                        if score[k] is None:
                            continue
                        elif score[k] == float('inf'):
                            logging.info(f'  {k} == inf: discarded')
                            continue
                        self.stats[k] += score[k]
                    ## store to disk
                    if self.config['output_dir']:
                        output_dir = self.config['output_dir']
                        imname = os.path.split(hr_path)[-1]
                        imname,ext = os.path.splitext(imname)
                        imname += f'_x{self.scale}'
                        savepath = os.path.join(output_dir, imname+ext)
                        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(savepath, sr_img)
        logging.info(f'****** finished ******')
        for k in ['psnr', 'ssim', 'lpips']:
            self.stats[k] /= self.num
            logging.info(f'ave_{k}: {self.stats[k]:.4f}')

    def join(self):
        for p in self.pre_procs:
            p.join()
        self.q.put(None)
        self.relay.join()
        self.model.put()
        self.post.join()

    def get_stats(self):
        stats = self.stats.copy()
        return stats


def load_lpips():
    global lpips, loss_fn
    lpips = importlib.import_module('lpips')
    loss_fn = lpips.LPIPS(net='alex', version='0.1')    # Initializing the model


def config_logger(logdir=None, logfile='output.log'):
    handlers = []
    chlr = logging.StreamHandler()
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    STREAM_FORMAT = '[%(levelname)s] %(message)s'
    stream_fmt = logging.Formatter(STREAM_FORMAT, datefmt=DATE_FORMAT)
    chlr.setFormatter(stream_fmt)
    chlr.setLevel('INFO')
    handlers.append(chlr)
    if logdir and logfile:
        os.makedirs(logdir, exist_ok=True)
        fhlr = logging.FileHandler(os.path.join(logdir, logfile), mode='w')
        FILE_FORMAT = '[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s'
        file_fmt = logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
        fhlr.setFormatter(file_fmt)
        fhlr.setLevel('DEBUG')
        handlers.append(fhlr)
    logging.basicConfig(force=True, handlers=handlers, level='DEBUG')


@harness('sr')      # register in `_harness_functions`
def harness_main(tree, config, args):
    """ `args` is from `config['harness']['args']`
        for args in config['harness']['args']:
            stats = harness(tree, config, args)
    """
    ds_config = config['dataset']
    lr_dir = tree.expand_variables(config, ds_config['lr_dir'])
    hr_dir = tree.expand_variables(config, ds_config['hr_dir'])
    mean = ds_config.get('mean', [0, 0, 0])
    scale = ds_config.get('scale', [1, 1, 1])
    bgr2rgb = ds_config.get('bgr2rgb', True)
    self_ensemble = ds_config.get('self_ensemble', False)
    criterion = ds_config.get('criterion', ('psnr','ssim','lpips'))
    criterion = [v.lower() for v in criterion]
    if 'lpips' in criterion:
        load_lpips()

    bmodel = tree.expand_variables(config, args['bmodel'])
    overlap = args.get('overlap', (6, 6))
    blend = args.get('blend', False)
    output_dir = tree.expand_variables(config, args.get('output_dir', ''))
    config_logger(output_dir)
    devices = tree.global_config['devices']

    name = args.get('name', '')
    if name:
        logging.info(f'name: {name}')

    pre_config = dict(mean=mean, scale=scale, output_dir=output_dir, 
                      self_ensemble=self_ensemble, overlap=overlap,
                      blend=blend, criterion=criterion, bgr2rgb=bgr2rgb)
    runner = Runner(
        bmodel, devices, lr_dir, hr_dir, pre_config,
        args.get('threads', 2))
    runner.join()
    return runner.get_stats()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='tpu_perf super-resolution harness')
    parser.add_argument('--bmodel', required=True, type=str, help='Bmodel path')
    parser.add_argument('--lr_dir', required=True, type=str, help='Low-resolution image directory')
    parser.add_argument('--hr_dir', required=True, type=str, help='High-resolution image directory')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory. No save if empty.')
    parser.add_argument('--mean', type=str, default='0,0,0', help='Mean value like 128,128,128')
    parser.add_argument('--scale', type=str, default='1,1,1', help='Scale value like 1,1,1')
    parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble method')
    parser.add_argument('--overlap', type=str, default='6,6', help='Overlap width like 6,6')
    parser.add_argument('--blend', action='store_true', help='Blend SR pixels in overlap region')
    parser.add_argument('--criterion', type=str, default='psnr,ssim,lpips', help='Criterion like psnr,ssim,lpips')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--devices', '-d', type=int, nargs='*', help='Devices', default=[0])
    parser.add_argument('--bgr2rgb', type=str, default=True, help="default True to convert image into rgb format")
    args = parser.parse_args()

    config_logger(args.output_dir)

    mean = [float(v) for v in args.mean.split(',')]
    if len(mean) != 3:
        mean = mean[:1] * 3
    scale = [float(v) for v in args.scale.split(',')]
    if len(scale) != 3:
        scale = scale[:1] * 3
    overlap = [int(v) for v in args.overlap.split(',')]
    criterion = [v.lower() for v in args.criterion.split(',')]
    if 'lpips' in criterion:
        load_lpips()
    if len(overlap) != 2:
        overlap = overlap[:1] * 2

    config = dict(mean=mean, scale=scale, output_dir=args.output_dir, 
                  self_ensemble=args.self_ensemble, overlap=overlap,
                  blend=args.blend, criterion=criterion, bgr2rgb=args.bgr2rgb)
    print(config)
    runner = Runner(args.bmodel, args.devices, args.lr_dir, args.hr_dir, config, args.threads)
    runner.join()
    print(runner.get_stats())


if __name__ == '__main__':
    main()


###### run in a shell:

# python sr.py --bmodel ../MSRN_BM168X/models/BM1684X/msrn_x4_1684x_int8_sym.bmodel \
#     --lr_dir /home/linaro/benchmark/Set14/LR_bicubic/X4/ \
#     --hr_dir /home/linaro/benchmark/Set14/HR/ \
#     --output_dir output/ \
#     --criterion psnr,ssim,lpips \


###### configure in *.mlir.config.yaml:

# BM1684X:
#   deploy:
#     ...

#   dataset:
#     lr_dir: ...
#     hr_dir: ...
#     mean: [0, 0, 0]
#     scale: [1, 1, 1]
#     bgr2rgb: true
#     self_ensemble: false
#     criterion: ['psnr', 'ssim', 'lpips']

#   harness:
#     type: sr
#     args:
#       - name: FP32
#         bmodel: $(workdir)/$(name)_1684x_f32.bmodel
#         overlap: [6, 6]
#         blend: false
#         output_dir: $(workdir)/output_images_f32/
#         threads: 2
#       - name: FP16
#         bmodel: $(workdir)/$(name)_1684x_f16.bmodel
#         overlap: [6, 6]
#         blend: false
#         output_dir: $(workdir)/output_images_f16/
#         threads: 2
#       - name: INT8
#         bmodel: $(workdir)/$(name)_1684x_int8_sym.bmodel
#         overlap: [6, 6]
#         blend: false
#         output_dir: $(workdir)/output_images_int8/
#         threads: 2
