import torch
from lib.config import cfg, args
import numpy as np
import os


def run_evaluate():
    import time
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    
    # 网络和模型加载
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    # 数据加载器和评估器
    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    outputs = []

    # 推理过程和时间统计
    tot_elapsed_time = 0.0 # 有效推理所消耗的时间
    tot_valid_cnt = 0 # 有效的样本数

    print('Start inference...')
    with torch.inference_mode():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            inp = batch['inp'].cuda()
            K = batch['K'].cuda()
            x_ini = batch['x_ini'].cuda()
            bbox = batch['bbox'].cuda()
            x2s = batch['x2s'].cuda()
            x4s = batch['x4s'].cuda()
            x8s = batch['x8s'].cuda()
            xfc = batch['xfc'].cuda()
            output, elapsed_time, is_valid = network(inp, K, x_ini, bbox, x2s,x4s, x8s, xfc) 
            # 返回推理所花费的时间（elapsed_time）和是否为有效样本（is_valid）
            
            if is_valid:
                tot_elapsed_time += elapsed_time
                tot_valid_cnt += 1

            outputs.append(output)

    # 计算 ADD(-S) 评估指标
    print('Start computing ADD(-S) metrics...')
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        output = outputs[i]
        evaluator.evaluate(output, batch)

    # 计算并打印 FPS（每秒帧数）
    print('Average FPS:', 1000 / (tot_elapsed_time / tot_valid_cnt))
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        inp = batch['inp'].cuda()
        K = batch['K'].cuda()
        x_ini = batch['x_ini'].cuda()
        bbox = batch['bbox'].cuda()
        x2s = batch['x2s'].cuda()
        x4s = batch['x4s'].cuda()
        x8s = batch['x8s'].cuda()
        xfc = batch['xfc'].cuda()
        with torch.inference_mode():
            output, _, _ = network(inp, K, x_ini, bbox, x2s, x4s, x8s, xfc)
        visualizer.visualize(output, batch, i)


def run_linemod():
    from lib.datasets.linemod import linemod_to_coco
    linemod_to_coco.linemod_to_coco(cfg)


if __name__ == '__main__':
    globals()['run_' + args.type]()
