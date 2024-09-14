import argparse
import copy
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from u2pl.dataset.augmentation import generate_unsup_data
from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.dist_helper import setup_distributed
from u2pl.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
    get_criterion_2,
    compute_contra_memobank_prototype_loss
)
from u2pl.utils.lr_helper import get_optimizer, get_scheduler
from u2pl.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
    queue_high_low_feat_sample,
    queue_back_feat_sample,
    queue_random_feat_sample,
    prototype_KMeans,
    prototype_KMeans_random,
    prototype_KMeans_for_coco,
    dynamic_copy_paste,
    cal_category_confidence,
    class_balance_sample_high_low,
    class_balance_sample_random
)
# from apex import amp
from torch.cuda.amp import autocast as autocast, GradScaler

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)  


def main():
    global args, cfg, prototype  
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)  

    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    cfg["exp_path"] = os.path.dirname(args.config)  
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])  
    cfg['save_path_proto'] = os.path.join(cfg["exp_path"], 'prototype')

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port)  

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(
            osp.join(cfg["exp_path"], "log/events_seg/" + current_time)
        )  
    else:
        tb_logger = None

    if args.seed is not None:  
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:  
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network
    model = ModelBuilder(cfg["net"])  
    
    #     model.decoder.init_params_decoder()
    #     print("-----------------INIT------------------------------")
    modules_back = [model.encoder]  
    if cfg["net"].get("aux_loss", False):  
        modules_head = [model.auxor, model.decoder]
    else:  
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):  
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  

    model.cuda()

    sup_loss_fn = get_criterion(cfg)  
    sup_loss_fn_2 = get_criterion_2(cfg) 

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)  

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]  # SGD
    times = 10 if "pascal" in cfg["dataset"]["type"] or "coco" in cfg["dataset"]["type"] else 1  

    params_list = []
    for module in modules_back:  
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:  
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)  

    
    

    acp = cfg["dataset"].get("acp", False)  

    if acp:  
        global class_criterion
        class_criterion = (
            torch.zeros(3, cfg["net"]["num_classes"]).type(torch.float32).cuda()
        )  

    local_rank = int(os.environ["LOCAL_RANK"])  
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )  

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])  
    model_teacher = model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )  

    for p in model_teacher.parameters():  
        p.requires_grad = False

    global best_prec
    best_prec = 0
    last_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):  
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")  
        if not os.path.exists(lastest_model):  
            "No checkpoint found in '{}'".format(lastest_model)
        else:  
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )  
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )  

    elif cfg["saver"].get("pretrain", False):  
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)  
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )  

    
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(cfg["net"]["num_classes"]):
        memobank.append([torch.zeros(0, 256)])  
        queue_size.append(30000)  
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))  
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros(
        (
            cfg["net"]["num_classes"],
            cfg["trainer"]["contrastive"]["num_queries"],
            1,
            256,
        )
    ).cuda()  

    
    global prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis, prototype_center_list, prototype_center_list_prob, prototype_id_list, prototype_num_list
    if cfg["trainer"].get('prototype', False):
        # print("Sample!!!!")
        prototype_feat_list = []
        prototype_feat_num = []  
        prototype_feat_ptrlis = []  
        if cfg["trainer"]['prototype']['type'] == 'cluster' or cfg["trainer"]['prototype'][
            'type'] == 'prob':  
            for i in range(cfg["net"]["num_classes"]):  
                
                if "coco" in cfg["dataset"]["type"]:
                    prototype_feat_list.append(
                        {'high': torch.zeros(0, 257), 'low': torch.zeros(0, 257), 'back': torch.zeros(0, 257)})
                    prototype_feat_num.append(
                        {'high': cfg["trainer"]['prototype']['queue_len'],
                         'low': cfg["trainer"]['prototype']['queue_len'],
                         'back': cfg["trainer"]['prototype']['queue_len'] * 2})
                    prototype_feat_ptrlis.append(
                        {'high': torch.zeros(1, dtype=torch.long),
                         'low': torch.zeros(1, dtype=torch.long),
                         'back': torch.zeros(1, dtype=torch.long)})

                else:  
                    prototype_feat_list.append(
                        {'high': torch.zeros(0, 257), 'low': torch.zeros(0, 257)})  
                    prototype_feat_num.append(
                        {'high': cfg["trainer"]['prototype']['queue_len'], 'low': cfg["trainer"]['prototype']['queue_len']})
                    prototype_feat_ptrlis.append(
                        {'high': torch.zeros(1, dtype=torch.long), 'low': torch.zeros(1, dtype=torch.long)})

                # prototype_feat_list.append(
                
                # prototype_feat_num.append(
                #     {'high': cfg["trainer"]['prototype']['queue_len'], 'low': cfg["trainer"]['prototype']['queue_len']})
                # prototype_feat_ptrlis.append(
                #     {'high': 30000, 'low': 30000})
        elif cfg["trainer"]['prototype']['type'] == 'random':  
            for i in range(cfg["net"]["num_classes"]):  
                prototype_feat_list.append({'random': torch.zeros(0, 257)})  
                prototype_feat_num.append({'random': cfg["trainer"]['prototype']['queue_len'] * 2})  
                prototype_feat_ptrlis.append({'random': torch.zeros(1, dtype=torch.long)})  

                # prototype_feat_list.append(
                
                # prototype_feat_num.append(
                #     {'random': cfg["trainer"]['prototype']['queue_len'] * 2})
                # prototype_feat_ptrlis.append(
                #     {'random': 60000})

        else:
            raise ValueError
        prototype_center_list, prototype_center_list_prob, prototype_id_list, prototype_num_list = [], [], [], []

    
    global confidence_bank, confidence_sample_ema
    confidence_bank = [[] for i in range(cfg["net"]["num_classes"])]
    confidence_sample_ema = np.ones((cfg["net"]["num_classes"],)) * cfg["trainer"]['prototype'][
        'init_prob']  

    scaler = GradScaler()

    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):  
        # Training
        train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,  
            sup_loss_fn_2,
            train_loader_sup,
            train_loader_unsup,
            val_loader,
            epoch,
            tb_logger,
            logger,
            memobank,
            queue_ptrlis,
            queue_size,
            scaler
        )  

        
        if cfg_trainer["eval_on"]:
            if rank == 0:
                logger.info("start evaluation")

            if epoch < cfg["trainer"].get("sup_only_epoch", 1):  
                prec = validate(model, val_loader, epoch, logger)
            else:  
                prec = validate(model_teacher, val_loader, epoch, logger)

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "best_miou": best_prec,
                }
                if prec > best_prec:
                    best_prec = prec
                    torch.save(
                        state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                    )


                torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], f"ckpt_{epoch}.pth"))

                logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                        best_prec * 100
                    )
                )
                tb_logger.add_scalar("mIoU val", prec, epoch)


def train(
        model,
        model_teacher,
        optimizer,
        lr_scheduler,
        sup_loss_fn,
        sup_loss_fn_2,
        loader_l,
        loader_u,
        val_loader,
        epoch,
        tb_logger,
        logger,
        memobank,
        queue_ptrlis,
        queue_size,
        scaler
):
    global prototype  
    global prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis, prototype_center_list, prototype_center_list_prob, prototype_id_list, prototype_num_list
    global confidence_bank, confidence_sample_ema
    global best_prec
    global class_criterion

    ema_decay_origin = cfg["net"]["ema_decay"]  

    model.train()  

    loader_l.sampler.set_epoch(epoch)  
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)  
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    rank, world_size = dist.get_rank(), dist.get_world_size()  

    acp = cfg["dataset"].get("acp", False)  # True

    if acp:  
        all_cat = [i for i in range(cfg['net']['num_classes'])]  
        ignore_cat = cfg["dataset"]['acp'][
            'ignore_cat']  
        target_cat = list(set(all_cat) - set(ignore_cat))  
        class_momentum = cfg["dataset"]["acp"].get("momentum", 0.999)  # 0.999
        num_cat = cfg["dataset"]["acp"].get("number", 3)  

    if acp:  
        conf = 1 - class_criterion[
            0]  
        conf = conf[target_cat]  
        conf = (conf ** 0.5).cpu().numpy()  
        conf_print = np.exp(conf) / np.sum(np.exp(conf))  
        if rank == 0:
            print("epoch [", epoch, ": ]", "sample_rate_target_class_conf", conf_print)  
            print("epoch [", epoch, ": ]", "criterion_per_class", class_criterion[0])  
            print(
                "epoch [",
                epoch,
                ": ]",
                "sample_rate_per_class_conf",
                (1 - class_criterion[0]) / (torch.max(1 - class_criterion[0]) + 1e-12),
            )  

    sup_losses = AverageMeter(10)  
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    pro_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()  
    for step in range(len(loader_l)):  

        batch_start = time.time()
        data_times.update(batch_start - batch_end)  

        i_iter = epoch * len(loader_l) + step  
        lr = lr_scheduler.get_lr()  
        learning_rates.update(lr[0])  
        lr_scheduler.step()  # lr-step

        if acp:  
            conf = 1 - class_criterion[0]  
            conf = conf[target_cat]  
            conf = (conf ** 0.5).cpu().numpy()  
            conf = np.exp(conf) / np.sum(np.exp(conf))  
            query_cat = []
            for rc_idx in range(num_cat):  
                query_cat.append(np.random.choice(target_cat, p=conf))  
            query_cat = list(set(query_cat))  

        image_l, label_l = loader_l_iter.next()  # labeled data  bchw bhw
        batch_size, _, h, w = image_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()
        # print(image_l.shape, label_l.shape)
        if acp:
            
            if epoch < cfg['dataset']['acp']['delay_epoch']:  
                select_prob = 0
            elif epoch >= cfg['dataset']['acp']['delay_epoch'] and epoch < cfg['dataset']['acp']['delay_epoch'] + \
                    cfg['dataset']['acp']['promote_epoch']:  # >=2 <10
                select_prob = (i_iter - cfg['dataset']['acp']['delay_epoch'] * len(loader_l)) / (
                        cfg['dataset']['acp']['promote_epoch'] * len(loader_l))
                select_prob = 1 if select_prob > 1 else select_prob
            else:
                select_prob = 1

            image_l, label_l = dynamic_copy_paste(
                image_l, label_l, query_cat, select_prob=select_prob
            )  
        # print(image_l.shape, label_l.shape)

        image_u, image_u_s, _ = loader_u_iter.next()  
        image_u, image_u_s = image_u.cuda(), image_u_s.cuda()

        
        if i_iter % 10 == 0 and rank == 0:
            confidence_print = []
            for cla_list in confidence_bank:
                if cla_list:
                    confidence_print.append(np.mean(np.array(cla_list)))
                else:
                    confidence_print.append(0)
            # print(f"Confidence: {confidence_print}")
            # print(f"Confidence_ema: {list(confidence_sample_ema)}")

            if acp:  
                conf = 1 - class_criterion[
                    0]  
                conf = conf[target_cat]  
                conf = (conf ** 0.5).cpu().numpy()  
                conf_print = np.exp(conf) / np.sum(np.exp(conf))  
                if rank == 0:
                    print("epoch [", epoch, ": ]", "sample_rate_target_class_conf", conf_print)  
                    print("epoch [", epoch, ": ]", "criterion_per_class", class_criterion[0])  
                    print(
                        "epoch [",
                        epoch,
                        ": ]",
                        "sample_rate_per_class_conf",
                        (1 - class_criterion[0]) / (torch.max(1 - class_criterion[0]) + 1e-12),
                    )  

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):  
            contra_flag = "none"  
            
            with autocast():
                outs = model(image_l)
                pred, rep = outs["pred"], outs["rep"]
                pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)  
                # print(pred.shape, label_l.shape)
                # supervised loss
                if "aux_loss" in cfg["net"].keys():  # cityscapes
                    aux = outs["aux"]
                    aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)  
                    sup_loss = sup_loss_fn([pred, aux], label_l) * cfg['criterion']['weight'] + sup_loss_fn_2([pred, aux], label_l) * cfg['criterion_2']['weight']
                else:  
                    sup_loss = sup_loss_fn(pred, label_l) * cfg['criterion']['weight'] + sup_loss_fn_2(pred, label_l) * cfg['criterion_2']['weight']

                model_teacher.train()
                _ = model_teacher(image_l)  

                unsup_loss = 0 * rep.sum()  
                contra_loss = 0 * rep.sum()
                # print(unsup_loss, contra_loss)

                
                with torch.no_grad():
                    prob = torch.softmax(pred.detach().clone(), dim=1)  # bchw
                    prob = prob.permute(0, 2, 3, 1)  # bhwc
                    
                    cla_tmp = np.unique(label_l[label_l != 255].detach().clone().cpu().numpy()).astype(int)
                    for cl in cla_tmp:  
                        mask = label_l.clone() == cl  # bhw
                        cl_conf = torch.mean(prob[mask][:, cl]).item()  
                        confidence_bank[cl].append(cl_conf)
                        if len(confidence_bank[cl]) > 10:
                            confidence_bank[cl] = confidence_bank[cl][-10:]

                        if epoch >= cfg["trainer"]['prototype']['sample_delay']:  
                            confidence_sample_ema[cl] = confidence_sample_ema[cl] * cfg["trainer"]['prototype'][
                                'thre_update'] + (1 - cfg["trainer"]['prototype']['thre_update']) * cl_conf

                    
                    if acp and epoch >= cfg['dataset']['acp']['delay_epoch']:
                        category_entropy = cal_category_confidence(
                            pred.detach().clone(),  
                            
                            label_l.clone(),  
                            
                            cfg['net']['num_classes']
                        )  
                        
                        
                        # perform momentum update
                        category_entropy = category_entropy.cuda()
                        for cla in range(cfg['net']['num_classes']):  
                            if category_entropy[cla] > 0:  
                                if class_criterion[0][cla] == 0:  
                                    class_criterion[0][cla] = category_entropy[cla]
                                else:
                                    class_criterion[0][cla] = class_criterion[0][cla] * class_momentum + (
                                            1 - class_momentum) * category_entropy[cla]

            
            if cfg["trainer"].get('prototype', False):  
                prototype_dict = cfg["trainer"]['prototype']
                if epoch < prototype_dict['sample_delay']:
                    with autocast():
                        loss_proto = 0 * rep.sum()

                elif epoch >= prototype_dict['sample_delay'] and epoch < prototype_dict['sample_delay'] + \
                        prototype_dict['sample_last']:  
                    with torch.no_grad():
                        prob = torch.softmax(outs['pred'], dim=1)  
                        pred_logits, pred_label = torch.max(prob, dim=1)  
                        
                        rep = rep  
                        # print(rep.max(), rep.min(), rep.type())

                        
                        rep_logits_label = torch.cat(
                            (rep, pred_logits.clone().unsqueeze(1), pred_label.clone().unsqueeze(1)),
                            dim=1)  

                        
                        if prototype_dict['threshold_type'] == 'fix':  

                            
                            # print(label_l.shape)
                            label_l_small_for_sample = (
                                F.interpolate(label_l.clone().unsqueeze(1).float(), size=prob.shape[2:],
                                              mode="nearest")).squeeze(1)  
                            
                            if "coco" in cfg["dataset"]["type"]:
                                
                                label_l_for_coco = label_l_small_for_sample.clone().long().unsqueeze(3)  # bhw1
                                one_z = torch.zeros(label_l_for_coco.shape[0], label_l_for_coco.shape[1],
                                                    label_l_for_coco.shape[2], cfg['net']['num_classes']).cuda()  
                                one_hot_label = one_z.scatter(3, label_l_for_coco, 1)  
                                one_hot_label = one_hot_label.permute(0, 3, 1, 2)  # b 81 hw
                                prob_true_class = torch.max(one_hot_label * prob, dim=1)[0]  
                                rep_logits_label_for_coco = torch.cat((rep, prob_true_class.clone().unsqueeze(1),
                                                                       label_l_small_for_sample.clone().unsqueeze(1)), dim=1)  # b 258 h w
                                rep_logits_label_for_coco = rep_logits_label_for_coco.permute(0, 2, 3, 1)  # b h w 258
                                
                                sample_for_coco = []
                                tmp_class_list_for_coco = np.unique(label_l_small_for_sample.clone().cpu().numpy()).astype(int)
                                for cc in tmp_class_list_for_coco:
                                    mm = rep_logits_label_for_coco[:, :, :, -1] == cc  
                                    
                                    tmp_sample = rep_logits_label_for_coco[mm]  # num 258
                                    
                                    if tmp_sample.shape[0] > 5000:
                                        tmp_id = list(np.arange(tmp_sample.shape[0]))
                                        random.shuffle(tmp_id)
                                        tmp_id = tmp_id[:5000]
                                        sample_for_coco.append(tmp_sample[tmp_id])
                                    else:
                                        sample_for_coco.append(tmp_sample)
                                sample_for_coco = torch.cat(sample_for_coco, dim=0)  # num 258
                                
                                queue_back_feat_sample(sample_for_coco, prototype_feat_list, prototype_feat_num,
                                                       prototype_feat_ptrlis)

                            high_threshold = prototype_dict['high_threshold']
                            low_threshold = prototype_dict.get('low_threshold', 0)
                            choice_mask = label_l_small_for_sample == pred_label  
                            high_feat_mask = choice_mask.clone()
                            low_feat_mask = choice_mask.clone()

                            high_feat_mask[pred_logits < high_threshold] = False  

                            low_feat_mask[pred_logits >= high_threshold] = False  
                            low_feat_mask[pred_logits <= low_threshold] = False

                            
                            # bchw->bhwc
                            rep_logits_label = rep_logits_label.permute(0, 2, 3, 1)  # bhwc
                            high_feat = rep_logits_label[torch.where(high_feat_mask == True)]  # num * 258
                            low_feat = rep_logits_label[torch.where(low_feat_mask == True)]  # num * 258
                            # print(high_feat.shape, low_feat.shape)
                            # print("!!!!!")
                        elif prototype_dict['threshold_type'] == 'csp':  
                            label_l_small_for_sample = (
                                F.interpolate(label_l.clone().unsqueeze(1).float(), size=prob.shape[2:],
                                              mode="nearest")).squeeze(1)  
                            
                            choice_mask = label_l_small_for_sample == pred_label  
                            rep_logits_label = rep_logits_label.permute(0, 2, 3, 1)  # b h w 258
                            rep_logits_label_choose = rep_logits_label[choice_mask]  # num 258
                            
                            tmp_cla_list = np.unique(rep_logits_label_choose[:, -1].detach().cpu().numpy()).astype(
                                int)  
                            high_feat, low_feat = [], []
                            for cla in tmp_cla_list:  
                                rep_cla = rep_logits_label_choose[rep_logits_label_choose[:, -1] == cla]  # num 258
                                conf_thre_cla = min(max(confidence_sample_ema[cla], prototype_dict['thre_min']),
                                                    prototype_dict['thre_max']) if cla not in prototype_dict[
                                    'fix_class'] else prototype_dict['high_threshold']  
                                high_feat_cla_mask = rep_cla[:, -2] >= conf_thre_cla
                                low_feat_cla_mask = rep_cla[:, -2] < conf_thre_cla
                                high_feat_cla = rep_cla[high_feat_cla_mask]  
                                low_feat_cla = rep_cla[low_feat_cla_mask]
                                high_feat.append(high_feat_cla)
                                low_feat.append(low_feat_cla)
                            high_feat = torch.cat(high_feat, dim=0)  # num 258
                            low_feat = torch.cat(low_feat, dim=0)  # num 258

                        if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'prob':
                            sample_pixel_num = prototype_dict['sample_pixel_num'] * batch_size
                            if not prototype_dict['class_balance_sample']:  
                                

                                if high_feat.shape[0] > sample_pixel_num:
                                    high_id_list = list(np.arange(high_feat.shape[0]))
                                    random.shuffle(high_id_list)
                                    high_id_list = high_id_list[:sample_pixel_num]
                                    high_feat = high_feat[high_id_list, :]  # num 258
                                else:
                                    high_feat = high_feat

                                
                                if low_feat.shape[0] > sample_pixel_num:
                                    low_id_list = list(np.arange(low_feat.shape[0]))
                                    random.shuffle(low_id_list)
                                    low_id_list = low_id_list[:sample_pixel_num]
                                    low_feat = low_feat[low_id_list, :]  # num 258
                                else:
                                    low_feat = low_feat
                            else:  
                                high_feat, low_feat = class_balance_sample_high_low(high_feat, low_feat,
                                                                                    sample_pixel_num=sample_pixel_num,
                                                                                    gamma=prototype_dict['gamma'])

                            queue_high_low_feat_sample(high_feat, low_feat, prototype_feat_list, prototype_feat_num,
                                                       prototype_feat_ptrlis)

                        elif prototype_dict['type'] == 'random':
                            all_feat = torch.cat((high_feat, low_feat), dim=0)  
                            sample_pixel_num = prototype_dict['sample_pixel_num'] * batch_size * 2
                            if not prototype_dict['class_balance_sample']:  
                                if all_feat.shape[0] > sample_pixel_num:
                                    all_id_list = list(np.arange(all_feat.shape[0]))
                                    random.shuffle(all_id_list)
                                    all_id_list = all_id_list[:sample_pixel_num]
                                    all_feat = all_feat[all_id_list, :]  # num 258
                                else:
                                    all_feat = all_feat
                            else:  
                                all_feat = class_balance_sample_random(all_feat,
                                                                       sample_pixel_num=sample_pixel_num,
                                                                       gamma=prototype_dict['gamma'])

                            queue_random_feat_sample(all_feat, prototype_feat_list, prototype_feat_num,
                                                     prototype_feat_ptrlis)

                    
                    with autocast():
                        loss_proto = 0 * rep.sum()
                    # print(loss_proto)

                else:  
                    if epoch == prototype_dict['sample_delay'] + prototype_dict[
                        'sample_last'] and step == 0:  
                        
                        if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'prob':
                            if "coco" in cfg["dataset"]["type"]:
                                prototype_center_list, prototype_center_list_prob, prototype_id_list, prototype_num_list = prototype_KMeans_for_coco(
                                    prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis,
                                    prototype_dict['high_prototype_num'],
                                    prototype_dict['low_prototype_num'],
                                    cfg
                                )  
                            else:
                                prototype_center_list, prototype_center_list_prob, prototype_id_list, prototype_num_list = prototype_KMeans(
                                    prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis,
                                    prototype_dict['high_prototype_num'],
                                    prototype_dict['low_prototype_num'],
                                    cfg
                                )  
                            if rank == 0:
                                print(prototype_num_list)
                        elif prototype_dict['type'] == 'random':  
                            prototype_center_list, prototype_center_list_prob, prototype_id_list = prototype_KMeans_random(
                                prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis,
                                prototype_dict['high_prototype_num'] + prototype_dict['low_prototype_num'])

                        
                        del prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis
                    proto_list = prototype_center_list if prototype_dict['type'] == 'cluster' or prototype_dict[
                        'type'] == 'random' else prototype_center_list_prob
                    
                    with autocast():
                        rep_large = F.interpolate(rep, (h, w), mode="bilinear",
                                                  align_corners=True)  

                        
                        if prototype_dict['sup_sample'] == 'grid':
                            interval = (h - 1) // prototype_dict['grid_num']  
                            sample_id = [i * interval for i in range(prototype_dict['grid_num'])] + [h - 1]  
                            mask = torch.zeros((h, w))
                            mask[np.ix_(sample_id, sample_id)] = 1  
                            mask = mask.bool()  
                            
                            rep_large = rep_large.permute(0, 2, 3, 1)  
                            rep_sample = rep_large[:, mask]  # b 33*33 c
                            rep_sample_cal = rep_sample.view(-1, rep_sample.shape[-1])  # b*33*33 256
                            
                            label_l_sample = label_l.clone()[:, mask].view(-1)  # bhw-》b*33*33

                        elif prototype_dict['sup_sample'] == 'random':
                            pass

                        
                        with torch.no_grad():
                            
                            high_proto_max, low_proto_max = prototype_dict['high_prototype_num'], prototype_dict['low_prototype_num']
                            if prototype_dict['proto_add']:  
                                high_proto_max += prototype_dict['add_num']
                                low_proto_max += prototype_dict['add_num']

                            proto_tensor = []  
                            for cla in range(len(proto_list)):
                                tmp_cla = proto_list[cla]
                                if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'prob':
                                    tmp_cla_high_pro, tmp_cla_low_pro = tmp_cla['high'], tmp_cla['low']

                                    if tmp_cla_high_pro.shape[0] < high_proto_max:  
                                        lack_num = high_proto_max - tmp_cla_high_pro.shape[0]
                                        for i in range(lack_num):  
                                            tmp_cla_high_pro = torch.cat((tmp_cla_high_pro, (tmp_cla_high_pro[-1]).unsqueeze(0)), dim=0)  # max 256

                                    if tmp_cla_low_pro.shape[0] < low_proto_max:  
                                        lack_num = low_proto_max - tmp_cla_low_pro.shape[0]
                                        for i in range(lack_num):  
                                            tmp_cla_low_pro = torch.cat((tmp_cla_low_pro, (tmp_cla_low_pro[-1]).unsqueeze(0)), dim=0)  # max 256

                                    tmp_proto = torch.cat((tmp_cla_high_pro, tmp_cla_low_pro), dim=0)  
                                elif prototype_dict['type'] == 'random':  
                                    tmp_proto = tmp_cla['random']  # 6 256
                                proto_tensor.append(tmp_proto)
                            proto_tensor = torch.stack(proto_tensor, dim=0)  # 21 highmax+lowmax 256
                            proto_tensor_cal = proto_tensor.clone().view(-1, proto_tensor.shape[-1])  # 21*(highmax+lowmax) 256

                        
                        cos_sim = F.cosine_similarity(rep_sample_cal.unsqueeze(1), proto_tensor_cal.unsqueeze(0),
                                                      dim=2)  
                        cos_sim = cos_sim.view(-1, len(proto_list),
                                               int(high_proto_max) + int(low_proto_max))  # b*33*33 21 (highmax+lowmax)
                        
                        max_sim, max_sim_id = torch.max(cos_sim, dim=2)  

                        
                        _, pro_label_l_sample = torch.max(max_sim, dim=1)  # b*33*33
                        fault_mask = ~(pro_label_l_sample == label_l_sample)  
                        weight_mask = torch.ones_like(fault_mask)
                        weight_mask[fault_mask] = prototype_dict['fault_weight']  
                        

                        loss_proto = F.cross_entropy(
                            max_sim / prototype_dict['cosine_tao'], label_l_sample.long().cuda(),
                            ignore_index=cfg["dataset"]["ignore_label"])  

                    
                    with torch.no_grad():
                        ll = list(np.unique(label_l_sample.detach().cpu().numpy()))  
                        for cla in range(len(proto_list)):  
                            if cla not in ll:
                                continue
                            
                            cla_mask = label_l_sample == cla  
                            feature = rep_sample_cal[cla_mask]  
                            max_sim_id_cla = max_sim_id[cla_mask, cla]  
                            
                            lack_num_high = int(high_proto_max - prototype_num_list['high'][cla])  
                            lack_num_low = int(low_proto_max - prototype_num_list['low'][cla])

                            clone_pos_tmp = max_sim_id_cla.clone()  
                            high_pro_mask_tmp = clone_pos_tmp < high_proto_max  
                            low_pro_mask_tmp = clone_pos_tmp >= high_proto_max  
                            
                            if lack_num_high > 0:  
                                
                                tmp_mask = (high_pro_mask_tmp * (max_sim_id_cla >= prototype_num_list['high'][cla])).bool()
                                max_sim_id_cla[tmp_mask] = int(prototype_num_list['high'][cla] - 1)

                            
                            if lack_num_low > 0:
                                max_sim_id_cla[(low_pro_mask_tmp * (max_sim_id_cla >= (high_proto_max + prototype_num_list['low'][cla]))).bool()] = int(high_proto_max + prototype_num_list['low'][cla] - 1)

                            if lack_num_high > 0:  
                                max_sim_id_cla[low_pro_mask_tmp] -= lack_num_high

                            weight_mask_cla = weight_mask[cla_mask]  
                            for p in range(int(prototype_num_list['high'][cla] + prototype_num_list['low'][cla])):  
                                
                                feat_p = feature[max_sim_id_cla == p]  
                                weight_mask_p = weight_mask_cla[max_sim_id_cla == p]
                                if feat_p.shape[0] > 0:  
                                    weight_mask_p = (weight_mask_p / torch.sum(weight_mask_p)).unsqueeze(1)  
                                    feat_p = torch.sum(feat_p * weight_mask_p, dim=0)  
                                    
                                else:
                                    continue
                                if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'prob':
                                    if p < int(prototype_num_list['high'][cla]):  # high
                                        
                                        #     proto_list[cla]['high'][p] = (1 - prototype_dict['prototype_ema_decay']) * fe + \
                                        #                                  prototype_dict['prototype_ema_decay'] * proto_list[cla]['high'][p]
                                        
                                        proto_list[cla]['high'][p] = (1 - prototype_dict[
                                            'prototype_ema_decay']) * feat_p + \
                                                                     prototype_dict['prototype_ema_decay'] * \
                                                                     proto_list[cla]['high'][p]

                                    else:  # low
                                        m_id = int(p - prototype_num_list['high'][cla])
                                        
                                        # proto_list[cla]['low'][m_id] = (1 - prototype_dict['prototype_ema_decay']) * fe + \
                                        #                              prototype_dict['prototype_ema_decay'] * proto_list[cla]['low'][m_id]
                                        proto_list[cla]['low'][m_id] = (1 - prototype_dict[
                                            'prototype_ema_decay']) * feat_p + \
                                                                       prototype_dict['prototype_ema_decay'] * \
                                                                       proto_list[cla]['low'][m_id]

                                elif prototype_dict['type'] == 'random':  
                                    proto_list[cla]['random'][p] = (1 - prototype_dict[
                                        'prototype_ema_decay']) * feat_p + \
                                                                   prototype_dict['prototype_ema_decay'] * \
                                                                   proto_list[cla]['random'][p]
                        if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'random':  
                            prototype_center_list = proto_list
                        else:
                            prototype_center_list_prob = proto_list
            else:
                with autocast():
                    loss_proto = 0 * rep.sum()

        else:  
            # print("semi!!!")
            if epoch == cfg["trainer"].get("sup_only_epoch",
                                           1) and step == 0:  
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                            model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data  

            
            with autocast():
                with torch.no_grad():
                    model_teacher.eval()
                    pred_u_teacher, rep_u_t = model_teacher(image_u)["pred"], model_teacher(image_u)["rep"]
                    pred_u_t = pred_u_teacher.clone()
                    pred_u_teacher = F.interpolate(
                        pred_u_teacher, (h, w), mode="bilinear", align_corners=True
                    )  
                    pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
                    logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)  
                    

                
                if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                        "apply_aug", False
                ):  
                    image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                        image_u_s,
                        label_u_aug.clone(),
                        logits_u_aug.clone(),
                        mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                    )  
                    
                else:
                    image_u_aug = image_u_s

                # forward
                num_labeled = len(image_l)  # batch
                image_all = torch.cat((image_l, image_u_aug))  
                outs = model(image_all)
                pred_all, rep_all = outs["pred"], outs["rep"]  
                pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]  
                pred_l_large = F.interpolate(
                    pred_l, size=(h, w), mode="bilinear", align_corners=True
                )
                pred_u_large = F.interpolate(
                    pred_u, size=(h, w), mode="bilinear", align_corners=True
                )  

                
                if "aux_loss" in cfg["net"].keys():
                    aux = outs["aux"][:num_labeled]
                    aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                    sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone()) * cfg['criterion']['weight'] + sup_loss_fn_2([pred_l_large, aux], label_l.clone()) * cfg['criterion_2']['weight']
                else:
                    sup_loss = sup_loss_fn(pred_l_large, label_l.clone()) * cfg['criterion']['weight'] + sup_loss_fn_2(pred_l_large, label_l.clone()) * cfg['criterion_2']['weight']

                with torch.no_grad():  
                    prob_l_tmp = torch.softmax(pred_l_large.detach().clone(), dim=1)  # bchw
                    prob_l_tmp = prob_l_tmp.permute(0, 2, 3, 1)  # bhwc
                    
                    cla_tmp = np.unique(label_l[label_l != 255].detach().clone().cpu().numpy()).astype(int)
                    for cl in cla_tmp:  
                        mask = label_l.clone() == cl  # bhw
                        cl_conf = torch.mean(prob_l_tmp[mask][:, cl]).item()  
                        confidence_bank[cl].append(cl_conf)
                        if len(confidence_bank[cl]) > 10:
                            confidence_bank[cl] = confidence_bank[cl][-10:]
                        if epoch >= cfg["trainer"]['prototype']['sample_delay']:  
                            confidence_sample_ema[cl] = confidence_sample_ema[cl] * cfg["trainer"]['prototype'][
                                'thre_update'] + (1 - cfg["trainer"]['prototype']['thre_update']) * cl_conf

                        
                    if acp and epoch >= cfg['dataset']['acp']['delay_epoch']:
                        category_entropy = cal_category_confidence(
                            pred_l_large.detach().clone(),  
                            
                            label_l.clone(),  
                            
                            cfg['net']['num_classes']
                        )  
                        
                        
                        # perform momentum update
                        category_entropy = category_entropy.cuda()
                        for cla in range(cfg['net']['num_classes']):  
                            if category_entropy[cla] > 0:  
                                if class_criterion[0][cla] == 0:  
                                    class_criterion[0][cla] = category_entropy[cla]
                                else:
                                    class_criterion[0][cla] = class_criterion[0][cla] * class_momentum + (
                                            1 - class_momentum) * category_entropy[cla]

                
                model_teacher.train()  
                with torch.no_grad():
                    out_t = model_teacher(image_all)
                    pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                    prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                    prob_l_teacher, prob_u_teacher = (
                        prob_all_teacher[:num_labeled],
                        prob_all_teacher[num_labeled:],
                    )

                    pred_u_teacher = pred_all_teacher[num_labeled:]
                    pred_u_large_teacher = F.interpolate(
                        pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                    )

                
                drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)  
                percent_unreliable = (100 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])  
                drop_percent = 100 - percent_unreliable
                unsup_loss = (
                        compute_unsupervised_loss(
                            pred_u_large,  
                            label_u_aug.clone(),  
                            drop_percent,  
                            pred_u_large_teacher.detach(),  
                        )
                        * cfg["trainer"]["unsupervised"].get("loss_weight", 1)
                )  

            
            if cfg["trainer"].get('prototype', False):  
                prototype_dict = cfg["trainer"]['prototype']
                if epoch < prototype_dict['sample_delay']:
                    with autocast():
                        loss_proto = 0 * rep_all.sum()

                elif epoch >= prototype_dict['sample_delay'] and epoch < prototype_dict['sample_delay'] + \
                        prototype_dict['sample_last']:  
                    with torch.no_grad():
                        
                        prob_l = torch.softmax(pred_l, dim=1)  
                        pred_l_logits, pred_l_label = torch.max(prob_l, dim=1)  
                        rep_l = rep_all[:num_labeled]  
                        
                        rep_logits_label_l = torch.cat(
                            (rep_l, pred_l_logits.clone().unsqueeze(1), pred_l_label.clone().unsqueeze(1)),
                            dim=1)  

                        
                        if prototype_dict['threshold_type'] == 'fix':  
                            
                            gt_label_l_small = (
                                F.interpolate(label_l.clone().unsqueeze(1).float(), size=prob_l.shape[2:],
                                              mode="nearest")).squeeze(
                                1)  

                            
                            if "coco" in cfg["dataset"]["type"]:
                                
                                gt_label_l_for_coco = gt_label_l_small.clone().long().unsqueeze(3)  # bhw1
                                one_z = torch.zeros(gt_label_l_for_coco.shape[0], gt_label_l_for_coco.shape[1],
                                                    gt_label_l_for_coco.shape[2], cfg['net']['num_classes']).cuda()  
                                one_hot_label_l = one_z.scatter(3, gt_label_l_for_coco, 1)  
                                one_hot_label_l = one_hot_label_l.permute(0, 3, 1, 2)  # b 81 hw
                                prob_l_true_class = torch.max(one_hot_label_l * prob_l, dim=1)[
                                    0]  
                                rep_logits_label_l_for_coco = torch.cat((rep_l, prob_l_true_class.clone().unsqueeze(1),
                                                                       gt_label_l_small.clone().unsqueeze(1)),
                                                                      dim=1)  # b 258 h w
                                rep_logits_label_l_for_coco = rep_logits_label_l_for_coco.permute(0, 2, 3, 1)  # b h w 258
                                
                                sample_for_coco = []
                                tmp_class_list_for_coco = np.unique(
                                    gt_label_l_small.clone().cpu().numpy()).astype(int)
                                for cc in tmp_class_list_for_coco:
                                    mm = rep_logits_label_l_for_coco[:, :, :, -1] == cc  
                                    
                                    tmp_sample = rep_logits_label_l_for_coco[mm]  # num 258
                                    
                                    if tmp_sample.shape[0] > 5000:
                                        tmp_id = list(np.arange(tmp_sample.shape[0]))
                                        random.shuffle(tmp_id)
                                        tmp_id = tmp_id[:5000]
                                        sample_for_coco.append(tmp_sample[tmp_id])
                                    else:
                                        sample_for_coco.append(tmp_sample)
                                sample_for_coco = torch.cat(sample_for_coco, dim=0)  # num 258
                                
                                queue_back_feat_sample(sample_for_coco, prototype_feat_list, prototype_feat_num,
                                                       prototype_feat_ptrlis)

                            high_threshold = prototype_dict['high_threshold']
                            low_threshold = prototype_dict.get('low_threshold', 0)
                            choice_mask_l = gt_label_l_small == pred_l_label  
                            high_feat_mask_l = choice_mask_l.clone()
                            low_feat_mask_l = choice_mask_l.clone()

                            high_feat_mask_l[pred_l_logits < high_threshold] = False  

                            low_feat_mask_l[pred_l_logits >= high_threshold] = False  
                            low_feat_mask_l[pred_l_logits <= low_threshold] = False

                            
                            # bchw->bhwc
                            rep_logits_label_l = rep_logits_label_l.permute(0, 2, 3, 1)  # bhwc
                            high_feat_l = rep_logits_label_l[torch.where(high_feat_mask_l == True)]  # num * 258
                            low_feat_l = rep_logits_label_l[torch.where(low_feat_mask_l == True)]  # num * 258

                        elif prototype_dict['threshold_type'] == 'csp':  
                            gt_label_l_small = (
                                F.interpolate(label_l.clone().unsqueeze(1).float(), size=prob_l.shape[2:],
                                              mode="nearest")).squeeze(
                                1)  
                            
                            choice_mask_l = gt_label_l_small == pred_l_label  
                            rep_logits_label_l = rep_logits_label_l.permute(0, 2, 3, 1)  # b h w 258
                            rep_logits_label_l_choose = rep_logits_label_l[choice_mask_l]  
                            
                            tmp_cla_l_list = np.unique(rep_logits_label_l_choose[:, -1].detach().cpu().numpy()).astype(
                                int)  
                            high_feat_l, low_feat_l = [], []
                            for cla in tmp_cla_l_list:  
                                rep_l_cla = rep_logits_label_l_choose[
                                    rep_logits_label_l_choose[:, -1] == cla]  # num 258
                                conf_thre_l_cla = min(max(confidence_sample_ema[cla], prototype_dict['thre_min']),
                                                      prototype_dict['thre_max']) if cla not in prototype_dict[
                                    'fix_class'] else prototype_dict['high_threshold']  
                                high_feat_l_cla_mask = rep_l_cla[:, -2] >= conf_thre_l_cla
                                low_feat_l_cla_mask = rep_l_cla[:, -2] < conf_thre_l_cla
                                high_feat_l_cla = rep_l_cla[high_feat_l_cla_mask]  
                                low_feat_l_cla = rep_l_cla[low_feat_l_cla_mask]
                                high_feat_l.append(high_feat_l_cla)
                                low_feat_l.append(low_feat_l_cla)
                            high_feat_l = torch.cat(high_feat_l, dim=0)  # num 258
                            low_feat_l = torch.cat(low_feat_l, dim=0)  # num 258

                        random_feat_l = torch.cat((high_feat_l, low_feat_l), dim=0)
                        # print(high_feat.shape, low_feat.shape)
                        # print("!!!!!")

                        

                        
                        prob_u_t = F.softmax(pred_u_t, dim=1)  # bchw
                        pred_u_logits, pred_u_label = torch.max(prob_u_t, dim=1)  
                        rep_u_t = rep_u_t
                        
                        rep_logits_label_u = torch.cat(
                            (rep_u_t, pred_u_logits.clone().unsqueeze(1), pred_u_label.clone().unsqueeze(1)),
                            dim=1)  

                        
                        high_threshold = prototype_dict['pseudo_threshold']
                        high_feat_mask_u = pred_u_logits >= high_threshold  # bhw
                        rep_logits_label_u = rep_logits_label_u.permute(0, 2, 3, 1)  # bhwc
                        high_feat_u = rep_logits_label_u[torch.where(high_feat_mask_u == True)]  # num * 258

                        # print(high_feat.shape, low_feat.shape)
                        # print("!!!!!")

                        
                        low_feat_all = low_feat_l
                        high_feat_all = torch.cat((high_feat_l, high_feat_u), dim=0)
                        random_list = list(np.arange(int(high_feat_all.shape[0])))
                        random.shuffle(random_list)
                        high_feat_all = high_feat_all[random_list, :]  

                        random_feat_all = torch.cat((random_feat_l, high_feat_u), dim=0)  
                        if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'prob':
                            

                            high_sample_pixel_num = prototype_dict['sample_pixel_num'] * pred_all.shape[0]
                            low_sample_pixel_num = prototype_dict[
                                                       'sample_pixel_num'] * num_labeled  
                            if not prototype_dict['class_balance_sample']:  
                                
                                if high_feat_all.shape[0] > high_sample_pixel_num:
                                    high_feat_all = high_feat_all[:high_sample_pixel_num]
                                else:
                                    high_feat_all = high_feat_all

                                
                                if low_feat_all.shape[0] > low_sample_pixel_num:
                                    low_id_list = list(np.arange(low_feat_all.shape[0]))
                                    random.shuffle(low_id_list)
                                    low_id_list = low_id_list[:low_sample_pixel_num]
                                    low_feat_all = low_feat_all[low_id_list, :]  # num 258
                                else:
                                    low_feat_all = low_feat_all
                            else:
                                
                                high_feat_l, low_feat_l = class_balance_sample_high_low(high_feat_l, low_feat_l,
                                                                                        sample_pixel_num=prototype_dict[
                                                                                                             'sample_pixel_num'] * num_labeled,
                                                                                        gamma=prototype_dict['gamma'])
                                high_feat_u = class_balance_sample_random(high_feat_u,
                                                                          sample_pixel_num=prototype_dict[
                                                                                               'sample_pixel_num'] * num_labeled,
                                                                          gamma=prototype_dict['gamma'])
                                
                                high_feat_all = torch.cat((high_feat_l, high_feat_u))
                                low_feat_all = low_feat_l

                            queue_high_low_feat_sample(high_feat_all, low_feat_all, prototype_feat_list,
                                                       prototype_feat_num,
                                                       prototype_feat_ptrlis)

                            # if i_iter % 10 == 0 and rank == 0:
                            #     if "coco" in cfg["dataset"]["type"]:
                            #         for cla in range(cfg["net"]["num_classes"]):
                            #             print(
                            #                 f"{cla}:  high:{prototype_feat_ptrlis[cla]['high']}  low:{prototype_feat_ptrlis[cla]['low']}  back:{prototype_feat_ptrlis[cla]['back']}")
                            #     else:
                            #         for cla in range(cfg["net"]["num_classes"]):
                            #             print(
                            #                 f"{cla}:  high:{prototype_feat_ptrlis[cla]['high']}  low:{prototype_feat_ptrlis[cla]['low']}")

                        elif prototype_dict['type'] == 'random':
                            random_sample_pixel_num = prototype_dict['sample_pixel_num'] * pred_all.shape[0] + \
                                                      prototype_dict['sample_pixel_num'] * num_labeled  
                            if not prototype_dict['class_balance_sample']:  
                                if random_feat_all.shape[0] > random_sample_pixel_num:
                                    random_id_list = list(np.arange(random_feat_all.shape[0]))
                                    random.shuffle(random_id_list)
                                    random_id_list = random_id_list[:random_sample_pixel_num]
                                    random_feat_all = random_feat_all[random_id_list, :]  # num 258
                                else:
                                    random_feat_all = random_feat_all
                            else:
                                random_feat_all = class_balance_sample_random(random_feat_all,
                                                                              sample_pixel_num=random_sample_pixel_num,
                                                                              gamma=prototype_dict['gamma'])

                            queue_random_feat_sample(random_feat_all, prototype_feat_list,
                                                     prototype_feat_num,
                                                     prototype_feat_ptrlis)
                            # if i_iter % 10 == 0 and rank == 0:
                            #     for cla in range(cfg["net"]["num_classes"]):
                            #         print(
                            #             f"{cla}:  random:{prototype_feat_ptrlis[cla]['random']}")

                    with autocast():
                        loss_proto = 0 * rep_all.sum()

                    
                else:  
                    # start_time = time.time()
                    if epoch == prototype_dict['sample_delay'] + prototype_dict[
                        'sample_last'] and step == 0:  
                        
                        if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'prob':
                            if "coco" in cfg["dataset"]["type"]:
                                prototype_center_list, prototype_center_list_prob, prototype_id_list, prototype_num_list = prototype_KMeans_for_coco(
                                    prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis,
                                    prototype_dict['high_prototype_num'],
                                    prototype_dict['low_prototype_num'],
                                    cfg
                                )  
                            else:
                                prototype_center_list, prototype_center_list_prob, prototype_id_list, prototype_num_list = prototype_KMeans(
                                    prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis,
                                    prototype_dict['high_prototype_num'],
                                    prototype_dict['low_prototype_num'],
                                    cfg)
                            if rank == 0:
                                print(prototype_num_list)
                        elif prototype_dict['type'] == 'random':  
                            prototype_center_list, prototype_center_list_prob, prototype_id_list = prototype_KMeans_random(
                                prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis,
                                prototype_dict['high_prototype_num'] + prototype_dict['low_prototype_num'])


                        del prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis

                    
                    proto_list = prototype_center_list if prototype_dict['type'] == 'cluster' or prototype_dict[
                        'type'] == 'random' else prototype_center_list_prob

                    with autocast():
                        
                        
                        rep_l = rep_all[:num_labeled]  
                        
                        rep_l_large = F.interpolate(rep_l, (h, w), mode="bilinear",
                                                    align_corners=True)  

                        # inter_time2 = time.time()

                        
                        if prototype_dict['sup_sample'] == 'grid':
                            interval = (h - 1) // prototype_dict['grid_num']  
                            sample_id = [i * interval for i in range(prototype_dict['grid_num'])] + [h - 1]  
                            mask = torch.zeros((h, w))
                            mask[np.ix_(sample_id, sample_id)] = 1  
                            mask = mask.bool()  
                            
                            rep_l_large = rep_l_large.permute(0, 2, 3, 1)  
                            rep_l_sample = rep_l_large[:, mask]  # b 33*33 c
                            rep_l_sample_cal = rep_l_sample.view(-1, rep_l_sample.shape[-1])  # b*33*33 256
                            
                            label_l_sample = label_l.clone()[:, mask].view(-1)  # bhw-》b*33*33

                        elif prototype_dict['sup_sample'] == 'random':
                            pass

                        
                        with torch.no_grad():
                            
                            high_proto_max, low_proto_max = prototype_dict['high_prototype_num'], prototype_dict[
                                'low_prototype_num']
                            if prototype_dict['proto_add']:  
                                high_proto_max += prototype_dict['add_num']
                                low_proto_max += prototype_dict['add_num']

                            proto_tensor = []  
                            for cla in range(len(proto_list)):
                                tmp_cla = proto_list[cla]
                                if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'prob':
                                    tmp_cla_high_pro, tmp_cla_low_pro = tmp_cla['high'], tmp_cla['low']

                                    if tmp_cla_high_pro.shape[0] < high_proto_max:  
                                        lack_num = high_proto_max - tmp_cla_high_pro.shape[0]
                                        for i in range(lack_num):  
                                            tmp_cla_high_pro = torch.cat((tmp_cla_high_pro, (tmp_cla_high_pro[-1]).unsqueeze(0)), dim=0)  # max 256

                                    if tmp_cla_low_pro.shape[0] < low_proto_max:  
                                        lack_num = low_proto_max - tmp_cla_low_pro.shape[0]
                                        for i in range(lack_num):  
                                            tmp_cla_low_pro = torch.cat((tmp_cla_low_pro, (tmp_cla_low_pro[-1]).unsqueeze(0)), dim=0)  # max 256


                                    tmp_proto = torch.cat((tmp_cla_high_pro, tmp_cla_low_pro), dim=0)  # 6 256
                                elif prototype_dict['type'] == 'random':
                                    tmp_proto = tmp_cla['random']  # 6 256
                                proto_tensor.append(tmp_proto)
                            proto_tensor = torch.stack(proto_tensor, dim=0)  # 21 6 256
                            proto_tensor_cal = proto_tensor.clone().view(-1, proto_tensor.shape[-1])  # 126 256

                        
                        cos_sim_l = F.cosine_similarity(rep_l_sample_cal.unsqueeze(1), proto_tensor_cal.unsqueeze(0),
                                                        dim=2)  
                        cos_sim_l = cos_sim_l.view(-1, len(proto_list),
                                                   int(high_proto_max) + int(low_proto_max))  # b*33*33 21 6
                        
                        max_sim_l, max_sim_id_l = torch.max(cos_sim_l, dim=2)  

                        
                        _, pro_label_l_sample = torch.max(max_sim_l, dim=1)  # b*33*33
                        fault_mask_l = ~(pro_label_l_sample == label_l_sample)  
                        weight_mask_l = torch.ones_like(fault_mask_l)
                        weight_mask_l[fault_mask_l] = prototype_dict['fault_weight']  

                        loss_proto_l = F.cross_entropy(
                            max_sim_l / prototype_dict['cosine_tao'], label_l_sample.long().cuda(),
                            ignore_index=cfg["dataset"]["ignore_label"])  

                        
                        
                        # tmp_delay_epoch = prototype_dict['sample_delay'] + prototype_dict['sample_last']  # 2
                        tmp_delay_epoch = 5
                        curr_pseudo_threshold = prototype_dict['pseudo_threshold'] + \
                                                (0.95 - prototype_dict['pseudo_threshold']) * \
                                                ((i_iter - tmp_delay_epoch * len(loader_l)) / ((cfg['trainer']['epochs'] - tmp_delay_epoch) * len(loader_l)))
                        # curr_pseudo_threshold = 0.85
                        # if i_iter % 10 == 0 and rank == 0:
                        #     print(f"psedu_thre: {curr_pseudo_threshold}")
                        
                        pseudo_mask = logits_u_aug >= curr_pseudo_threshold  
                        pseudo_label_u_aug = label_u_aug[pseudo_mask]  
                        rep_u = rep_all[num_labeled:]  
                        rep_u_large = F.interpolate(rep_u, (h, w), mode="bilinear",
                                                    align_corners=True)  
                        rep_u_large = rep_u_large.permute(0, 2, 3, 1)  # bhwc
                        rep_u_pseudo = rep_u_large[pseudo_mask]  

                        if rep_u_pseudo.shape[0] >= rep_u_large.shape[0] * prototype_dict[
                            'unsup_sample_num']:  
                            sample_u_num = rep_u_large.shape[0] * prototype_dict['unsup_sample_num']
                            sample_u_list = list(np.arange(rep_u_pseudo.shape[0]))
                            random.shuffle(sample_u_list)
                            sample_u_list = sorted(sample_u_list[:sample_u_num])  

                            rep_u_sample = rep_u_pseudo[sample_u_list, :]  
                            pseudo_label_u_sample = pseudo_label_u_aug[sample_u_list]  

                        else:  
                            rep_u_sample = rep_u_pseudo
                            pseudo_label_u_sample = pseudo_label_u_aug

                        
                        cos_sim_u = F.cosine_similarity(rep_u_sample.unsqueeze(1), proto_tensor_cal.unsqueeze(0),
                                                        dim=2)  
                        cos_sim_u = cos_sim_u.view(-1, len(proto_list),
                                                   int(high_proto_max) + int(low_proto_max))  # num 21 6
                        

                        max_sim_u, max_sim_id_u = torch.max(cos_sim_u, dim=2)  

                        
                        _, pro_label_u_sample = torch.max(max_sim_u, dim=1)  # num
                        fault_mask_u = ~(pro_label_u_sample == pseudo_label_u_sample)  
                        weight_mask_u = torch.ones_like(fault_mask_u)
                        weight_mask_u[fault_mask_u] = prototype_dict['fault_weight']  

                        loss_proto_u = F.cross_entropy(
                            max_sim_u / prototype_dict['cosine_tao'], pseudo_label_u_sample.long().cuda(),
                            ignore_index=cfg["dataset"]["ignore_label"])  

                    
                    with torch.no_grad():
                        
                        label_sample_all = torch.cat((label_l_sample, pseudo_label_u_sample))  # n
                        rep_sample_all = torch.cat((rep_l_sample_cal, rep_u_sample), dim=0)  # n 256
                        max_sim_id_sample_all = torch.cat((max_sim_id_l, max_sim_id_u), dim=0)  # n 21
                        weight_mask_all = torch.cat((weight_mask_l, weight_mask_u))  # n

                        ll = list(np.unique(label_sample_all.detach().cpu().numpy()))  
                        for cla in range(len(proto_list)):  
                            if cla not in ll:
                                continue
                            
                            cla_mask = label_sample_all == cla  
                            feature = rep_sample_all[cla_mask]  
                            max_sim_id_cla = max_sim_id_sample_all[cla_mask, cla]  

                            
                            lack_num_high = int(high_proto_max - prototype_num_list['high'][cla])  
                            lack_num_low = int(low_proto_max - prototype_num_list['low'][cla])

                            clone_pos_tmp = max_sim_id_cla.clone()  
                            high_pro_mask_tmp = clone_pos_tmp < high_proto_max  
                            low_pro_mask_tmp = clone_pos_tmp >= high_proto_max  
                            
                            if lack_num_high > 0:  
                                
                                max_sim_id_cla[
                                    (high_pro_mask_tmp * (max_sim_id_cla >= prototype_num_list['high'][cla])).bool()] = \
                                int(prototype_num_list['high'][cla] - 1)

                            
                            if lack_num_low > 0:
                                max_sim_id_cla[(low_pro_mask_tmp * (max_sim_id_cla >= (
                                            high_proto_max + prototype_num_list['low'][cla]))).bool()] = int(high_proto_max + \
                                                                                                prototype_num_list[
                                                                                                    'low'][cla] - 1)

                            if lack_num_high > 0:  
                                max_sim_id_cla[low_pro_mask_tmp] -= lack_num_high

                            weight_mask_cla = weight_mask_all[cla_mask]  

                            for p in range(int(prototype_num_list['high'][cla] + prototype_num_list['low'][cla])):  
                                
                                feat_p = feature[max_sim_id_cla == p]  
                                weight_mask_p = weight_mask_cla[max_sim_id_cla == p]

                                if feat_p.shape[0] > 0:  
                                    weight_mask_p = (weight_mask_p / torch.sum(weight_mask_p)).unsqueeze(
                                        1)  
                                    feat_p = torch.sum(feat_p * weight_mask_p, dim=0)  

                                    
                                else:
                                    continue

                                if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'prob':

                                    if p < int(prototype_num_list['high'][cla]):  # high
                                        
                                        # proto_list[cla]['high'][p] = (1 - prototype_dict['prototype_ema_decay']) * fe + \
                                        #                              prototype_dict['prototype_ema_decay'] * proto_list[cla]['high'][p]
                                        proto_list[cla]['high'][p] = (1 - prototype_dict[
                                            'prototype_ema_decay']) * feat_p + \
                                                                     prototype_dict['prototype_ema_decay'] * \
                                                                     proto_list[cla]['high'][p]

                                    else:  # low
                                        m_id = int(p - prototype_num_list['high'][cla])
                                        
                                        # proto_list[cla]['low'][m_id] = (1 - prototype_dict['prototype_ema_decay']) * fe + \
                                        #                              prototype_dict['prototype_ema_decay'] * proto_list[cla]['low'][m_id]
                                        proto_list[cla]['low'][m_id] = (1 - prototype_dict[
                                            'prototype_ema_decay']) * feat_p + \
                                                                       prototype_dict['prototype_ema_decay'] * \
                                                                       proto_list[cla]['low'][m_id]

                                elif prototype_dict['type'] == 'random':
                                    proto_list[cla]['random'][p] = (1 - prototype_dict[
                                        'prototype_ema_decay']) * feat_p + \
                                                                   prototype_dict['prototype_ema_decay'] * \
                                                                   proto_list[cla]['random'][p]

                        if prototype_dict['type'] == 'cluster' or prototype_dict['type'] == 'random':  
                            prototype_center_list = proto_list
                        else:
                            prototype_center_list_prob = proto_list
                    with autocast():
                        loss_proto = loss_proto_l + loss_proto_u

                        dist.all_reduce(loss_proto)  
                        loss_proto = (
                                loss_proto
                                / world_size
                                * cfg["trainer"]["prototype"].get("loss_weight", 1)
                        )  
            else:
                with autocast():
                    loss_proto = 0 * rep_all.sum()
            
            contra_flag = "none"
            if (cfg["trainer"].get("contrastive", False) and epoch >= cfg["trainer"]['contrastive'][
                'delay']) or (cfg['trainer']["prototype"].get("contrastive", False) and epoch >= (
                    cfg['trainer']['prototype']['sample_delay'] + cfg['trainer']['prototype'][
                'sample_last'])):  
                cfg_contra = cfg["trainer"]["contrastive"]
                contra_flag = "{}:{}".format(
                    cfg_contra["low_rank"], cfg_contra["high_rank"]
                )  
                alpha_t = cfg_contra["low_entropy_threshold"] * (
                        1 - epoch / cfg["trainer"]["epochs"]
                )  

                with torch.no_grad():
                    with autocast():
                        prob = torch.softmax(pred_u_large_teacher, dim=1)  
                        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)  
                        
                        
                        low_thresh = np.percentile(
                            entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                        )  
                        low_entropy_mask = (
                                entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                        )  

                        high_thresh = np.percentile(
                            entropy[label_u_aug != 255].cpu().numpy().flatten(),
                            100 - alpha_t,
                        )
                        high_entropy_mask = (
                                entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()
                        )  

                        low_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                low_entropy_mask.unsqueeze(1),
                            )
                        )  

                        low_mask_all = F.interpolate(
                            low_mask_all, size=pred_all.shape[2:], mode="nearest"
                        )
                        

                        if cfg_contra.get("negative_high_entropy", True):  
                            contra_flag += " high"
                            high_mask_all = torch.cat(
                                (
                                    (label_l.unsqueeze(1) != 255).float(),
                                    high_entropy_mask.unsqueeze(1),
                                )
                            )  
                        else:  
                            contra_flag += " low"
                            high_mask_all = torch.cat(
                                (
                                    (label_l.unsqueeze(1) != 255).float(),
                                    torch.ones(logits_u_aug.shape)
                                        .float()
                                        .unsqueeze(1)
                                        .cuda(),
                                ),
                            )
                        high_mask_all = F.interpolate(
                            high_mask_all, size=pred_all.shape[2:], mode="nearest"
                        )  # down sample

                        
                        label_l_small = F.interpolate(
                            label_onehot(label_l, cfg["net"]["num_classes"]),
                            size=pred_all.shape[2:],
                            mode="nearest",
                        )
                        label_u_small = F.interpolate(
                            label_onehot(label_u_aug, cfg["net"]["num_classes"]),
                            size=pred_all.shape[2:],
                            mode="nearest",
                        )
                    if cfg['trainer']["prototype"].get("contrastive", False):  
                        new_keys, contra_loss = compute_contra_memobank_prototype_loss(
                            rep_all,  
                            label_l_small.long(),  
                            label_u_small.long(),
                            prob_l_teacher.detach(),  
                            prob_u_teacher.detach(),
                            low_mask_all,  
                            high_mask_all,  
                            cfg_contra,  
                            memobank,  
                            queue_ptrlis,  
                            queue_size,  
                            rep_all_teacher.detach(),  
                            proto_list  
                        )

                    else:  

                        if cfg_contra.get("binary", False):  
                            contra_flag += " BCE"
                            contra_loss = compute_binary_memobank_loss(
                                rep_all,
                                torch.cat((label_l_small, label_u_small)).long(),
                                low_mask_all,
                                high_mask_all,
                                prob_all_teacher.detach(),
                                cfg_contra,
                                memobank,
                                queue_ptrlis,
                                queue_size,
                                rep_all_teacher.detach(),
                            )
                        else:  
                            if not cfg_contra.get("anchor_ema", False):  
                                new_keys, contra_loss = compute_contra_memobank_loss(
                                    rep_all,  
                                    label_l_small.long(),  
                                    label_u_small.long(),
                                    prob_l_teacher.detach(),  
                                    prob_u_teacher.detach(),
                                    low_mask_all,  
                                    high_mask_all,  
                                    cfg_contra,  
                                    memobank,  
                                    queue_ptrlis,  
                                    queue_size,  
                                    rep_all_teacher.detach(),  
                                )
                            else:
                                prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                                    rep_all,
                                    label_l_small.long(),
                                    label_u_small.long(),
                                    prob_l_teacher.detach(),
                                    prob_u_teacher.detach(),
                                    low_mask_all,
                                    high_mask_all,
                                    cfg_contra,
                                    memobank,
                                    queue_ptrlis,
                                    queue_size,
                                    rep_all_teacher.detach(),
                                    prototype,
                                )
                with autocast():
                    dist.all_reduce(contra_loss)  
                    contra_loss = (
                            contra_loss
                            / world_size
                            * cfg["trainer"]["contrastive"].get("loss_weight", 1)
                    )  

            else:
                with autocast():
                    contra_loss = 0 * rep_all.sum()
        with autocast():
            loss = sup_loss + unsup_loss + contra_loss + loss_proto  

        #  with amp.scale_loss(loss, optimizer) as scaled_loss:
        

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):  
            with torch.no_grad():
                ema_decay = min(
                    1
                    - 1
                    / (
                            i_iter
                            - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1)
                            + 1
                    ),
                    ema_decay_origin,
                )  
                for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (
                            ema_decay * t_params.data + (1 - ema_decay) * s_params.data
                    )  

        
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = contra_loss.clone().detach()
        dist.all_reduce(reduced_con_loss)
        con_losses.update(reduced_con_loss.item())

        reduced_pro_loss = loss_proto.clone().detach()
        dist.all_reduce(reduced_pro_loss)
        pro_losses.update(reduced_pro_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}][{}] "
                "Iter [{}/{}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "Pro {pro_loss.val:.3f} ({pro_loss.avg:.3f})\t"
                "LR {lr.val:.5f}".format(
                    cfg["dataset"]["n_sup"],
                    contra_flag,
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    data_time=data_times,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    pro_loss=pro_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Con Loss", con_losses.val, i_iter)  
            tb_logger.add_scalar("Pro Loss", pro_losses.val, i_iter)  

        
        if "coco" in cfg["dataset"]["type"]:
            if cfg['trainer']["eval_on"] and i_iter % 1500 == 0 and i_iter != 0:  
                if rank == 0:
                    logger.info("start evaluation")

                if epoch < cfg["trainer"].get("sup_only_epoch", 1):  
                    prec = validate(model, val_loader, epoch, logger)
                else:  
                    prec = validate(model_teacher, val_loader, epoch, logger)

                if rank == 0:
                    state = {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "teacher_state": model_teacher.state_dict(),
                        "best_miou": best_prec,
                    }
                    if prec > best_prec:
                        best_prec = prec
                        torch.save(
                            state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                        )


                    torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], f"ckpt_{epoch}.pth"))


                    logger.info(
                        "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                            best_prec * 100
                        )
                    )
                    tb_logger.add_scalar("mIoU val", prec, epoch)
                model.train()
        else:
            if cfg['trainer']["eval_on"] and i_iter % 150 == 0:  
                if rank == 0:
                    logger.info("start evaluation")

                if epoch < cfg["trainer"].get("sup_only_epoch", 1):  
                    prec = validate(model, val_loader, epoch, logger)
                else:  
                    prec = validate(model_teacher, val_loader, epoch, logger)

                if rank == 0:
                    state = {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "teacher_state": model_teacher.state_dict(),
                        "best_miou": best_prec,
                    }
                    if prec > best_prec:
                        best_prec = prec
                        torch.save(
                            state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                        )


                    torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], f"ckpt_{epoch}.pth"))


                    logger.info(
                        "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                            best_prec * 100
                        )
                    )
                    tb_logger.add_scalar("mIoU val", prec, epoch)
                model.train()


def validate(
        model,
        data_loader,
        epoch,
        logger,
):  
    model.eval()
    data_loader.sampler.set_epoch(epoch)  

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()  

    for step, batch in enumerate(data_loader):  
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)  

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )  
        
        output = output.data.max(1)[1].cpu().numpy()  
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )  

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)  
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)  
    
    mIoU = np.mean(iou_class)  

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


if __name__ == "__main__":
    main()
