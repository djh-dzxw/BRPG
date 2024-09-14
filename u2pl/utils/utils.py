import logging
import os
import random
from collections import OrderedDict
import itertools
import bisect
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

# from skimage.filters import gaussian
from skimage.measure import label, regionprops

from sklearn.cluster import MiniBatchKMeans


@torch.no_grad()
def gather_together(data):
    dist.barrier()

    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)

    return gather_data


@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    # gather keys before updating queue
    keys = keys.detach().clone().cpu()
    gathered_list = gather_together(keys)
    keys = torch.cat(gathered_list, dim=0).cuda()

    batch_size = keys.shape[0]

    ptr = int(queue_ptr)

    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
    if queue[0].shape[0] >= queue_size:
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr

    return batch_size


@torch.no_grad()
def queue_back_feat_sample(back_feat, prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis):
    # gather keys before updating queue

    # back_feat
    back_feat = back_feat.detach().clone().cpu()
    gathered_list_all = gather_together(back_feat)
    back_feat = torch.cat(gathered_list_all, dim=0).cuda()
    # batch_size_all = back_feat.shape[0]

    class_id_all = np.unique(back_feat[:, -1].detach().clone().cpu().numpy()).astype(int)
    for cla in class_id_all:
        back_feat_tmp = back_feat[back_feat[:, -1] == cla][:, :-1]
        batch_size_tmp = back_feat_tmp.shape[0]
        ptr = int(prototype_feat_ptrlis[cla]['back'])
        queue = torch.cat((prototype_feat_list[cla]['back'], back_feat_tmp.cpu()), dim=0)

        queue_prob = queue[:, -1]
        index_ascend = torch.argsort(queue_prob, descending=False)
        queue = queue[index_ascend, :]

        if queue.shape[0] >= prototype_feat_num[cla]['back']:
            queue = queue[-prototype_feat_num[cla]['back']:, :]
            ptr = prototype_feat_num[cla]['back']
        else:
            ptr = (ptr + batch_size_tmp) % prototype_feat_num[cla]['back']

        prototype_feat_list[cla]['back'] = queue
        prototype_feat_ptrlis[cla]['back'] = ptr


@torch.no_grad()
def queue_high_low_feat_sample(high_feat, low_feat, prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis):
    # gather keys before updating queue

    # high-feat
    high_feat = high_feat.detach().clone().cpu()
    gathered_list_high = gather_together(high_feat)
    high_feat = torch.cat(gathered_list_high, dim=0).cuda()
    # batch_size_high = high_feat.shape[0]

    class_id_high = np.unique(high_feat[:, -1].detach().clone().cpu().numpy()).astype(int)
    for cla in class_id_high:
        high_feat_tmp = high_feat[high_feat[:, -1] == cla][:, :-1]
        batch_size_tmp = high_feat_tmp.shape[0]
        ptr = int(prototype_feat_ptrlis[cla]['high'])
        queue = torch.cat((prototype_feat_list[cla]['high'], high_feat_tmp.cpu()), dim=0)
        if queue.shape[0] >= prototype_feat_num[cla]['high']:
            queue = queue[-prototype_feat_num[cla]['high']:, :]
            ptr = prototype_feat_num[cla]['high']
        else:
            ptr = (ptr + batch_size_tmp) % prototype_feat_num[cla]['high']

        prototype_feat_list[cla]['high'] = queue
        prototype_feat_ptrlis[cla]['high'] = ptr

    # low-feat
    low_feat = low_feat.detach().clone().cpu()
    gathered_list_low = gather_together(low_feat)
    low_feat = torch.cat(gathered_list_low, dim=0).cuda()
    # batch_size_low = low_feat.shape[0]

    class_id_low = np.unique(low_feat[:, -1].detach().clone().cpu().numpy()).astype(int)
    for cla in class_id_low:
        low_feat_tmp = low_feat[low_feat[:, -1] == cla][:, :-1]
        batch_size_tmp = low_feat_tmp.shape[0]
        ptr = int(prototype_feat_ptrlis[cla]['low'])
        queue = torch.cat((prototype_feat_list[cla]['low'], low_feat_tmp.cpu()), dim=0)
        if queue.shape[0] >= prototype_feat_num[cla]['low']:
            queue = queue[-prototype_feat_num[cla]['low']:, :]
            ptr = prototype_feat_num[cla]['low']
        else:
            ptr = (ptr + batch_size_tmp) % prototype_feat_num[cla]['low']

        prototype_feat_list[cla]['low'] = queue
        prototype_feat_ptrlis[cla]['low'] = ptr


@torch.no_grad()
def queue_random_feat_sample(all_feat, prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis):
    # gather keys before updating queue

    # all-feat
    all_feat = all_feat.detach().clone().cpu()
    gathered_list_all = gather_together(all_feat)
    all_feat = torch.cat(gathered_list_all, dim=0).cuda()
    # batch_size_all = all_feat.shape[0]

    class_id_all = np.unique(all_feat[:, -1].detach().clone().cpu().numpy()).astype(int)
    for cla in class_id_all:
        all_feat_tmp = all_feat[all_feat[:, -1] == cla][:, :-1]
        batch_size_tmp = all_feat_tmp.shape[0]
        ptr = int(prototype_feat_ptrlis[cla]['random'])
        queue = torch.cat((prototype_feat_list[cla]['random'], all_feat_tmp.cpu()), dim=0)
        if queue.shape[0] >= prototype_feat_num[cla]['random']:
            queue = queue[-prototype_feat_num[cla]['random']:, :]
            ptr = prototype_feat_num[cla]['random']
        else:
            ptr = (ptr + batch_size_tmp) % prototype_feat_num[cla]['random']

        prototype_feat_list[cla]['random'] = queue
        prototype_feat_ptrlis[cla]['random'] = ptr


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def cal_pixel_num(pred_map):
    res = [0] * 19
    vals = torch.unique(pred_map)
    for val in vals:
        if val != 255:
            res[val] = torch.sum(pred_map == val).item()
    return np.array(res)


def init_cutmix(crop_size):
    h = crop_size
    w = crop_size
    n_masks = 1
    prop_range = 0.5
    mask_props = np.random.uniform(prop_range, prop_range, size=(n_masks, 1))
    y_props = np.exp(
        np.random.uniform(low=0.0, high=1.0, size=(n_masks, 1)) * np.log(mask_props)
    )
    x_props = mask_props / y_props
    sizes = np.round(
        np.stack([y_props, x_props], axis=2) * np.array((h, w))[None, None, :]
    )
    positions = np.round(
        (np.array((h, w)) - sizes)
        * np.random.uniform(low=0.0, high=1.0, size=sizes.shape)
    )
    rectangles = np.append(positions, positions + sizes, axis=2)[0, 0]
    return rectangles


def padding_bbox_old(rectangles, size):
    area = size ** 2
    y0, x0, y1, x1 = rectangles
    if (y1 - y0) >= (x1 - x0):
        y0 = max(y0 - 40, 0)
        y1 = min(y1 + 40, size)
        new_delta = area / (y1 - y0)
        if new_delta <= (x1 - x0):
            pass
        else:
            new_delta = (new_delta - (x1 - x0)) / 2
            x0 = max(x0 - new_delta, 0)
            x1 = min(x1 + new_delta, size)
    else:
        x0 = max(x0 - 40, 0)
        x1 = max(x1 + 40, size)
        new_delta = area / (x1 - x0)
        if new_delta <= (y1 - y0):
            pass
        else:
            new_delta = (new_delta - (y1 - y0)) / 2
            y0 = max(y0 - new_delta, 0)
            y1 = min(y1 + new_delta, size)
    return [y0, x0, y1, x1]


def padding_bbox_new(rectangles, size):
    area = 0.5 * (size ** 2)
    y0, x0, y1, x1 = rectangles
    h = y1 - y0
    w = x1 - x0
    upper_h = min(int(area / w), size)
    upper_w = min(int(area / h), size)
    new_h = int(
        size * (np.exp(np.random.uniform(low=0.0, high=1.0, size=(1)) * np.log(0.5)))
    )
    new_w = int(area / new_h)
    delta_h = new_h - h
    delta_w = new_w - w
    y_ratio = y0 / (size - y1 + 1)
    x_ratio = x0 / (size - x1 + 1)
    x1 = min(x1 + int(delta_w * (1 / (1 + x_ratio))), size)
    x0 = max(x0 - int(delta_w * (x_ratio / (1 + x_ratio))), 0)
    y1 = min(y1 + int(delta_h * (1 / (1 + y_ratio))), size)
    y0 = max(y0 - int(delta_h * (y_ratio / (1 + y_ratio))), 0)
    return [y0, x0, y1, x1]


def sliming_bbox(rectangles, size):
    area = 0.5 * (size ** 2)
    y0, x0, y1, x1 = rectangles
    h = y1 - y0
    w = x1 - x0
    lower_h = int(area / w)
    if lower_h > h:
        print("wrong")
        new_h = h
    else:
        new_h = random.randint(lower_h, h)
    new_w = int(area / new_h)
    if new_w > w:
        print("wrong")
        new_w = w - 1
    delta_h = h - new_h
    delta_w = w - new_w
    prob = random.random()
    if prob > 0.5:
        y1 = max(random.randint(y1 - delta_h, y1), y0)
        y0 = max(y1 - new_h, y0)
    else:
        y0 = min(random.randint(y0, y0 + delta_h), y1)
        y1 = min(y0 + new_h, y1)
    prob = random.random()
    if prob > 0.5:
        x1 = max(random.randint(x1 - delta_w, x1), x0)
        x0 = max(x1 - new_w, x0)
    else:
        x0 = min(random.randint(x0, x0 + delta_w), x1)
        x1 = min(x0 + new_w, x1)
    return [y0, x0, y1, x1]


def padding_bbox(rectangles, size):
    area = 0.5 * (size ** 2)
    y0, x0, y1, x1 = rectangles
    h = y1 - y0
    w = x1 - x0
    upper_h = int(area / w)
    upper_w = int(area / h)
    if random.random() > 0.5:
        if upper_h > h:
            new_h = random.randint(h, upper_h)
        else:
            new_h = h
        new_w = int(area / new_h)
    else:
        new_w = random.randint(w, upper_w)
        new_h = int(area / new_w)
    delta_h = new_h - h
    delta_w = new_w - w
    prob = random.random()
    if prob > 0.5:
        y1 = min(random.randint(y1, y1 + delta_h), size)
        y0 = max(y1 - new_h, 0)
    else:
        y0 = max(random.randint(y0 - delta_h, y0), 0)
        y1 = min(y0 + new_h, size)
    prob = random.random()
    if prob > 0.5:
        x1 = min(random.randint(x1, x1 + delta_w), size)
        x0 = max(x1 - new_w, 0)
    else:
        x0 = max(random.randint(x0 - delta_w, x0), 0)
        x1 = min(x0 + new_w, size)
    return [y0, x0, y1, x1]


def generate_cutmix(pred, cat, area_thresh, no_pad=False, no_slim=False):
    h = pred.shape[0]
    # print('h',h)
    area_all = h ** 2
    pred = (pred == cat) * 1
    pred = label(pred)
    prop = regionprops(pred)
    values = np.unique(pred)[1:]
    random.shuffle(values)

    flag = 0
    for value in values:
        if np.sum(pred == value) > area_thresh * area_all:
            flag = 1
            break
    if flag == 1:
        rectangles = prop[value - 1].bbox
        # area = prop[value-1].area
        area = (rectangles[2] - rectangles[0]) * (rectangles[3] - rectangles[1])
        if area >= 0.5 * area_all and not no_slim:
            rectangles = sliming_bbox(rectangles, h)
        elif area < 0.5 * area_all and not no_pad:
            rectangles = padding_bbox_new(rectangles, h)
        else:
            pass
    else:
        rectangles = init_cutmix(h)
    return rectangles


def sample_from_bank(cutmix_bank, conf, smooth=False):
    # cutmix_bank [num_classes, len(dataset)]
    conf = (1 - conf).numpy()
    if smooth:
        conf = conf ** (1 / 3)
    conf = np.exp(conf) / np.sum(np.exp(conf))
    classes = [i for i in range(cutmix_bank.shape[0])]
    class_id = np.random.choice(classes, p=conf)
    sample_bank = torch.nonzero(cutmix_bank[class_id])
    if len(sample_bank) > 0:
        sample_id = random.choice(sample_bank)
    else:
        sample_id = random.randint(0, cutmix_bank.shape[1] - 1)
    return sample_id, class_id


def generate_cutmix_mask(
        pred, sample_cat, area_thresh=0.0001, no_pad=False, no_slim=False
):
    h, w = pred.shape[0], pred.shape[1]
    valid_mask = np.zeros((h, w))
    values = np.unique(pred)
    if not sample_cat in values:
        rectangles = init_cutmix(h)
    else:
        rectangles = generate_cutmix(
            pred, sample_cat, area_thresh, no_pad=no_pad, no_slim=no_slim
        )
    y0, x0, y1, x1 = rectangles
    valid_mask[int(y0): int(y1), int(x0): int(x1)] = 1
    valid_mask = torch.from_numpy(valid_mask).long().cuda()

    return valid_mask


def update_cutmix_bank(
        cutmix_bank, preds_teacher_unsup, img_id, sample_id, area_thresh=0.0001
):
    # cutmix_bank [num_classes, len(dataset)]
    # preds_teacher_unsup [2,num_classes,h,w]
    area_all = preds_teacher_unsup.shape[-1] ** 2
    pred1 = preds_teacher_unsup[0].max(0)[1]  # (h,w)
    pred2 = preds_teacher_unsup[1].max(0)[1]  # (h,w)
    values1 = torch.unique(pred1)
    values2 = torch.unique(pred2)
    # for img1
    for idx in range(cutmix_bank.shape[0]):
        if idx not in values1:
            cutmix_bank[idx][img_id] = 0
        elif torch.sum(pred1 == idx) < area_thresh * area_all:
            cutmix_bank[idx][img_id] = 0
        else:
            cutmix_bank[idx][img_id] = 1
    # for img2
    for idx in range(cutmix_bank.shape[0]):
        if idx not in values2:
            cutmix_bank[idx][sample_id] = 0
        elif torch.sum(pred2 == idx) < area_thresh * area_all:
            cutmix_bank[idx][sample_id] = 0
        else:
            cutmix_bank[idx][sample_id] = 1

    return cutmix_bank


def update_cutmix_mask(pred_map, num_classes):
    # Input: H,W
    # Output: List of num_classes *4
    rectangles = np.zeros((num_classes * 4))
    values = np.unique(pred_map)
    for idx in range(num_classes):
        if idx not in values:
            rectangles[4 * idx + 4: 4 * idx + 8] = [0, 0, 0, 0]
            continue
        rectangles[4 * idx + 4: 4 * idx + 8] = generate_cutmix(pred_map, idx)
    return rectangles


def init_cutmix_bank(cutmix_bank, crop_size):
    # input cutmix_bank (num_images, 4*(num_classes+1))
    # for initialization, only initial the first four values
    h = crop_size
    w = crop_size
    n_masks = 1
    prop_range = 0.5
    mask_props = np.random.uniform(prop_range, prop_range, size=(n_masks, 1))
    for n in range(cutmix_bank.shape[0]):
        y_props = np.exp(
            np.random.uniform(low=0.0, high=1.0, size=(n_masks, 1)) * np.log(mask_props)
        )
        x_props = mask_props / y_props
        sizes = np.round(
            np.stack([y_props, x_props], axis=2) * np.array((h, w))[None, None, :]
        )
        positions = np.round(
            (np.array((h, w)) - sizes)
            * np.random.uniform(low=0.0, high=1.0, size=sizes.shape)
        )
        rectangles = np.append(positions, positions + sizes, axis=2)[0, 0]
        for ind in range(len(rectangles)):
            cutmix_bank[n][ind] = rectangles[ind]
    return cutmix_bank


def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dynamic_copy_paste(images_sup, labels_sup, query_cat, select_prob=1):
    images_sup, paste_imgs = torch.chunk(images_sup, 2, dim=1)  # 4 6 h w
    labels_sup, paste_labels = torch.chunk(labels_sup, 2, dim=1)  # 4 2h w
    labels_sup, paste_labels = labels_sup.squeeze(1), paste_labels.squeeze(1)  # bhw

    compose_imgs = []
    compose_labels = []
    for idx in range(images_sup.shape[0]):
        paste_label = paste_labels[idx]
        image_sup = images_sup[idx]
        label_sup = labels_sup[idx]
        if torch.sum(paste_label) > 0 and random.random() < select_prob:
            paste_img = paste_imgs[idx]
            alpha = torch.zeros_like(paste_label).int()
            for cat in query_cat:
                alpha = alpha.__or__((paste_label == cat).int())
            alpha = (alpha > 0).int()
            compose_img = (1 - alpha) * image_sup + alpha * paste_img
            compose_label = (1 - alpha) * label_sup + alpha * paste_label
            compose_imgs.append(compose_img.unsqueeze(0))
            compose_labels.append(compose_label.unsqueeze(0))
        else:
            compose_imgs.append(image_sup.unsqueeze(0))
            compose_labels.append(label_sup.unsqueeze(0))
    compose_imgs = torch.cat(compose_imgs, dim=0)
    compose_labels = torch.cat(compose_labels, dim=0)
    return compose_imgs, compose_labels


def cal_category_confidence(
        preds_student_sup, gt, num_classes
):
    category_confidence = torch.zeros(num_classes).type(torch.float32)
    preds_student_sup = F.softmax(preds_student_sup, dim=1)
    for ind in range(num_classes):
        cat_mask_sup_gt = gt == ind
        if torch.sum(cat_mask_sup_gt) == 0:  # 如果没有该类别，value置-1
            value = -1
        else:
            conf_map_sup = preds_student_sup[:, ind, :, :]
            value = torch.sum(conf_map_sup * cat_mask_sup_gt) / (
                    torch.sum(cat_mask_sup_gt) + 1e-12
            )
        category_confidence[ind] = value

    return category_confidence


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def ignore_state_head(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "head" in k:
            continue
        new_state_dict[k] = v
    return new_state_dict


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def colorize(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])  # hwc
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]

    return Image.fromarray(np.uint8(color_mask))


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


def load_state(path, model, optimizer=None, key="state_dict"):
    rank = dist.get_rank()

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        state_dict = checkpoint[key]

        for k, v in state_dict.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    if rank == 0:
                        print(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            )
                        )

        for k in ignore_keys:
            checkpoint.pop(k)

        model.load_state_dict(state_dict, strict=False)

        if rank == 0:
            ckpt_keys = set(state_dict.keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            best_metric = checkpoint["best_miou"]
            last_iter = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if rank == 0:
                print(
                    "=> also loaded optimizer from checkpoint '{}' (epoch {})".format(
                        path, last_iter
                    )
                )
            return best_metric, last_iter
    else:
        if rank == 0:
            print("=> no checkpoint found at '{}'".format(path))


def create_cityscapes_label_colormap():
    """Creates a label colormap used in CityScapes segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [128, 64, 128]
    colormap[1] = [244, 35, 232]
    colormap[2] = [70, 70, 70]
    colormap[3] = [102, 102, 156]
    colormap[4] = [190, 153, 153]
    colormap[5] = [153, 153, 153]
    colormap[6] = [250, 170, 30]
    colormap[7] = [220, 220, 0]
    colormap[8] = [107, 142, 35]
    colormap[9] = [152, 251, 152]
    colormap[10] = [70, 130, 180]
    colormap[11] = [220, 20, 60]
    colormap[12] = [255, 0, 0]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]

    return colormap

def create_coco_label_colormap():
    """Creates a label colormap used in CityScapes segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    cmap = np.zeros((256, 3), dtype='uint8')

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap
def create_pascal_label_colormap():
    """Creates a label colormap used in Pascal segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = 255 * np.ones((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [128, 0, 0]
    colormap[2] = [0, 128, 0]
    colormap[3] = [128, 128, 0]
    colormap[4] = [0, 0, 128]
    colormap[5] = [128, 0, 128]
    colormap[6] = [0, 128, 128]
    colormap[7] = [128, 128, 128]
    colormap[8] = [64, 0, 0]
    colormap[9] = [192, 0, 0]
    colormap[10] = [64, 128, 0]
    colormap[11] = [192, 128, 0]
    colormap[12] = [64, 0, 128]
    colormap[13] = [192, 0, 128]
    colormap[14] = [64, 128, 128]
    colormap[15] = [192, 128, 128]
    colormap[16] = [0, 64, 0]
    colormap[17] = [128, 64, 0]
    colormap[18] = [0, 192, 0]
    colormap[19] = [128, 192, 0]
    colormap[20] = [0, 64, 128]

    return colormap


@torch.no_grad()
def prototype_KMeans(prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis, high_prototype_num,
                     low_prototype_num, cfg=None):
    prototype_center_list = []
    prototype_center_list_prob = []
    prototype_id_list = []

    prototype_num_list = {'high': torch.ones((len(prototype_feat_list),)) * high_prototype_num,
                          'low': torch.ones((len(prototype_feat_list),)) * low_prototype_num}
    if cfg['trainer']['prototype']['proto_add']:
        if cfg['trainer']['prototype']['intra_var_eval'] == 'l2':
            l2_dis_high = []
            l2_dis_low = []
            for cla in range(len(prototype_feat_list)):
                cla_feat_high = prototype_feat_list[cla]['high'][:, :-1]  # 30000, 256
                cla_feat_low = prototype_feat_list[cla]['low'][:, :-1]  # 30000, 256
                cla_feat_high_mean = (torch.mean(cla_feat_high, dim=0)).unsqueeze(0)
                cla_feat_low_mean = (torch.mean(cla_feat_low, dim=0)).unsqueeze(0)

                cla_l2_dis_high = torch.sqrt(torch.sum((cla_feat_high - cla_feat_high_mean) ** 2, dim=1))
                cla_l2_dis_high_mean = torch.mean(cla_l2_dis_high)
                l2_dis_high.append(cla_l2_dis_high_mean.item())

                cla_l2_dis_low = torch.sqrt(
                    torch.sum((cla_feat_low - cla_feat_low_mean) ** 2, dim=1))
                cla_l2_dis_low_mean = torch.mean(cla_l2_dis_low)
                l2_dis_low.append(cla_l2_dis_low_mean.item())
            l2_dis_high, l2_dis_low = np.array(l2_dis_high), np.array(l2_dis_low)

            add_cla_num = int(
                round(cfg['trainer']['prototype']['add_proportion'] * cfg['net']['num_classes']))
            add_cla_num = add_cla_num + 1 if add_cla_num == 0 else add_cla_num
            threshold_high = np.sort(l2_dis_high.copy())[-add_cla_num]
            threshold_low = np.sort(l2_dis_low.copy())[-add_cla_num]

            high_added_list = l2_dis_high >= threshold_high
            low_added_list = l2_dis_low >= threshold_low

            prototype_num_list['high'][high_added_list] = high_prototype_num + cfg['trainer']['prototype']['add_num']
            prototype_num_list['low'][low_added_list] = low_prototype_num + cfg['trainer']['prototype']['add_num']

        elif cfg['trainer']['prototype']['intra_var_eval'] == 'l2_norm':

            l2_dis_high_norm = []
            l2_dis_low_norm = []
            for cla in range(len(prototype_feat_list)):
                cla_feat_high = prototype_feat_list[cla]['high'][:, :-1]  # 30000, 256
                cla_feat_low = prototype_feat_list[cla]['low'][:, :-1]  # 30000, 256

                cla_feat_high_norm = cla_feat_high / (torch.sqrt(torch.sum(cla_feat_high ** 2, dim=1))).unsqueeze(
                    1)  # 30000, 256
                cla_feat_low_norm = cla_feat_low / (torch.sqrt(torch.sum(cla_feat_low ** 2, dim=1))).unsqueeze(
                    1)  # 30000, 256
                cla_feat_high_mean = (torch.mean(cla_feat_high, dim=0)).unsqueeze(0)
                cla_feat_low_mean = (torch.mean(cla_feat_low, dim=0)).unsqueeze(0)
                cla_high_mean_norm = cla_feat_high_mean / (torch.sqrt(torch.sum(cla_feat_high_mean ** 2, dim=1))
                                                           ).unsqueeze(1)  # 1 256
                cla_low_mean_norm = cla_feat_low_mean / (torch.sqrt(torch.sum(cla_feat_low_mean ** 2, dim=1))
                                                         ).unsqueeze(1)  # 1 256
                cla_l2_dis_high_norm = torch.sqrt(
                    torch.sum((cla_feat_high_norm - cla_high_mean_norm) ** 2, dim=1))
                cla_l2_dis_high_mean_norm = torch.mean(cla_l2_dis_high_norm)
                l2_dis_high_norm.append(cla_l2_dis_high_mean_norm.item())

                cla_l2_dis_low_norm = torch.sqrt(
                    torch.sum((cla_feat_low_norm - cla_low_mean_norm) ** 2, dim=1))
                cla_l2_dis_low_mean_norm = torch.mean(cla_l2_dis_low_norm)
                l2_dis_low_norm.append(cla_l2_dis_low_mean_norm.item())

            l2_dis_high_norm, l2_dis_low_norm = np.array(l2_dis_high_norm), np.array(l2_dis_low_norm)

            add_cla_num = int(
                round(cfg['trainer']['prototype']['add_proportion'] * cfg['net']['num_classes']))
            add_cla_num = add_cla_num + 1 if add_cla_num == 0 else add_cla_num
            threshold_high = np.sort(l2_dis_high_norm.copy())[-add_cla_num]
            threshold_low = np.sort(l2_dis_low_norm.copy())[-add_cla_num]

            high_added_list = l2_dis_high_norm >= threshold_high
            low_added_list = l2_dis_low_norm >= threshold_low

            prototype_num_list['high'][high_added_list] = high_prototype_num + cfg['trainer']['prototype']['add_num']
            prototype_num_list['low'][low_added_list] = low_prototype_num + cfg['trainer']['prototype']['add_num']

        elif cfg['trainer']['prototype']['intra_var_eval'] == 'cos_sim':
            cosine_sim_high = []
            cosine_sim_low = []
            for cla in range(len(prototype_feat_list)):
                cla_feat_high = prototype_feat_list[cla]['high'][:, :-1]  # 30000, 256
                cla_feat_low = prototype_feat_list[cla]['low'][:, :-1]  # 30000, 256

                cla_feat_high_norm = cla_feat_high / (torch.sqrt(torch.sum(cla_feat_high ** 2, dim=1))).unsqueeze(
                    1)  # 30000, 256
                cla_feat_low_norm = cla_feat_low / (torch.sqrt(torch.sum(cla_feat_low ** 2, dim=1))).unsqueeze(
                    1)  # 30000, 256
                cla_feat_high_mean = (torch.mean(cla_feat_high, dim=0)).unsqueeze(0)
                cla_feat_low_mean = (torch.mean(cla_feat_low, dim=0)).unsqueeze(0)
                cla_high_mean_norm = cla_feat_high_mean / (torch.sqrt(torch.sum(cla_feat_high_mean ** 2, dim=1))
                                                           ).unsqueeze(1)  # 1 256
                cla_low_mean_norm = cla_feat_low_mean / (torch.sqrt(torch.sum(cla_feat_low_mean ** 2, dim=1))
                                                         ).unsqueeze(1)  # 1 256

                cla_cos_sim_high = torch.mean((F.cosine_similarity(cla_feat_high_norm.unsqueeze(1),
                                                                   cla_high_mean_norm.unsqueeze(0),
                                                                   dim=2)).squeeze()).item()
                cla_cos_sim_low = torch.mean((F.cosine_similarity(cla_feat_low_norm.unsqueeze(1),
                                                                  cla_low_mean_norm.unsqueeze(0),
                                                                  dim=2)).squeeze()).item()
                cosine_sim_high.append(cla_cos_sim_high)
                cosine_sim_low.append(cla_cos_sim_low)

            cosine_sim_high, cosine_sim_low = np.array(cosine_sim_high), np.array(cosine_sim_low)

            add_cla_num = int(
                round(cfg['trainer']['prototype']['add_proportion'] * cfg['net']['num_classes']))
            add_cla_num = add_cla_num + 1 if add_cla_num == 0 else add_cla_num
            threshold_high = np.sort(cosine_sim_high.copy())[add_cla_num - 1]
            threshold_low = np.sort(cosine_sim_low.copy())[add_cla_num - 1]

            high_added_list = cosine_sim_high <= threshold_high
            low_added_list = cosine_sim_low <= threshold_low

            prototype_num_list['high'][high_added_list] = high_prototype_num + cfg['trainer']['prototype']['add_num']
            prototype_num_list['low'][low_added_list] = low_prototype_num + cfg['trainer']['prototype']['add_num']

        else:
            raise ValueError

    for cla in range(len(prototype_feat_list)):
        cla_high_pro_num, cla_low_pro_num = int(prototype_num_list['high'][cla]), int(prototype_num_list['low'][cla])
        center_dict = {}
        id_dict = {}
        center_prob_dict = {'high': [], 'low': []}

        feat_high, feat_low = prototype_feat_list[cla]['high'][:, :-1], prototype_feat_list[cla]['low'][:,
                                                                        :-1]
        feat_high_prob, feat_low_prob = prototype_feat_list[cla]['high'][:, -1], prototype_feat_list[cla]['low'][:,
                                                                                 -1]
        feat_high_np, feat_low_np = feat_high.detach().cpu().numpy(), feat_low.detach().cpu().numpy()
        feat_high_prob_np, feat_low_prob_np = feat_high_prob.detach().cpu().numpy(), feat_low_prob.detach().cpu().numpy()

        k_means_high = MiniBatchKMeans(n_clusters=cla_high_pro_num, random_state=0).fit(feat_high_np)
        center_dict['high'] = torch.from_numpy(k_means_high.cluster_centers_).cuda()
        id_dict['high'] = list(k_means_high.labels_)
        id_dict['high'] = [int(x) for x in id_dict['high']]

        for ih in range(cla_high_pro_num):
            mask = k_means_high.labels_ == ih
            tmp_feat = feat_high_np[mask, :]
            prob_norm = feat_high_prob_np[mask] / np.sum(feat_high_prob_np[mask])
            prob_norm = np.expand_dims(prob_norm, axis=1)  # num * 1
            center_tmp = np.sum(tmp_feat * prob_norm, axis=0)
            center_prob_dict['high'].append(torch.from_numpy(center_tmp).cuda())
        center_prob_dict['high'] = torch.stack(center_prob_dict['high'], dim=0)  # 3 256

        k_means_low = MiniBatchKMeans(n_clusters=cla_low_pro_num, random_state=0).fit(feat_low_np)
        center_dict['low'] = torch.from_numpy(k_means_low.cluster_centers_).cuda()
        id_dict['low'] = list(k_means_low.labels_)
        id_dict['low'] = [int(x) for x in id_dict['low']]

        for il in range(cla_low_pro_num):
            mask = k_means_low.labels_ == il
            tmp_feat = feat_low_np[mask, :]
            prob_norm = feat_low_prob_np[mask] / np.sum(feat_low_prob_np[mask])
            prob_norm = np.expand_dims(prob_norm, axis=1)  # num * 1
            center_tmp = np.sum(tmp_feat * prob_norm, axis=0)
            center_prob_dict['low'].append(torch.from_numpy(center_tmp).cuda())
        center_prob_dict['low'] = torch.stack(center_prob_dict['low'], dim=0)  # 3 256

        prototype_center_list.append(center_dict)
        prototype_id_list.append(id_dict)
        prototype_center_list_prob.append(center_prob_dict)

    return prototype_center_list, prototype_center_list_prob, prototype_id_list, prototype_num_list


@torch.no_grad()
def prototype_KMeans_for_coco(prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis, high_prototype_num,
                              low_prototype_num, cfg=None):
    prototype_center_list = []
    prototype_center_list_prob = []
    prototype_id_list = []

    for cla in range(len(prototype_feat_list)):
        if prototype_feat_ptrlis[cla]['high'] >= 10 and prototype_feat_ptrlis[cla]['low'] >= 10:
            continue
        elif prototype_feat_ptrlis[cla]['high'] >= 10 and prototype_feat_ptrlis[cla]['low'] < 10:
            raise ValueError
        elif prototype_feat_ptrlis[cla]['high'] < 10 and prototype_feat_ptrlis[cla]['low'] >= 10:
            if prototype_feat_ptrlis[cla]['back'] > int(prototype_feat_num[cla]['back'] / 2):
                prototype_feat_list[cla]['high'] = prototype_feat_list[cla]['back'][-int(prototype_feat_num[cla]['back'] / 2):, :]
            else:
                prototype_feat_list[cla]['high'] = prototype_feat_list[cla]['back']
        else:
            n_for_low = int(prototype_feat_ptrlis[cla]['back'] / 2)
            prototype_feat_list[cla]['low'], prototype_feat_list[cla]['high'] = prototype_feat_list[cla]['back'][:n_for_low,:], prototype_feat_list[cla]['back'][n_for_low:,:]

    prototype_num_list = {'high': torch.ones((len(prototype_feat_list),)) * high_prototype_num,
                          'low': torch.ones((len(prototype_feat_list),)) * low_prototype_num}
    if cfg['trainer']['prototype']['proto_add']:
        if cfg['trainer']['prototype']['intra_var_eval'] == 'l2':
            l2_dis_high = []
            l2_dis_low = []
            for cla in range(len(prototype_feat_list)):
                cla_feat_high = prototype_feat_list[cla]['high'][:, :-1]  # 30000, 256
                cla_feat_low = prototype_feat_list[cla]['low'][:, :-1]  # 30000, 256
                cla_feat_high_mean = (torch.mean(cla_feat_high, dim=0)).unsqueeze(0)
                cla_feat_low_mean = (torch.mean(cla_feat_low, dim=0)).unsqueeze(0)

                cla_l2_dis_high = torch.sqrt(torch.sum((cla_feat_high - cla_feat_high_mean) ** 2, dim=1))  # 30000 L2距离
                cla_l2_dis_high_mean = torch.mean(cla_l2_dis_high)
                l2_dis_high.append(cla_l2_dis_high_mean.item())

                cla_l2_dis_low = torch.sqrt(
                    torch.sum((cla_feat_low - cla_feat_low_mean) ** 2, dim=1))
                cla_l2_dis_low_mean = torch.mean(cla_l2_dis_low)
                l2_dis_low.append(cla_l2_dis_low_mean.item())
            l2_dis_high, l2_dis_low = np.array(l2_dis_high), np.array(l2_dis_low)

            add_cla_num = int(
                round(cfg['trainer']['prototype']['add_proportion'] * cfg['net']['num_classes']))
            add_cla_num = add_cla_num + 1 if add_cla_num == 0 else add_cla_num
            threshold_high = np.sort(l2_dis_high.copy())[-add_cla_num]
            threshold_low = np.sort(l2_dis_low.copy())[-add_cla_num]

            high_added_list = l2_dis_high >= threshold_high
            low_added_list = l2_dis_low >= threshold_low

            prototype_num_list['high'][high_added_list] = high_prototype_num + cfg['trainer']['prototype']['add_num']
            prototype_num_list['low'][low_added_list] = low_prototype_num + cfg['trainer']['prototype']['add_num']

        elif cfg['trainer']['prototype']['intra_var_eval'] == 'l2_norm':

            l2_dis_high_norm = []
            l2_dis_low_norm = []
            for cla in range(len(prototype_feat_list)):
                cla_feat_high = prototype_feat_list[cla]['high'][:, :-1]  # 30000, 256
                cla_feat_low = prototype_feat_list[cla]['low'][:, :-1]  # 30000, 256

                cla_feat_high_norm = cla_feat_high / (torch.sqrt(torch.sum(cla_feat_high ** 2, dim=1))).unsqueeze(
                    1)  # 30000, 256
                cla_feat_low_norm = cla_feat_low / (torch.sqrt(torch.sum(cla_feat_low ** 2, dim=1))).unsqueeze(
                    1)  # 30000, 256
                cla_feat_high_mean = (torch.mean(cla_feat_high, dim=0)).unsqueeze(0)
                cla_feat_low_mean = (torch.mean(cla_feat_low, dim=0)).unsqueeze(0)
                cla_high_mean_norm = cla_feat_high_mean / (torch.sqrt(torch.sum(cla_feat_high_mean ** 2, dim=1))
                                                           ).unsqueeze(1)  # 1 256
                cla_low_mean_norm = cla_feat_low_mean / (torch.sqrt(torch.sum(cla_feat_low_mean ** 2, dim=1))
                                                         ).unsqueeze(1)  # 1 256
                cla_l2_dis_high_norm = torch.sqrt(
                    torch.sum((cla_feat_high_norm - cla_high_mean_norm) ** 2, dim=1))
                cla_l2_dis_high_mean_norm = torch.mean(cla_l2_dis_high_norm)
                l2_dis_high_norm.append(cla_l2_dis_high_mean_norm.item())

                cla_l2_dis_low_norm = torch.sqrt(
                    torch.sum((cla_feat_low_norm - cla_low_mean_norm) ** 2, dim=1))
                cla_l2_dis_low_mean_norm = torch.mean(cla_l2_dis_low_norm)
                l2_dis_low_norm.append(cla_l2_dis_low_mean_norm.item())

            l2_dis_high_norm, l2_dis_low_norm = np.array(l2_dis_high_norm), np.array(l2_dis_low_norm)

            add_cla_num = int(
                round(cfg['trainer']['prototype']['add_proportion'] * cfg['net']['num_classes']))
            add_cla_num = add_cla_num + 1 if add_cla_num == 0 else add_cla_num
            threshold_high = np.sort(l2_dis_high_norm.copy())[-add_cla_num]
            threshold_low = np.sort(l2_dis_low_norm.copy())[-add_cla_num]

            high_added_list = l2_dis_high_norm >= threshold_high
            low_added_list = l2_dis_low_norm >= threshold_low

            prototype_num_list['high'][high_added_list] = high_prototype_num + cfg['trainer']['prototype']['add_num']
            prototype_num_list['low'][low_added_list] = low_prototype_num + cfg['trainer']['prototype']['add_num']

        elif cfg['trainer']['prototype']['intra_var_eval'] == 'cos_sim':
            cosine_sim_high = []
            cosine_sim_low = []
            for cla in range(len(prototype_feat_list)):
                cla_feat_high = prototype_feat_list[cla]['high'][:, :-1]  # 30000, 256
                cla_feat_low = prototype_feat_list[cla]['low'][:, :-1]  # 30000, 256

                cla_feat_high_norm = cla_feat_high / (torch.sqrt(torch.sum(cla_feat_high ** 2, dim=1))).unsqueeze(
                    1)  # 30000, 256
                cla_feat_low_norm = cla_feat_low / (torch.sqrt(torch.sum(cla_feat_low ** 2, dim=1))).unsqueeze(
                    1)  # 30000, 256
                cla_feat_high_mean = (torch.mean(cla_feat_high, dim=0)).unsqueeze(0)
                cla_feat_low_mean = (torch.mean(cla_feat_low, dim=0)).unsqueeze(0)
                cla_high_mean_norm = cla_feat_high_mean / (torch.sqrt(torch.sum(cla_feat_high_mean ** 2, dim=1))
                                                           ).unsqueeze(1)  # 1 256
                cla_low_mean_norm = cla_feat_low_mean / (torch.sqrt(torch.sum(cla_feat_low_mean ** 2, dim=1))
                                                         ).unsqueeze(1)  # 1 256

                cla_cos_sim_high = torch.mean((F.cosine_similarity(cla_feat_high_norm.unsqueeze(1),
                                                                   cla_high_mean_norm.unsqueeze(0),
                                                                   dim=2)).squeeze()).item()
                cla_cos_sim_low = torch.mean((F.cosine_similarity(cla_feat_low_norm.unsqueeze(1),
                                                                  cla_low_mean_norm.unsqueeze(0),
                                                                  dim=2)).squeeze()).item()
                cosine_sim_high.append(cla_cos_sim_high)
                cosine_sim_low.append(cla_cos_sim_low)

            cosine_sim_high, cosine_sim_low = np.array(cosine_sim_high), np.array(cosine_sim_low)

            add_cla_num = int(
                round(cfg['trainer']['prototype']['add_proportion'] * cfg['net']['num_classes']))
            add_cla_num = add_cla_num + 1 if add_cla_num == 0 else add_cla_num
            threshold_high = np.sort(cosine_sim_high.copy())[add_cla_num - 1]
            threshold_low = np.sort(cosine_sim_low.copy())[add_cla_num - 1]

            high_added_list = cosine_sim_high <= threshold_high
            low_added_list = cosine_sim_low <= threshold_low

            prototype_num_list['high'][high_added_list] = high_prototype_num + cfg['trainer']['prototype']['add_num']
            prototype_num_list['low'][low_added_list] = low_prototype_num + cfg['trainer']['prototype']['add_num']

        else:
            raise ValueError

    for cla in range(len(prototype_feat_list)):
        cla_high_pro_num, cla_low_pro_num = int(prototype_num_list['high'][cla]), int(prototype_num_list['low'][cla])
        center_dict = {}
        id_dict = {}
        center_prob_dict = {'high': [], 'low': []}

        feat_high, feat_low = prototype_feat_list[cla]['high'][:, :-1], prototype_feat_list[cla]['low'][:,
                                                                        :-1]
        feat_high_prob, feat_low_prob = prototype_feat_list[cla]['high'][:, -1], prototype_feat_list[cla]['low'][:,
                                                                                 -1]
        feat_high_np, feat_low_np = feat_high.detach().cpu().numpy(), feat_low.detach().cpu().numpy()
        feat_high_prob_np, feat_low_prob_np = feat_high_prob.detach().cpu().numpy(), feat_low_prob.detach().cpu().numpy()
        # high!!!!
        k_means_high = MiniBatchKMeans(n_clusters=cla_high_pro_num, random_state=0).fit(feat_high_np)
        center_dict['high'] = torch.from_numpy(k_means_high.cluster_centers_).cuda()
        id_dict['high'] = list(k_means_high.labels_)
        id_dict['high'] = [int(x) for x in id_dict['high']]

        for ih in range(cla_high_pro_num):
            mask = k_means_high.labels_ == ih
            tmp_feat = feat_high_np[mask, :]
            prob_norm = feat_high_prob_np[mask] / np.sum(feat_high_prob_np[mask])
            prob_norm = np.expand_dims(prob_norm, axis=1)  # num * 1
            center_tmp = np.sum(tmp_feat * prob_norm, axis=0)
            center_prob_dict['high'].append(torch.from_numpy(center_tmp).cuda())
        center_prob_dict['high'] = torch.stack(center_prob_dict['high'], dim=0)  # 3 256

        # low！！！
        k_means_low = MiniBatchKMeans(n_clusters=cla_low_pro_num, random_state=0).fit(feat_low_np)
        center_dict['low'] = torch.from_numpy(k_means_low.cluster_centers_).cuda()
        id_dict['low'] = list(k_means_low.labels_)
        id_dict['low'] = [int(x) for x in id_dict['low']]

        for il in range(cla_low_pro_num):
            mask = k_means_low.labels_ == il
            tmp_feat = feat_low_np[mask, :]
            prob_norm = feat_low_prob_np[mask] / np.sum(feat_low_prob_np[mask])
            prob_norm = np.expand_dims(prob_norm, axis=1)  # num * 1
            center_tmp = np.sum(tmp_feat * prob_norm, axis=0)
            center_prob_dict['low'].append(torch.from_numpy(center_tmp).cuda())
        center_prob_dict['low'] = torch.stack(center_prob_dict['low'], dim=0)  # 3 256

        prototype_center_list.append(center_dict)
        prototype_id_list.append(id_dict)
        prototype_center_list_prob.append(center_prob_dict)

    return prototype_center_list, prototype_center_list_prob, prototype_id_list, prototype_num_list


@torch.no_grad()
def prototype_KMeans_random(prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis, prototype_num):
    prototype_center_list = []
    prototype_center_list_prob = []
    prototype_id_list = []
    for cla in range(len(prototype_feat_list)):
        center_dict = {}
        id_dict = {}
        center_prob_dict = {'random': []}

        feat_random = prototype_feat_list[cla]['random'][:, :-1]
        feat_random_prob = prototype_feat_list[cla]['random'][:, -1]
        feat_random_np = feat_random.detach().cpu().numpy()
        feat_random_prob_np = feat_random_prob.detach().cpu().numpy()  # numpy
        # random!!!!
        k_means_random = MiniBatchKMeans(n_clusters=prototype_num, random_state=0).fit(feat_random_np)
        center_dict['random'] = torch.from_numpy(k_means_random.cluster_centers_).cuda()
        id_dict['random'] = list(k_means_random.labels_)
        id_dict['random'] = [int(x) for x in id_dict['random']]

        for ir in range(prototype_num):
            mask = k_means_random.labels_ == ir
            tmp_feat = feat_random_np[mask, :]
            prob_norm = feat_random_prob_np[mask] / np.sum(feat_random_prob_np[mask])
            prob_norm = np.expand_dims(prob_norm, axis=1)  # num * 1
            center_tmp = np.sum(tmp_feat * prob_norm, axis=0)
            center_prob_dict['random'].append(torch.from_numpy(center_tmp).cuda())
        center_prob_dict['random'] = torch.stack(center_prob_dict['random'], dim=0)  # 3 256

        prototype_center_list.append(center_dict)
        prototype_id_list.append(id_dict)
        prototype_center_list_prob.append(center_prob_dict)

    return prototype_center_list, prototype_center_list_prob, prototype_id_list


@torch.no_grad()
def prototype_KAverageMeans(prototype_feat_list, prototype_feat_num, prototype_feat_ptrlis, high_prototype_num,
                            low_prototype_num):
    pass


def class_balance_sample_high_low(high_feat, low_feat, sample_pixel_num, gamma=0.5):
    '''

    :param high_feat: num 258
    :param low_feat:
    :param sample_pixel_num:
    :return:
    '''

    if high_feat.shape[0] <= sample_pixel_num:
        high_feat_sample = high_feat
    else:

        cla_list_high = np.unique(high_feat[:, -1].detach().cpu().numpy()).astype(
            int)

        cla_pix_num_high = np.array([(torch.sum(high_feat[:, -1] == cla)).item() for cla in cla_list_high])
        cla_pix_num_high = np.power(cla_pix_num_high, gamma)
        sample_prob_high = 1 / cla_pix_num_high
        sample_prob_high = sample_prob_high / np.sum(sample_prob_high)

        sample_weight_high = np.zeros((high_feat.shape[0],))  # num
        for i in range(cla_list_high.shape[0]):
            cla = cla_list_high[i]
            sample_weight_high[high_feat[:, -1].clone().detach().cpu().numpy() == cla] = sample_prob_high[i]

        population_high = np.arange(high_feat.shape[0])  # num
        high_feat_sample_id = weighted_sample(list(population_high), list(sample_weight_high),
                                              k=sample_pixel_num)
        high_feat_sample = high_feat[high_feat_sample_id]

    # low!!!!!!!!!
    if low_feat.shape[0] <= sample_pixel_num:
        low_feat_sample = low_feat
    else:

        cla_list_low = np.unique(low_feat[:, -1].detach().cpu().numpy()).astype(
            int)

        cla_pix_num_low = np.array([(torch.sum(low_feat[:, -1] == cla)).item() for cla in cla_list_low])
        cla_pix_num_low = np.power(cla_pix_num_low, gamma)
        sample_prob_low = 1 / cla_pix_num_low
        sample_prob_low = sample_prob_low / np.sum(sample_prob_low)

        sample_weight_low = np.zeros((low_feat.shape[0],))  # num
        for i in range(cla_list_low.shape[0]):
            cla = cla_list_low[i]
            sample_weight_low[low_feat.clone().detach().cpu().numpy()[:, -1] == cla] = sample_prob_low[i]

        population_low = np.arange(low_feat.shape[0])  # num
        low_feat_sample_id = weighted_sample(list(population_low), list(sample_weight_low),
                                             k=sample_pixel_num)
        low_feat_sample = low_feat[low_feat_sample_id]

    return high_feat_sample, low_feat_sample


def class_balance_sample_random(all_feat, sample_pixel_num, gamma=0.5):
    '''

    :param all_feat: num 258
    :param sample_pixel_num:
    :param gamma:
    :return:
    '''

    if all_feat.shape[0] <= sample_pixel_num:
        all_feat_sample = all_feat
    else:

        cla_list_all = np.unique(all_feat[:, -1].detach().cpu().numpy()).astype(
            int)

        cla_pix_num_all = np.array([(torch.sum(all_feat[:, -1] == cla)).item() for cla in cla_list_all])
        cla_pix_num_all = np.power(cla_pix_num_all, gamma)
        sample_prob_all = 1 / cla_pix_num_all
        sample_prob_all = sample_prob_all / np.sum(sample_prob_all)

        sample_weight_all = np.zeros((all_feat.shape[0],))  # num
        for i in range(cla_list_all.shape[0]):
            cla = cla_list_all[i]
            sample_weight_all[all_feat[:, -1].clone().detach().cpu().numpy() == cla] = sample_prob_all[i]

        population_all = np.arange(all_feat.shape[0])  # num
        all_feat_sample_id = weighted_sample(list(population_all), list(sample_weight_all),
                                             k=sample_pixel_num)
        all_feat_sample = all_feat[all_feat_sample_id]

    return all_feat_sample


def weighted_sample(population, weights, k=1):

    n = len(population)
    if n == 0:
        return []
    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")
    if len(weights) != n:
        raise ValueError('The number of weights does not match the population')

    cum_weights = list(itertools.accumulate(weights))
    total = cum_weights[-1]
    if total <= 0:
        return random.sample(population, k=k)
    hi = len(cum_weights) - 1

    selected = set()
    _bisect = bisect.bisect
    _random = random.random
    selected_add = selected.add
    result = [None] * k
    for i in range(k):
        j = _bisect(cum_weights, _random() * total, 0, hi)
        while j in selected:
            j = _bisect(cum_weights, _random() * total, 0, hi)
        selected_add(j)
        result[i] = population[j]
    return result
