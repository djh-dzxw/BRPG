import logging

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):  
    def __init__(self, d_list, **kwargs):  # ../../../../data/splits/cityscapes/744/labeled.txt
        # parse the input list
        self.parse_input_list(d_list, **kwargs)

    def parse_input_list(self, d_list, max_sample=-1, start_idx=-1, end_idx=-1):
        logger = logging.getLogger("global")
        assert isinstance(d_list, str)
        if "cityscapes" in d_list:  
            self.list_sample = [
                [
                    line.strip(),
                    "gtFine/" + line.strip()[12:-15] + "gtFine_labelTrainIds.png",
                ]  # leftImg8bit/train/ulm/ulm_000079_000019_leftImg8bit.png，gtFine/train/ulm/ulm_000079_000019_gtFine_labelTrainIds.png
                
                for line in open(d_list, "r")
            ]  
        elif "pascal" in d_list or "VOC" in d_list:  # VOC
            self.list_sample = [
                [
                    "JPEGImages/{}.jpg".format(line.strip()),
                    "SegmentationClassAug/{}.png".format(line.strip()),
                ]  # JPEGImages/2007_000032.jpg; SegmentationClassAug/2007_000032.jpg;
                
                for line in open(d_list, "r")
            ]
        elif 'coco' in d_list:
            self.list_sample = [
            [
                line.strip().split()[0],
                line.strip().split()[1]
            ]
                for line in open(d_list, 'r')
            ]  
        else:
            raise "unknown dataset!"

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)  
        assert self.num_sample > 0
        logger.info("# samples: {}".format(self.num_sample))

    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()
            if "cityscapes" in path:
                return img.convert(mode)  
            elif "pascal" in path or "VOC" in path:
                if mode == 'RGB':
                    return img.convert(mode)
                elif mode == 'L':
                    return img
            elif 'coco' in path:
                if mode == 'RGB':
                    return img.convert(mode)
                elif mode == 'L':
                    return img
            else:
                raise "unknown dataset!"

    def __len__(self):
        return self.num_sample
