import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    iou_width_height as iou,
    non_max_suppression as nms,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        S=[13, 26, 52],
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        # For list, add operation executes concatenation! => self.anchors.shape: [9,2]
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])

        bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2)  # [class, x, y, w, h]
        bboxes = np.roll(bboxes, 4, axis=1).tolist()  # [x, y, w, h, class]

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        targets = [
            torch.zeors((self.num_anchors // 3, S, S, 6)) for S in self.S
        ]  # self.S : [ 13, 26, 52 ]
        # 6 : [p_obj, x, y, w, h, class]

        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False, False, False]

            for anchor_idx in anchor_indices:
                # which scale
                scale_idx = anchor_idx // self.num_anchors_per_scale  # 0, 1, 2 / 3, 4, 5 / 6, 7, 8
                S = self.S[scale_idx]

                # which index on the scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                i, j = int(S * y), int(S * x)  # x = 0.5, S = 13 ==> 6
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if (anchor_taken == 0) and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both are between [0, 1]
                    width_cell, height_cell = width * S, height * S

                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif (anchor_taken == 0) and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore this
        return image, tuple(targets)
