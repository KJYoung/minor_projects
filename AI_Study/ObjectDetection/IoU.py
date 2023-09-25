# https://youtu.be/XXYG5ZWtjj0?si=mMW7DrBY_PJ0KbGV
# Given ..
# Box1 = [x11, y11, x12, y12]
# Box2 = [x21, y21, x22, y22]
# Intersection = [max(x11, x21), max(y11, y21), min(x12, x22), min(y12, y22)]
# Union = (Box1 Area) + (Box2 Area) - (Intersection)

import torch


# box_format: midpoint : (cx, cy, w, h)
# box_format: corners : (x1, y1, x2, y2)
def intersection_over_union(boxes_preds, boxes_labels, box_format="corners"):
    # boxes_preds shape is (N, 4) where N is the number of bboxes
    # boxes_labels shape is (N, 4) where N is the number of bboxes
    if box_format == "midpoint":
        x11 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
        y11 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
        x12 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
        y12 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

        x21 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
        y21 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
        x22 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
        y22 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    if box_format == "corners":
        x11 = boxes_preds[..., 0]
        y11 = boxes_preds[..., 1]
        x12 = boxes_preds[..., 2]
        y12 = boxes_preds[..., 3]

        x21 = boxes_labels[..., 0]
        y21 = boxes_labels[..., 1]
        x22 = boxes_labels[..., 2]
        y22 = boxes_labels[..., 3]

    x1 = torch.max(x11, x21)
    y1 = torch.max(y11, y21)
    x2 = torch.min(x12, x22)
    y2 = torch.min(y12, y22)

    # .clamp(0) is for the case when they do not intersect!
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((x12 - x11) * (y12 - y11))
    box2_area = abs((x22 - x21) * (y22 - y21))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
