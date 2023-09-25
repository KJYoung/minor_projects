# https://youtu.be/YDkjWEN8jNA?si=PrkSIHeC9PsZYtOi
import torch
from IoU import intersection_over_union


def non_max_suppression(predictions, prob_threshold, iou_threshold, box_format="corners"):
    # predictions : lists of [class, prob, x1, y1, x2, y2]
    assert type(predictions) == list

    bboxes = [box for box in predictions if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)  # Prob 내림차순 정렬
    bboxes_after_nms = []

    while bboxes:
        chosen_bbox = bboxes.pop(0)  # pop()은 last item 뽑는데, 0 index를 줘서 first item 뽑음 ㅎㅎ..

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_bbox[0]
            or intersection_over_union(
                torch.tensor(chosen_bbox[2:]), torch.tensor(box[2:]), box_format=box_format
            )
            < iou_threshold
        ]  # Same class && IOU크면 삭제되므로, 그 역 조건을 만족하면 살아 남음

        bboxes_after_nms.append(chosen_bbox)

    return bboxes_after_nms


predictions = [
    [1, 0.8, 22, 2, 44, 4],
    [1, 0.7, 22, 2, 44, 3.5],
    [1, 0.44, 3, 4, 10, 9],
    [1, 0.66, 2, 2, 4, 4],
    [1, 0.1, 22, 22, 23, 23],
]
print(non_max_suppression(predictions, 0.2, 0.4))
