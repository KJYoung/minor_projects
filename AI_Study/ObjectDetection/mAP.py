# https://youtu.be/FppOzcDvaDI?si=sOBaBhF-8-F6-Kfe
# Counter: https://www.daleseo.com/python-collections-counter/
import torch
from collections import Counter
from IoU import intersection_over_union


def mean_average_precision(
    pred_boxes, gt_boxes, iou_threshold=0.5, box_format="corners", num_classes=20
):
    # pred_boxes : list of [train_idx, class_pred, prob_score, x1, y1, x2, y2]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        predictions = []
        ground_truths = []

        for pred in pred_boxes:
            if pred[1] == c:
                predictions.append(pred)

        for gt in gt_boxes:
            if gt[1] == c:
                ground_truths.append(gt)

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
            # amount_bboxes = {0: torch.tensor([0,0,0]), 1: torch.tensor([0,0,0,0,0])}

        predictions.sort(key=lambda x: x[2], reverse=True)  # Objectiveness 내림차순 정렬

        TP = torch.zeros((len(predictions)))
        FP = torch.zeros((len(predictions)))
        total_true_bboxes = len(ground_truths)
        for pred_idx, pred in enumerate(predictions):  # Objectiveness Score 높은 것부터 순회.
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == pred[0]
            ]  # gt bboxes from same image
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(pred[3:]), torch.tensor(gt[3:]), box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # best_iou, best_gt_idx를 찾은 상태!
            if best_iou > iou_threshold:
                if amount_bboxes[pred[0]][best_gt_idx] == 0:  # Has not yet been covered
                    TP[pred_idx] = 1
                    amount_bboxes[pred[0]][best_gt_idx] = 1  # Now, covered
                else:  # Already Covered
                    FP[pred_idx] = 1
            else:  # Invalid BBox
                FP[pred_idx] = 1

        # [1, 1, 0, 1, 0] => [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        recalls = TP_cumsum / (total_true_bboxes + epsilon)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # torch.trapz(y, x)는 그 곡선의 아래 넓이를 구해줌. torch.trapezoid의 alias임.
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


# Multiple IoU 값에 대해서 위 함수를 호출하고 그 평균을 구해야 최종 mAP를 구할 수 있음.
