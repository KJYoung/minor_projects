import torch
import torch.nn as nn
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_max, bestbox = torch.max(ious, dim=0)  # "Responsible Box"

        exists_box = target[..., 20].unsqueeze(3)  # I^obj_i

        # For Box Coordinates
        box_predictions = exists_box * (
            (1 - bestbox) * predictions[..., 21:25] + bestbox * predictions[..., 26:30]
        )
        box_target = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        # (N, S, S, 4) => (N*S*S, 4)
        loss_coord = self.mse(
            torch.flatten(box_predictions, end_dim=-2), torch.flatten(box_target, end_dim=-2)
        )
        # For Object Loss
        pred_box = (
            # [25:26]과 같이 하면 차원 유지.
            bestbox * predictions[..., 25:26]
            + (1 - bestbox) * predictions[..., 20:21]
        )

        # (N*S*S)
        loss_object = self.mse(
            torch.flatten(exists_box * pred_box), torch.flatten(exists_box * target[..., 20:21])
        )
        # For No Object Loss
        pred_box = (
            # [25:26]과 같이 하면 차원 유지.
            bestbox * predictions[..., 25:26]
            + (1 - bestbox) * predictions[..., 20:21]
        )

        # (N, S, S, 1) => (N, S*S)
        loss_noobj = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        ) + self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        # For Class Loss
        loss_class = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        return (
            self.lambda_coord * loss_coord
            + loss_object
            + self.lambda_noobj * loss_noobj
            + loss_class
        )
