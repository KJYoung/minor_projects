import time
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YoloV1
from dataset import VOCDataset
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    load_checkpoint,
    save_checkpoint,
)
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# HyperParameters
LEARNING_RATE = 2e-5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True

LOAD_MODEL = False
LOAD_MODEL_FILE = "trained_models/overfit.pth.tar_epoch950"

IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transforms = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def main():
    model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset("data/100examples.csv", IMG_DIR, LABEL_DIR, transform=transforms)
    test_dataset = VOCDataset("data/test.csv", IMG_DIR, LABEL_DIR, transform=transforms)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(train_loader, model, 0.5, 0.4)
        mAP = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Epoch {epoch} - Trained mAP: {mAP}")

        if epoch % 50 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"trial2_100_epochs{epoch}.pth.tar")
            time.sleep(15)

        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == "__main__":
    main()
