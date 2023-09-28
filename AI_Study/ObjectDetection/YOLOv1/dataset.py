import os
import torch
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S, self.B, self.C = S, B, C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])  # index 행의 1열
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                # label is an each line of the label text file.
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])
        boxes = torch.tensor(boxes)

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        # label_matrix = torch.zeros((self.S, self.S, self.C + 5 )) 로 해도 문제는 없는데,
        # prediction 과의 수월한 비교를 위해 그냥 빈 값으로 형태만 맞춰 둠.
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)  # Grid Cell Index
            x_cell, y_cell = (self.S * x - j), (self.S * y - i)

            width_cell, height_cell = width * self.S, height * self.S

            if label_matrix[i, j, 20] == 0:  # Not yet registered
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1
            else:
                print("Warning: Label_Matrix already registered!")
        return image, label_matrix
