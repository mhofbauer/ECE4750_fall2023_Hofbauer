import torch
import pandas as pd
import math
from torchvision import transforms
from PIL import Image
import os

class CustomDataloader:
    def __init__(self, x, y=None, batch_size=1, randomize=False, image_mode=False, image_dir=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        self.image_mode = image_mode
        self.image_dir = image_dir 
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((60, 60)),
            transforms.ToTensor(),
        ])
        self.iter = None
        self.num_batches_per_epoch = math.ceil(self.get_length() / self.batch_size)

    def get_length(self):
        return len(self.x)

    def randomize_dataset(self):
        indices = torch.randperm(len(self.x))
        self.x = [self.x[i] for i in indices]
        if self.y is not None:
            self.y = [self.y[i] for i in indices]

    def generate_iter(self):
        if self.randomize:
            self.randomize_dataset()

        batches = []
        for b_idx in range(self.num_batches_per_epoch):
            x_batch = self.x[b_idx * self.batch_size: (b_idx + 1) * self.batch_size]
            y_batch = self.y[b_idx * self.batch_size: (b_idx + 1) * self.batch_size] if self.y is not None else None

            if self.image_mode:
                x_batch = [self.load_and_preprocess_image(img_path) for img_path in x_batch]

            batches.append({
                'x_batch': x_batch,
                'y_batch': y_batch,
                'batch_idx': b_idx,
            })

        self.iter = iter(batches)

    def load_and_preprocess_image(self, img_path):
        # Construct the full path using the provided image directory
        full_path = os.path.join(self.image_dir, img_path)
        img = Image.open(full_path).convert('L')
        img = self.transform(img)
        return img

    def fetch_batch(self):
        if self.iter is None:
            self.generate_iter()

        batch = next(self.iter)

        if batch['batch_idx'] == self.num_batches_per_epoch - 1:
            self.generate_iter()

        if batch['y_batch'] is not None:
            batch['y_batch'] = torch.tensor(batch['y_batch'], dtype=torch.float32)

        return batch
