import io
import logging
import os
import random

import h5py
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

logger = logging.getLogger("dinov2")
_Target = int


class Ego4d(Dataset):
    corrupted = [(24, 14), (60, 16), (61, 13), (64, 12), (65, 9), (40, 8)]
    readded = [71, 56, 67, 74]

    def __init__(self, data_root, transform, gaze_size=224, time_window=15, center_crop=False, resize_gs=False,
                 **kwargs):
        super().__init__()
        assert gaze_size in [112, 114, 160, 224, 313, 336, 440, 448, 540]

        self.data_root = data_root
        self.transform = transform
        self.time_window = time_window
        self.center_crop = center_crop
        self.gaze_size = gaze_size
        self.resize_gs = resize_gs

        self.hdf5_file = h5py.File(os.path.join(self.data_root, f"data_all95.h5"), "r")
        self.dataset = h5py.File(os.path.join(self.data_root, f"dataset_all95.h5"), "r")["data"]

        b = np.ones((self.dataset.shape[0],), dtype=bool)
        for c in self.corrupted:
            b = b & ((self.dataset[:, 5] != c[0]) | (self.dataset[:, 11] != c[1]))
        self.dataset = self.dataset[b]

        self.gaze_index = (9, 10)
        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def open_image(self, row):
        index, number, partition = int(row[6]), int(row[11]), str(int(row[5]))

        gaze_size = self.gaze_size
        if gaze_size == -1:
            gaze_size = random.choice([114, 160, 224, 313, 439, 540])

        binimg = self.hdf5_file.get(partition).get("frames").get(f"images540_{str(number)}")[index]
        img = Image.open(io.BytesIO(binimg))

        if self.center_crop:
            img = torchvision.transforms.functional.center_crop(img, (self.gaze_size, self.gaze_size))
        elif gaze_size == 540:
            if self.resize_gs:
                img = torchvision.transforms.functional.resize(img, 224, InterpolationMode.BICUBIC)
        else:
            gaze_x, gaze_y = row[self.gaze_index[0]], row[self.gaze_index[1]]
            ### We control the gaze the boundaries of the gaze to not go beyond the image boundaries
            gaze_x += - max(0, gaze_x + gaze_size // 2 - 540) - min(0, gaze_x - gaze_size // 2)
            gaze_y += - max(0, gaze_y + gaze_size // 2 - 540) - min(0, gaze_y - gaze_size // 2)

            img = torchvision.transforms.functional.crop(img,
                                                         gaze_y - gaze_size // 2,
                                                         gaze_x - gaze_size // 2,
                                                         gaze_size,
                                                         gaze_size,
                                                         )
        return img

    def __getitem__(self, idx):
        self.idx = idx
        r = self.dataset[idx]
        image, video_name = self.open_image(r), r[0]

        if self.time_window == 0:
            return self.transform(image, image), -1

        new_video_name, new_idx, try_cpt = "", idx, 0
        while video_name != new_video_name:
            new_idx = idx + random.randint(-self.time_window, self.time_window)
            new_idx = max(0, min(new_idx, self.size - 1))
            if try_cpt > 5:
                new_idx = idx
            rn = self.dataset[new_idx]
            new_video_name = rn[0]
            try_cpt += 1

        image_pair = self.open_image(rn) if new_idx != idx else image
        return self.transform(image, image_pair), -10
