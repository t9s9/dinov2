import io
from pathlib import Path
from typing import Union, Callable, Optional, Tuple

import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class H5ClassificationDataset(Dataset):
    _SPLITS = ["train", "val", "test"]
    _H5_FILENAME = "{split}.h5"
    _MAPPER_FILENAME = "{split}_mapper.parquet"

    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            split: str = "train",
            driver: Optional[str] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        assert self.split in self._SPLITS

        h5_path = self.root / self._H5_FILENAME.format(split=self.split)
        if not h5_path.exists():
            raise FileNotFoundError(f"{h5_path} does not exists.")
        self.h5_file = h5py.File(h5_path, "r", driver=driver)

        self.mapper = None
        if "targets" not in self.h5_file.keys():
            self.mapper = pd.read_parquet(self.root / self._MAPPER_FILENAME.format(split=self.split))

    def __len__(self) -> int:
        return len(self.h5_file.get("images")) if self.mapper is None else len(self.mapper)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        if self.mapper is None:
            raw_image = self.h5_file.get("images")[idx]
            target = self.h5_file.get("targets")[idx]
        else:
            dp = self.mapper.loc[idx]
            raw_image = self.h5_file.get("images")[dp["h5_index"]]
            target = dp["target"]

        image = Image.open(io.BytesIO(raw_image)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, target


class CustomImgNetDataset(H5ClassificationDataset):
    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            split: str = "train",
            subset: Optional[str] = None
    ):
        super().__init__(root, transform, split, driver="core" if split == "train" and subset is not None else None)
        if subset == '1pct' and split == "train":
            subset_df = pd.read_csv("solo/data/dataset_subset/imagenet_1percent.txt", names=['filename'])
            self.mapper = self.mapper.query("filename.isin(@subset_df.filename)").copy().reset_index(drop=True)
            print("Using IN 1%")
        elif subset == '10pct' and split == "train":
            subset_df = pd.read_csv("solo/data/dataset_subset/imagenet_10percent.txt", names=['filename'])
            self.mapper = self.mapper.query("filename.isin(@subset_df.filename)").copy().reset_index(drop=True)
            print("Using IN 10%")
        elif subset == "imgnet100":
            with open(self.root / 'imagenet100_classes.txt') as f:
                imgnet100_classes = sorted(f.readline().strip().split())
            imgnet100_class_wn_2_class_index = {class_wn: class_index for class_index, class_wn in
                                                enumerate(imgnet100_classes)}

            self.mapper = self.mapper.query('wn_name in @imgnet100_classes').reset_index(drop=True)
            self.mapper['target'] = self.mapper['wn_name'].apply(lambda x: imgnet100_class_wn_2_class_index[x])

        self.n_classes = self.mapper['target'].nunique()
        self.target_2_class_name = self.mapper[['target', 'class_name']].drop_duplicates().set_index('target')[
            'class_name'].to_dict()
