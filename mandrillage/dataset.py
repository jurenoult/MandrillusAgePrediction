import pandas as pd
import datetime
import numpy as np
import os
from tqdm import tqdm
import cv2
import torch
import random
from enum import IntEnum

from torch.utils.data import Dataset

CSV_ROWS = [
    "Photo_Name",
    "Id",
    "Sex",
    "dob",
    "dob_estimated",
    "error_dob",
    "FaceView",
    "FaceQual",
    "Shootdate",
]


def csvdate_to_date(shoot_date):
    year, month, day = shoot_date.split("/")
    return datetime.date(int(year), int(month), int(day))


def compute_age(row):
    photo_date = csvdate_to_date(row["shootdate"])
    dob_date = csvdate_to_date(row["dob"])
    age = photo_date - dob_date
    return age.days


def add(row):
    return row[0] + row[1] + row[2]


def filter_by_age(data, age_in_days):
    return data[data["age"] <= age_in_days]


def filter_by_certainty(data, max_dob_error):
    return data[data["error_dob"] <= max_dob_error]


def filter_dob_errors(data):
    return data[data["age"] >= 0]


def read_dataset(path, filter_dob_error=True, filter_certainty=False, max_age=0, max_dob_error=0):
    data = pd.read_csv(path, dtype={"shootdate": str})
    data["shootdate"].replace("nan", np.nan, inplace=True)
    data["shootdate"].replace("#N/D", np.nan, inplace=True)
    data["dob"].replace("#N/D", np.nan, inplace=True)

    data = data.dropna()
    data["age"] = data.apply(compute_age, axis=1)

    if filter_certainty:
        data = filter_by_certainty(data, max_dob_error)
    if filter_dob_error:
        data = filter_dob_errors(data)

    if max_age > 0:
        data = filter_by_age(data, age_in_days=max_age)

    data.reset_index(drop=True, inplace=True)
    return data


def filter_by_qty(df, bins, qty_per_bin=20):
    value_range = pd.cut(df["age"], bins)

    # Count the occurrences of each value range
    range_counts = value_range.value_counts()

    # Find the minimum count among the value ranges
    min_count = range_counts.min()

    # Filter the DataFrame to have the same number of occurrences for each value range
    filtered_df = df.groupby(value_range).apply(
        lambda x: x.sample(qty_per_bin, replace=True)
    )

    # Reset the index of the filtered DataFrame
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


import albumentations as A

AUGMENTATION_PIPELINE = transform = A.Compose(
    [
        A.Flip(p=0.5),
        A.ShiftScaleRotate(p=0.5, shift_limit=0.01, scale_limit=0.2, rotate_limit=180),
    ],
    p=0.5,
)


class AugmentedDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        image = x.numpy()
        image = np.moveaxis(image, 0, -1)
        image = AUGMENTATION_PIPELINE(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        return image, y


class MandrillImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        dataframe,
        img_size=(224, 224),
        device="cuda",
        in_mem=True,
        max_days=1,
        compute_stats=False,
    ):
        self.df = dataframe
        self.root_dir = root_dir
        self.img_size = img_size
        self.in_mem = in_mem
        self.max_days = max_days

        self.mean = np.array([0.29712812, 0.35389232, 0.39265153])
        self.std = np.array([0.1428217, 0.14765707, 0.15931044])

        if self.in_mem:
            self.images = []
            for i in tqdm(range(len(self.df))):
                row = self.df.iloc[[i]]
                self.images.append(self.load_photo(row))
            if compute_stats:
                self.compute_mean_std()
                for i in tqdm(range(len(self.images))):
                    self.images[i] = (self.images[i] - self.mean) / (self.std + 1e-9)

    def compute_mean_std(self):
        mean = np.zeros(3)
        std = np.zeros(3)
        print("==> Computing mean and std..")
        for image in tqdm(self.images):
            for i in range(3):
                mean[i] += image[:, :, i].mean()
                std[i] += image[:, :, i].std()
        mean /= len(self.images)
        std /= len(self.images)
        print(mean, std)
        self.mean = mean
        self.std = std

    def load_photo(self, row):
        image_path = self.photo_path(row)
        image = cv2.imread(image_path)
        if image.shape[0:2] != self.img_size:
            image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        return image

    def photo_path(self, row):
        path = self.root_dir
        if "parent_folder" in row:
            path = os.path.join(path, self.value_to_str(row["parent_folder"]))
        path = os.path.join(path, f"{self.value_to_str(row['id'])}")
        path = os.path.join(path, f"{self.value_to_str(row['photo_name'])}")
        return path

    def value_to_str(self, value):
        if not isinstance(value, str):
            return value.values[0]
        return value

    def __len__(self):
        return len(self.df)

    def _getpair(self, idx):
        row = self.df.iloc[[idx]]

        target = float(row["age"].values[0])
        if self.max_days > 0:
            target = target / self.max_days

        if self.in_mem:
            image = self.images[idx]
        else:
            image = self.load_photo(row)
            image = (image - self.mean) / (self.std + 1e-9)

        image = np.moveaxis(image, -1, 0).astype(np.float32)  # Channel first format

        return torch.tensor(image), torch.tensor(target)

    def set_images(self, images):
        self.images = images
        self.in_mem = True

    def __getitem__(self, idx):
        return self._getpair(idx)


class ClassificationMandrillImageDataset(MandrillImageDataset):
    def __init__(
        self,
        root_dir,
        dataframe,
        img_size=(224, 224),
        device="cuda",
        in_mem=True,
        n_classes=2,
        days_step=365,
        compute_stats=False,
    ):
        super(ClassificationMandrillImageDataset, self).__init__(
            root_dir=root_dir,
            dataframe=dataframe,
            img_size=img_size,
            device=device,
            in_mem=in_mem,
            max_days=1,
        )
        self.days_step = days_step
        self.n_classes = n_classes

    def __getitem__(self, idx):
        x, age = self._getpair(idx)

        y_c = age / self.days_step
        y_c = max(0, np.ceil(y_c.numpy()) - 1)

        y = torch.zeros([self.n_classes])
        y[int(y_c)] = 1

        return x, y


<<<<<<< HEAD
class MandrillDualClassificationDataset(MandrillImageDataset):
    def __init__(
        self, root_dir, dataframe, img_size=(224, 224), device="cuda", in_mem=False, max_days=0
=======
class RankingClass(IntEnum):
    NONE = 0
    SAME_MANDRILL_YOUNGER = 1
    SAME_MANDRILL_SAME_AGE = 2
    SAME_MANDRILL_OLDER = 3
    DIFF_MANDRILL_YOUNGER = 4
    DIFF_MANDRILL_SAME_AGE = 5
    DIFF_MANDRILL_OLDER = 6


class MandrillDualClassificationDataset(MandrillImageDataset):
    def __init__(
        self,
        root_dir,
        dataframe,
        img_size=(224, 224),
        device="cuda",
        in_mem=False,
        max_days=0,
        same_age_gap=10,
        n_classes=3,
>>>>>>> b775df2 (fix: issue with random choice with enum and class integer with different mandrills)
    ):
        super(MandrillDualClassificationDataset, self).__init__(
            root_dir=root_dir,
            dataframe=dataframe,
            img_size=img_size,
            device=device,
            in_mem=in_mem,
            max_days=max_days,
        )
<<<<<<< HEAD
=======
        self.same_age_gap = same_age_gap
        self.n_classes = n_classes

        assert (
            n_classes == 3 or n_classes == 6
        ), f"Expected n_classes to be one of [3,6] but found {n_classes}"

        # Build an index to get directly classes for a given photo
        self.indexes = {}

        for i, current_row in tqdm(self.df.iterrows(), total=len(self.df)):
            index = self.df.apply(
                lambda row: self.compute_class(row, current_row, self.same_age_gap),
                axis=1,
            )
            target_indexes = {}
            for j in range(1, 7):
                target_indexes[j] = index[index == j].index.tolist()
            self.indexes[i] = target_indexes

    def compute_class(self, row, target_row, margin):
        """Compute the class given two row sample.

        class 0: same row => ignore

        # Intra mandrill
        class 1: same mandrill younger
        class 2: same mandrill same age
        class 3: same mandrill older

        # Inter mandrill
        class 4: different mandrill younger
        class 5: different mandrill same age
        class 6: different mandrill older
        """
        if row["Photo_Name"] == target_row["Photo_Name"]:
            return 0

        age_class = self.compute_age_class(row["age"], target_row["age"], margin)

        # Same mandrill
        if row["Id"] == target_row["Id"]:
            return age_class

        # Two different mandrills so shift class
        return age_class + 3

    def compute_age_class(self, age, target_age, margin):
        diff = target_age - age
        if abs(diff) <= margin:
            return 2
        elif diff < 0:
            return 1
        else:
            return 3

    def random_same_mandrill(self):
        return random.random() > 0.5

    def add_list_class(self, dic, index, value):
        if len(index[value]) > 0:
            dic[value] = index[value]

    def get_mandril(self, idx, classes):
        # Get all possible samples
        indexes = self.indexes[idx]
        possible_classes = {}
        for _class in classes:
            self.add_list_class(possible_classes, indexes, _class)

        if len(possible_classes) == 0:
            return None, None

        c = random.choices(list(possible_classes.keys()))[0]
        target_list = possible_classes[c]
        target_sample_idx = target_list[random.randint(0, len(target_list) - 1)]
        x, y = self._getpair(target_sample_idx)

        return x, c

    def get_same_mandrill(self, idx):
        classes = [
            RankingClass.SAME_MANDRILL_YOUNGER,
            RankingClass.SAME_MANDRILL_SAME_AGE,
            RankingClass.SAME_MANDRILL_OLDER,
        ]
        return self.get_mandril(idx, classes)

    def get_different_mandrill(self, idx):
        classes = [
            RankingClass.DIFF_MANDRILL_YOUNGER,
            RankingClass.DIFF_MANDRILL_SAME_AGE,
            RankingClass.DIFF_MANDRILL_OLDER,
        ]
        x, c = self.get_mandril(idx, classes)

        # Shift classes to the same mandrill classes
        if self.n_classes == 3:
            c = c - 3
        return x, c
>>>>>>> b775df2 (fix: issue with random choice with enum and class integer with different mandrills)

    def __getitem__(self, idx):
        # Randomize
        idx = random.randint(0, len(self) - 1)
        x1, y1 = self._getpair(idx)

        second_idx = random.randint(0, len(self) - 1)
        x2, y2 = self._getpair(second_idx)

        # Don't allow same age mandrill comparison
        while y1 - y2 == 0:
            second_idx = random.randint(0, len(self) - 1)
            x2, y2 = self._getpair(second_idx)

        sign = torch.sign(y1 - y2)
        y = torch.zeros([3])
        y[int(sign) + 1] = 1
        return [x1, x2], y


class MandrillTripleImageDataset(MandrillImageDataset):
    def __init__(
        self, root_dir, dataframe, img_size=(224, 224), device="cuda", in_mem=False, max_days=0
    ):
        super(MandrillTripleImageDataset, self).__init__(
            root_dir=root_dir,
            dataframe=dataframe,
            img_size=img_size,
            device=device,
            in_mem=in_mem,
            max_days=max_days,
        )

    def compute_margin(self, y):
        y1, y2, y3 = y
        d1 = abs(y1 - y2)
        d2 = abs(y1 - y3)
        m = abs(d1 - d2)
        return m

    def __getitem__(self, idx):
        # Get three random idx
        idx2 = random.randint(0, len(self) - 1)
        idx3 = random.randint(0, len(self) - 1)
        x1, y1 = self._getpair(idx)
        x2, y2 = self._getpair(idx2)
        x3, y3 = self._getpair(idx3)

        y2y1 = abs(y1 - y2)
        y3y1 = abs(y1 - y3)

        if y2y1 > y3y1:
            triplet = ((x1, x3, x2), (y1, y3, y2))
        else:
            triplet = ((x1, x2, x3), (y1, y2, y3))
        x, y = triplet
        margin = self.compute_margin(y)
        return x, margin
