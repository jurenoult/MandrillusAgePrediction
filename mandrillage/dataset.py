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
    "photo_name",
    "id",
    "sex",
    "dob",
    "dob_estimated",
    "error_dob",
    "faceview",
    "facequal",
    "shootdate",
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


def filter_by_sex(data, sex):
    return data[data["sex"] == sex]


def read_dataset(
    path,
    filter_dob_error=True,
    filter_certainty=False,
    max_age=0,
    max_dob_error=0,
    sex=None,
):
    data = pd.read_csv(
        path,
        dtype={
            "ShootDate": str,
            "shootdate": str,
            "Shootdate": str,
            "dob": str,
        },
    )
    # All columns to lowercase
    data.columns = data.columns.str.lower()

    data = data.drop("pos_pic", axis=1)

    data["shootdate"].replace("nan", np.nan, inplace=True)
    data["shootdate"].replace("#N/D", np.nan, inplace=True)
    data["dob"].replace("#N/D", np.nan, inplace=True)
    # data["shootdate"].replace("", np.nan, inplace=True)
    len_raw = len(data)
    data = data.dropna()
    len_filter_errors = len(data)
    print(f"Filtered #{len_filter_errors-len_raw} ({len_filter_errors}/{len_raw})")
    data["age"] = data.apply(compute_age, axis=1)

    if sex:
        assert (
            sex == "f" or sex == "m"
        ), "Expected sex to be either 'f' (female) or 'm' (male)"
        data = filter_by_sex(data, sex)
    if filter_certainty:
        data = filter_by_certainty(data, max_dob_error)
    if filter_dob_error:
        data = filter_dob_errors(data)

    if max_age > 0:
        data = filter_by_age(data, age_in_days=max_age)

    data.reset_index(drop=True, inplace=True)
    return data


def resample(df, bins):
    value_range = pd.cut(df["age"], bins)

    # Count the occurrences of each value range
    range_counts = value_range.value_counts()

    # Find the minimum count among the value ranges
    max_count = range_counts.max()

    # Filter the DataFrame to have the same number of occurrences for each value range
    filtered_df = df.groupby(value_range).apply(
        lambda x: x.sample(max_count, replace=True)
    )

    # Reset the index of the filtered DataFrame
    filtered_df = filtered_df.reset_index(drop=True)
    return filtered_df


import albumentations as A

AUGMENTATION_PIPELINE = A.Compose(
    [
        # A.Flip(p=0.5),
        # A.ShiftScaleRotate(
        #     p=0.5,
        #     shift_limit=0.0,
        #     scale_limit=0.95,
        #     rotate_limit=0,
        #     border_mode=cv2.BORDER_CONSTANT,
        # ),
        A.OneOf(
            [A.Blur(blur_limit=5, p=1.0), A.Defocus(alias_blur=(0.1, 0.2), p=1.0)],
            p=0.5,
        ),
        # A.CoarseDropout(max_holes=3, max_height=16, max_width=16, min_holes=1),
    ],
    p=0.5,
)


class AugmentedDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    @property
    def images(self):
        return self.subset.images

    def _augment(self, x):
        image = x.numpy()
        image = np.moveaxis(image, 0, -1)
        image = AUGMENTATION_PIPELINE(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        return image

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self._augment(x), y


class AugmentedSimilarityDataset(AugmentedDataset):
    def __init__(self, subset):
        super(AugmentedSimilarityDataset, self).__init__(subset)

    def __getitem__(self, idx):
        x1, x2 = self.subset[idx]
        return self._augment(x1), self._augment(x2)


class MandrillImageDataset(Dataset):
    def __init__(
        self,
        root_dir,
        dataframe,
        img_size=(224, 224),
        in_mem=True,
        max_days=1,
        individuals_ids=[],
        max_nbins=12,
        training=False,
    ):
        self.df = dataframe
        self.root_dir = root_dir
        self.img_size = img_size
        self.in_mem = in_mem
        self.max_days = max_days
        self.max_nbins = max_nbins
        self.training = training

        if len(individuals_ids) != 0:
            #     print("No individuals data specified, using all dataset")
            #     raise NotImplementedError
            # else:
            # Filter dataframe with id array
            self.df = self.df[self.df["id"].isin(individuals_ids)]
            self.df.reset_index(drop=True, inplace=True)

        if self.in_mem:
            self.images = []
            for i in tqdm(range(len(self.df))):
                row = self.df.iloc[[i]]
                self.images.append(self.load_photo(row))

        # if self.training:
        #     self.partition_by_age()

    def partition_by_age(self):
        # Split the max days into n_bins
        days_step = self.max_days / self.max_nbins
        # Init classes
        self.age_partitions = [[] for _ in range(self.max_nbins)]
        for i in tqdm(range(len(self.df))):
            row = self.df.iloc[[i]]
            # get current age
            age = row["age"].values[0]
            # Find which interval the age fits in
            age_index = int(age // days_step)
            self.age_partitions[age_index].append(i)

        # Filter empty bins
        self.age_partitions = [v for v in self.age_partitions if len(v) > 0]
        print(f"Partitionned data into {len(self.age_partitions)} partitions")
        print(
            f"Partitions size distribution: {', '.join([str(len(x)) for x in self.age_partitions])}"
        )

    def load_photo(self, row, normalize=True):
        image_path = self.photo_path(row)
        image = cv2.imread(image_path)
        if image.shape[0:2] != self.img_size:
            image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)

        # Normalization
        if normalize:
            image = image.astype(np.float32) / 255.0
            # image = (image - image.min()) / image.ptp()

        return image

    def value_to_str(self, value):
        if not isinstance(value, str):
            return value.values[0]
        return value

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

    def _getpair_from_row(self, row, idx=-1):
        target = row["age"]
        if not isinstance(target, int):
            target = float(row["age"].values[0])
        if self.max_days > 0:
            target = target / self.max_days

        if self.in_mem and idx >= 0:
            image = self.images[idx]
        else:
            image = self.load_photo(row)

        image = np.moveaxis(image, -1, 0).astype(np.float32)  # Channel first format

        return torch.tensor(image), torch.tensor(target)

    def _getpair(self, idx):
        # All datas for this mandrill
        row = self.df.iloc[[idx]]
        return self._getpair_from_row(row, idx)

    def set_images(self, images):
        self.images = images
        self.in_mem = True

    def __getitem__(self, idx):
        # if self.training:
        #     partition_index = idx % len(self.age_partitions)
        #     idx = random.choice(self.age_partitions[partition_index])
        return self._getpair(idx)


class MandrillSimilarityImageDataset(MandrillImageDataset):
    def __init__(
        self,
        root_dir,
        dataframe,
        img_size=(224, 224),
        in_mem=True,
        max_days=1,
        individuals_ids=[],
    ):
        super(MandrillSimilarityImageDataset, self).__init__(
            root_dir, dataframe, img_size, in_mem, max_days, individuals_ids
        )
        self.ids = individuals_ids

        self.valid_id_date = {}
        valid_frames = self.df.groupby(["id", "shootdate"])
        valid_frames = valid_frames.filter(lambda x: len(x) > 1)

        for i, row in valid_frames.iterrows():
            _id = row["id"]
            if _id not in self.valid_id_date:
                self.valid_id_date[_id] = []
            shootdate = row["shootdate"]
            if shootdate not in self.valid_id_date[_id]:
                self.valid_id_date[_id].append(shootdate)

    def get_photo_pair(self, idx, shootdate):
        candidates = self.df[
            (self.df["id"] == idx) & (self.df["shootdate"] == shootdate)
        ].index
        candidates = list(candidates)
        first_candidate_idx = random.randint(0, len(candidates) - 1)
        first_candidate = candidates.pop(first_candidate_idx)
        second_candidate_idx = random.randint(0, len(candidates) - 1)
        second_candidate = candidates.pop(second_candidate_idx)

        return first_candidate, second_candidate

    def __len__(self):
        return len(self.valid_id_date)

    def __getitem__(self, idx):
        _id = list(self.valid_id_date.keys())[idx]
        shootdates = self.valid_id_date[_id]
        shootdate = shootdates[random.randint(0, len(shootdates) - 1)]
        id1, id2 = self.get_photo_pair(_id, shootdate)

        x1, y1 = self._getpair(id1)
        x2, y2 = self._getpair(id2)

        assert abs(y1 - y2) < 1

        return x1, x2


class ClassificationMandrillImageDataset(MandrillImageDataset):
    def __init__(
        self,
        root_dir,
        dataframe,
        img_size=(224, 224),
        in_mem=True,
        n_classes=2,
        days_step=365,
        individuals_ids=[],
        return_integer=False,
    ):
        super(ClassificationMandrillImageDataset, self).__init__(
            root_dir=root_dir,
            dataframe=dataframe,
            img_size=img_size,
            in_mem=in_mem,
            max_days=1,
            individuals_ids=individuals_ids,
        )
        self.days_step = days_step
        self.n_classes = n_classes
        self.return_integer = return_integer

    def to_class(self, age):
        y_c = age / self.days_step
        y_c = max(0, np.ceil(y_c.numpy()) - 1)
        return int(y_c)

    def __getitem__(self, idx):
        x, age = self._getpair(idx)

        y_c = self.to_class(age)

        if self.return_integer:
            return x, y_c

        y = torch.zeros([self.n_classes])
        y[int(y_c)] = 1

        return x, y


class MandrillDualClassificationDataset(MandrillImageDataset):
    def __init__(
        self, root_dir, dataframe, img_size=(224, 224), device="cuda", in_mem=False, max_days=0
    ):
        super(MandrillDualClassificationDataset, self).__init__(
            root_dir=root_dir,
            dataframe=dataframe,
            img_size=img_size,
            in_mem=in_mem,
            max_days=max_days,
            individuals_ids=individuals_ids,
        )

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
        self, root_dir, dataframe, img_size=(224, 224), device="cuda", in_mem=False, max_days=0, individuals_ids=None
    ):
        super(MandrillTripleImageDataset, self).__init__(
            root_dir=root_dir,
            dataframe=dataframe,
            img_size=img_size,
            in_mem=in_mem,
            max_days=max_days,
            individuals_ids=individuals_ids,
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
