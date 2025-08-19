import click
import pandas as pd
import shutil
from tqdm import tqdm
import os

def build_path(row):
    prefix = "MANDRILLS BKB_"
    indiv_id = str(row["id"])
    photo_path = row["photo_path"]
    photo_path = photo_path.split(prefix+f"{indiv_id}_")[-1]
    sub_path = os.path.join(indiv_id, photo_path)
    return indiv_id, sub_path

@click.command()
@click.option(
    "--val_csv",
    required=True,
    help="Path to the validation results in csv format.",
)
@click.option(
    "--im_folder",
    required=True,
    help="Path to the image folder.",
)
@click.option(
    "--output_folder",
    required=True,
    help="Path to the output folder where selected image will be placed.",
)
def main(val_csv, im_folder, output_folder):
    # Read csv
    df = pd.read_csv(val_csv, sep=",")

    # Create the output folder if it does not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # For each entry build the image path
    for _, row in tqdm(df.iterrows()):
        sub_dir, sub_path = build_path(row)

        # Make sure to create the same folder hierarchy in the output folder
        out_folder_path = os.path.join(output_folder, sub_dir)
        os.makedirs(out_folder_path, exist_ok=True)

        im_path = os.path.join(im_folder, sub_path)
        out_im_path = os.path.join(output_folder, sub_path)

        # print(f"Copying from {im_path} to {out_im_path}")
        # Copy the image to the output folder
        shutil.copy(im_path, out_im_path)

if __name__ == "__main__":
    main()
