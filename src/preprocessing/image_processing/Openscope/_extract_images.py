import os

import matplotlib.pyplot as plt
import pandas as pd

# I dont think its needed anymore
import sys

sys.path.append(
    "content/openscope_databook/databook_utils"
)  # to make the import of utils files work

from dandi_utils import dandi_download_open
from matplotlib import cm
from tqdm import tqdm


def _stim_obj_to_table(nwb):
    all_labeled_stim_timestamps = []
    for stim_type, stim_obj in nwb.stimulus.items():
        start_times = stim_obj.timestamps[:-1]
        stop_times = stim_obj.timestamps[1:]
        frames = stim_obj.data[:-1]
        l = len(start_times)
        labeled_timestamps = list(zip(start_times, stop_times, frames, [stim_type] * l))
        all_labeled_stim_timestamps += labeled_timestamps

    all_labeled_stim_timestamps.sort(key=lambda x: x[0])
    stim_table = pd.DataFrame(
        all_labeled_stim_timestamps,
        columns=("start time", "stop time", "frame", "stimulus type"),
    )

    return stim_table


def _download_image(nwb, stimulus_type):
    stimulus = nwb.get_stimulus_template(stimulus_type).data
    return stimulus


def download_images(nwb, save_folder="src/data/AllenOpenScope/"):
    stim_table = _stim_obj_to_table(nwb)
    stim_all = [i for i in stim_table.iloc[:, -1].unique()]
    for stim_type in stim_all:
        print(f"Downloading {stim_type}")
        stim_folder = f"{save_folder}/{stim_type}"
        if not os.path.exists(stim_folder):
            os.mkdir(f"{save_folder}/{stim_type}")

        images = _download_image(nwb, stimulus_type=stim_type)
        for i, frame in tqdm(enumerate(images)):
            plt.imshow(frame, cmap="gray")
            plt.savefig(f"{stim_folder}/image%04d.png" % i)
            plt.close()


if __name__ == "__main__":
    data_dict = {
        "dandiset_id": "000036",
        "dandi_filepath": "sub-389014/sub-389014_ses-20180705T152908_behavior+image+ophys.nwb",
        "download_loc": "Allen/",
        "dandi_api_key": None,
    }
    io = dandi_download_open(**data_dict)
    nwb = io.read()
    download_images(nwb)
