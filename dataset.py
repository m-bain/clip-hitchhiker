import os
import json
import torch
from torch.utils.data import Dataset
import pandas as pd
import decord
import numpy as np
from torchvision import transforms

DATA_ROOT = "./data"


class MSRVTT_dataset(Dataset):
    def __init__(self, preprocess, num_frames, split):
        annotation_fp = os.path.join(DATA_ROOT, "MSRVTT", "annotation", "MSR_VTT.json")

        if not os.path.isfile(annotation_fp):
            # download MSRVTT from public hosting repo https://github.com/m-bain/frozen-in-time
            os.system(
                "wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -P data; unzip data/MSRVTT.zip -d data")

        with open(annotation_fp, 'r') as fid:
            data = json.load(fid)

        df = pd.DataFrame(data['annotations'])

        test_list_path = "val_list_jsfusion.txt"
        js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"

        split_dir = os.path.join(DATA_ROOT, "MSRVTT", 'high-quality', 'structured-symlinks')

        if split != "test":
            raise NotImplementedError("Only supporting zero-shot eval currently")

        # keep only test set videos
        test_df = pd.read_csv(os.path.join(split_dir, test_list_path), names=['videoid'])
        df = df[df['image_id'].isin(test_df['videoid'])]
        self.metadata = df.groupby(['image_id'])['caption'].apply(list)

        # pick video-caption pairs according to 1k-A jsfusion split
        caps = pd.Series(np.load(os.path.join(split_dir, js_test_cap_idx_path), allow_pickle=True))
        new_res = pd.DataFrame({'caps': self.metadata, 'cap_idx': caps})
        new_res['test_caps'] = new_res.apply(lambda x: [x['caps'][x['cap_idx']]], axis=1)
        self.metadata = new_res['test_caps']
        self.metadata = pd.DataFrame({'captions': self.metadata})

        # remove to_rgb() and to tensor() so it works on Tensors
        preprocess = transforms.Compose([
            x for x in preprocess.transforms if isinstance(x, torch.nn.Module)
        ])
        self.preprocess = preprocess
        self.num_frames = num_frames
        self.split = split
        self.__getitem__(0)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        caption = sample['captions'][0]

        video_path = os.path.join(DATA_ROOT, "MSRVTT", 'videos', 'all', sample.name + '.mp4')
        frames = read_video_decord(video_path, num_frames=self.num_frames)
        frames = self.preprocess(frames)

        return {"text": caption, "video": frames}


def read_video_decord(video_path, num_frames):
    video_reader = decord.VideoReader(video_path, num_threads=0)
    available_frames = len(video_reader)

    intervals = np.linspace(start=0, stop=available_frames, num=num_frames + 1).astype(int)
    frame_ranges = [(interv, intervals[idx + 1]) for idx, interv in enumerate(intervals[:-1])]

    # equally chunk video into N segments where N = num_frames
    # sample frame at midpoint of each chunk
    frame_idxs = [(x[0] + x[1]) // 2 for x in frame_ranges]

    frames = video_reader.get_batch(frame_idxs)
    frames = torch.from_numpy(frames.asnumpy()).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames
