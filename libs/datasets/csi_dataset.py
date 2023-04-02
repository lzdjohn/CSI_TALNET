# python ./train_csi.py ./configs/csi.yaml --output reproduce
import os
import json
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from .data_utils import trivial_batch_collator, worker_init_reset_seed
from .datasets import register_dataset
from .data_utils import truncate_feats

def fft_process(signal):
    fft_signal = np.fft.fft(signal)
    fft_signal[100:] = 0
    ifft_signal = np.fft.ifft(fft_signal).astype("float32")
    return ifft_signal

@register_dataset("csi")
class CSIDataset(Dataset):
    def __init__(
        self,
        # feat_folder,
        # json_file,
        # test_feat_folder,
        # test_json_file,
        
        # trunc_thresh,
        # crop_ratio,
        # split,
        # default_fps=50,
        # input_dim=30,
        # max_seq_len=8500,
        # num_classes=7,
        # preprocess=True,
        # is_training = True,
        # feat_stride = 4,
        # downsample_rate = 1,
        

        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,
        file_ext,
        ):
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file
        
        # split / training mode
        self.split = split
        self.is_training = is_training
        self.preprocess = True
           
        # features meta info           
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.crop_ratio = crop_ratio
           
        # load database and select the subset           
        dict_db, label_dict = self._load_json_annotation(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict


        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict
        
        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'csi',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

        
    def _load_json_annotation(self, json_file):
        # 打开json标注文件
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data["database"]
        
        # 重建标签字典
        label_dict = {"run":0, "walk":1, "jump":2, "wave":3, "bend":4, "stand":5, "sit":6}
        
        dict_db = tuple()
        for video_name, value in json_db.items():
            vv = value["annotations"]
            segments, labels = [], []
            for i in range(len(vv)):
                segments.append(vv[i]["segment"])
                labels.append(label_dict[vv[i]["label"]])

            segments = np.asarray(segments, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int64)
            dict_db += ({'id': video_name,
                        'segments' : segments,
                        'labels' : labels
            }, )
        
        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        signal_dict = self.data_list[idx]
        
        # 加载特征
        filename = os.path.join(self.feat_folder, signal_dict['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)
        
        # 预处理
        if self.preprocess is True:
            feats = feats[::self.downsample_rate, :]
            feat_stride = self.feat_stride * self.downsample_rate
            feat_offset = 0
            # feat_offset = 0.5 * self.num_frames / feat_stride
            # feats = fft_process(feats)
            
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        if signal_dict['segments'] is not None:
            segments = torch.from_numpy(signal_dict['segments']/feat_stride - feat_offset)
            labels = torch.from_numpy(signal_dict['labels'])
        else:
            segments, labels = None, None
        
        
        data_dict = {'video_id'        : signal_dict['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,      # N
                     'fps'             : self.default_fps,
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : 8500,
                     'duration'        : 170
                    }
        
        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
            )
        
        return data_dict
    
# def main():
#     csi_dataset = CSIDataset(feat_folder=feat_folder, json_file=json_file)
#     dataloader = torch.utils.data.DataLoader(
#         csi_dataset,
#         batch_size=3,
#         num_workers=0,
#         collate_fn=trivial_batch_collator,
#         shuffle=True,
#         drop_last=True,
#     )
#     for batch_idx, data_dict in enumerate(dataloader):
#         print(batch_idx, data_dict)
#     return

# feat_folder = "/home/lzdjohn/actionformer/csi_signal/csi_signal/validation_npy/"
# json_file = "/home/lzdjohn/actionformer/csi_signal/csi_annotations/val_gt.json"
# if __name__=="main":
#     main()



# label_dict = {"run":1, "walk":2, "jump":3, "wave":4, "bend":5, "stand":6, "sit":7}
# with open("/home/lzdjohn/actionformer/csi_signal/csi_annotations/val_gt.json", 'r') as fid:
#     json_data = json.load(fid)
#     json_db = json_data["database"]
#     for key, value in json_db.items():
#         video_name = key
#         vv = value["annotations"]
#         segments, labels = [], []
#         for i in range(len(vv)):
#             segments.append(vv[i]["segment"])
#             labels.append(label_dict[vv[i]["label"]])

#         segments = np.asarray(segments, dtype=np.float32)
#         labels = np.asarray(labels, dtype=np.int64)
#         print(key)
#         print(segments)
#         print(labels)