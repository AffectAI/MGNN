# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np
from typing import Callable, List, Optional, Union

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class AffWildDataset(BaseActionDataset):


    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: ConfigType = dict(img=''),
                 filename_tmpl: str = 'img_{:05}.jpg',
                 with_offset: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 1,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 label_type: str = 'int',
                 timestamp_start: int = 1,
                 fps: int = 1,
                 multilabel: bool = True,
                 **kwargs) -> None:
        self._FPS = fps  # Keep this as standard
        self.timestamp_start = timestamp_start
        self.multilabel = multilabel
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        self.label_type = label_type
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()

            video_id = line_split[0]
            total_frames = int(line_split[1])
            timestamp = int(line_split[2])  # count by second or frame.
            img_key = f'{video_id},{timestamp:05d}'
            shot_info = (0, total_frames)

            frame_dir = video_id
            if self.data_prefix['img'] is not None:
                frame_dir = osp.join(self.data_prefix['img'], frame_dir)

            if self.label_type == 'int':
                label = [int(x) for x in line_split[3:]]
            elif self.label_type == 'float':
                label = [x for x in line_split[3:]]
            else:
                label = [int(x) for x in line_split[3:]]

            video_info = dict(
                frame_dir=frame_dir,
                video_id=video_id,
                timestamp=int(timestamp),
                img_key=img_key,
                shot_info=shot_info,
                fps=self._FPS,
                label=label)

            # add fake label for inference datalist without label
            if not label:
                label = [-1]
            if self.multi_class:
                assert self.num_classes is not None
                video_info['label'] = label
            else:
                assert len(label) == 1
                video_info['label'] = label[0]

            data_list.append(video_info)

        return data_list

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        data_info = super().get_data_info(idx)
        data_info['filename_tmpl'] = self.filename_tmpl
        data_info['timestamp_start'] = self.timestamp_start

        return data_info
