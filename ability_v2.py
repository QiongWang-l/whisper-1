#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# import numpy as np
# import cv2
import base64
# import requests
# import json
# import time
import json
import os

from abc import ABC, abstractmethod
import logging
# from typing import Tuple, Union
# import numpy as np
from typing import List

from torch import Tensor

import whisper
import torch
from whisper import transcribe_

from whisper import DecodingTask
from whisper.utils import compression_ratio


class BaseModel(ABC):
    __name__ = ""
    extra_info = None

    def __init__(self, model_name: str):
        self.logger = logging.getLogger("infer_base").getChild(self.__name__)

    @abstractmethod
    def __call__(self):
        pass

    def set_extrainfo(self, extra_info):
        self.extra_info = extra_info


# bgm 推理服务
# class AudioEvent(BaseModel):
#     def __init__(self, config=None):
#         self.__name__ = "audioevent"
#         self.__sid__ = 9
#         self.__version__ = 1
#         self.infer_batch_size = 1
#         self.ability = Ability(config, self.__name__)
#         self.extra_info = {}
#
#     def __call__(self, audio_path: str):
#         # 使用本地文件base64加密后的item
#         # audio_path= "example-1.wav"
#         audio_byte = open(audio_path, "rb").read()
#         b64code = base64.b64encode(audio_byte)
#         item = f"data:audio/wav;base64,{b64code}"
#         response = self.ability.__call_base__(item, self.extra_info)
#
#         if response:
#             return response.data
#         else:
#             return None


# bgm 推理服务
def save_to_json(audio_path: str, result: dict):
    parent_dir = os.path.dirname(audio_path)
    audio_name = os.path.basename(audio_path)
    json_name = audio_name.split('.')[0] + '.json'
    json_path = os.path.join(parent_dir, json_name)

    json.dump(result, open(json_path, mode='w', encoding='utf-8'), ensure_ascii=False)


class Speech2Text(BaseModel):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_name: str, language='zh', config=None):
        super().__init__(model_name)
        self.__name__ = "speech2text"
        self.__version__ = 1
        # self.infer_batch_size = 1
        # self.ability = Ability(config, self.__name__)
        self.extra_info = {}

        self.model_name = model_name
        self.language = language
        self.whisper_model = whisper.load_model(self.model_name).to(self.device)
        self.decoding_options = whisper.DecodingOptions()
        print('load whisper model: {} \n'.format(self.model_name))

    def __call__(self, audio_path: str):
        # audio_path= "example-1.wav"

        with torch.no_grad():
            result = transcribe_.new_transcrebe(self.whisper_model, self.device, audio_path)

        # 保存为json
        save_to_json(audio_path, result)

        return result


if __name__ == "__main__":
    model = Speech2Text('medium')
    print(model('./whisper/test4.mp4'))
    # print(model('/home/mgtv/test_whisper/test1.mp4'))
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = whisper.load_model('medium').to(device)
    # res = model.transcribe('./whisper/test4.mp4')
    #
    # print(res)
