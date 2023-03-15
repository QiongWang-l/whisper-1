#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# import numpy as np
# import cv2
import base64
# import requests
# import json
# import time

from abc import ABC, abstractmethod
import logging
# from typing import Tuple, Union
# import numpy as np
from typing import List

from torch import Tensor

import whisper
import torch

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
class Speech2Text(BaseModel):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, model_name: str, language='zh', config=None):
        super().__init__(model_name)
        self.__name__ = "speech2text"
        self.__version__ = 1
        self.infer_batch_size = 1
        # self.ability = Ability(config, self.__name__)
        self.extra_info = {}

        self.model_name = model_name
        self.language = language
        self.whisper_model = whisper.load_model(self.model_name).to(self.device)
        print('load whisper')

    def __call__(self, audio_path: str):
        # audio_path= "example-1.wav"

        # 预处理
        # load audio and pad/trim it to fit 30 seconds
        with torch.no_grad():
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.device)

            # if single audio
            if mel.ndim == 2:
                mel = mel.unsqueeze(0)

            # tokenizer = whisper.get_tokenizer(self.whisper_model.is_multilingual,
            #                                              language=self.language, task="transcribe")

            options = whisper.DecodingOptions()
            # result = whisper.decode(self.whisper_model, mel, options)

            # result = whisper.DecodingTask(self.whisper_model, options).run(mel)  # 正确的
            # 前
            decoding_task = whisper.DecodingTask(self.whisper_model, options)
            audio_features, tokens, languages = self.get_init_tokens_and_feats(decoding_task, mel)
            # decode the audio

            tokens, sum_logprobs, no_speech_probs = decoding_task._main_loop(audio_features, tokens)

            # 后
            result = self.token_2_result(decoding_task, audio_features, languages,
                                         tokens, sum_logprobs, no_speech_probs, mel.shape[0])[0]

        # result = self.whisper_model.transcribe(audio_path)

        return result

    @torch.no_grad()
    def get_audio_features(self, model, options, mel: Tensor):
        if options.fp16:
            mel = mel.half()

        if mel.shape[-2:] == (model.dims.n_audio_ctx, model.dims.n_audio_state):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = model.encoder(mel)

        if audio_features.dtype != (torch.float16 if options.fp16 else torch.float32):
            return TypeError(f"audio_features has an incorrect dtype: {audio_features.dtype}")

        return audio_features

    @torch.no_grad()
    def get_init_tokens_and_feats(self, decoding_task, mel):
        decoding_task.decoder.reset()
        tokenizer = decoding_task.tokenizer
        n_audio = mel.shape[0]

        audio_features: Tensor = self.get_audio_features(decoding_task.model, decoding_task.options,
                                                         mel)  # encoder forward pass

        tokens: Tensor = torch.tensor([decoding_task.initial_tokens]).repeat(n_audio, 1)

        # detect language if requested, overwriting the language token
        languages, language_probs = decoding_task._detect_language(audio_features, tokens)
        if decoding_task.options.task == "lang_id":
            return [
                whisper.DecodingResult(audio_features=features, language=language, language_probs=probs)
                for features, language, probs in zip(audio_features, languages, language_probs)
            ]

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = audio_features.repeat_interleave(decoding_task.n_group, dim=0)
        tokens = tokens.repeat_interleave(decoding_task.n_group, dim=0).to(audio_features.device)
        return audio_features, tokens, languages

    @torch.no_grad()
    def token_2_result(self, decoding_task, audio_features, languages, tokens, sum_logprobs, no_speech_probs, n_audio):
        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: decoding_task.n_group]
        no_speech_probs = no_speech_probs[:: decoding_task.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, decoding_task.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, decoding_task.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = decoding_task.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[decoding_task.sample_begin: (t == decoding_task.tokenizer.eot).nonzero()[0, 0]] for t in s] for s in
            tokens
        ]

        # select the top-ranked sample in each group
        selected = decoding_task.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [decoding_task.tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            whisper.DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=decoding_task.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
        ]


if __name__ == "__main__":
    model = Speech2Text('medium')
    print(model.__name__)
    print(model('./whisper/test1.mp4'))
    # print(model('/home/mgtv/test_whisper/test1.mp4'))
