"""
### new transcribe for AudioEvent
"""
import warnings
from typing import Optional, Tuple, Union, TYPE_CHECKING, List

import numpy as np
import torch
import tqdm
import time

from torch import Tensor

from .audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
from .decoding import DecodingOptions, DecodingResult
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import exact_div, format_timestamp, make_safe, optional_int, optional_float, str2bool, get_writer, \
    compression_ratio

if TYPE_CHECKING:
    from .model import Whisper

model: "Whisper"
# audio: Union[str, np.ndarray, torch.Tensor]
verbose: Optional[bool] = None
temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
compression_ratio_threshold: Optional[float] = 2.4
logprob_threshold: Optional[float] = -1.0
no_speech_threshold: Optional[float] = 0.6
condition_on_previous_text: bool = True
initial_prompt: Optional[str] = None


def get_audio_features(model, options, mel: Tensor):
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
def get_init_tokens_and_feats(decoding_task, mel):
    decoding_task.decoder.reset()
    tokenizer = decoding_task.tokenizer
    n_audio = mel.shape[0]

    audio_features: Tensor = get_audio_features(decoding_task.model, decoding_task.options,
                                                     mel)  # encoder forward pass

    tokens: Tensor = torch.tensor([decoding_task.initial_tokens]).repeat(n_audio, 1)

    # detect language if requested, overwriting the language token
    languages, language_probs = decoding_task._detect_language(audio_features, tokens)
    if decoding_task.options.task == "lang_id":
        return [
            DecodingResult(audio_features=features, language=language, language_probs=probs)
            for features, language, probs in zip(audio_features, languages, language_probs)
        ]

    # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
    audio_features = audio_features.repeat_interleave(decoding_task.n_group, dim=0)
    tokens = tokens.repeat_interleave(decoding_task.n_group, dim=0).to(audio_features.device)
    return audio_features, tokens, languages


@torch.no_grad()
def token_2_result(decoding_task, audio_features, languages, tokens, sum_logprobs, no_speech_probs, n_audio):
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
        DecodingResult(
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


def preprocess(audio: Union[str, np.ndarray, torch.Tensor], decode_options):
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    mel = log_mel_spectrogram(audio)

    # 如果没有指定语言，需要检测语言
    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            # 分割成固定大小的帧片段
            segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
            # 检测语言类型
            _, probs = model.detect_language(segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)

    return mel, tokenizer, input_stride, time_precision, mel.shape[-1], dtype


def decode_with_fallback(segment: torch.Tensor, decode_options) -> DecodingResult:
    temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
    decode_result = None

    for t in temperatures:
        kwargs = {**decode_options}
        if t > 0:
            # disable beam_size and patience when t > 0
            kwargs.pop("beam_size", None)
            kwargs.pop("patience", None)
        else:
            # disable best_of when t == 0
            kwargs.pop("best_of", None)

        options = DecodingOptions(**kwargs, temperature=t)
        # global all_real_decode_time
        # st1_time = time.time()
        decode_result = model.decode(segment, options)
        audio_features, tokens, languages = get_init_tokens_and_feats(decoding_task, mel)
        # decode the audio

        tokens, sum_logprobs, no_speech_probs = decoding_task._main_loop(audio_features, tokens)

        # 后
        result = self.token_2_result(decoding_task, audio_features, languages,
                                     tokens, sum_logprobs, no_speech_probs, mel.shape[0])


        # ed1_time = time.time()
        # all_real_decode_time += (ed1_time - st1_time)

        needs_fallback = False
        if compression_ratio_threshold is not None and decode_result.compression_ratio > compression_ratio_threshold:
            needs_fallback = True  # too repetitive
        if logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold:
            needs_fallback = True  # average log probability is too low

        if not needs_fallback:
            break

    return decode_result


def add_segment(tokenizer, all_segments, seek,  # 新增的
                start: float, end: float, text_tokens: torch.Tensor, result: DecodingResult):
    text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
    if len(text.strip()) == 0:  # skip empty text output
        return

    all_segments.append(
        {
            "id": len(all_segments),
            "seek": seek,
            "start": start,
            "end": end,
            "text": text,
            "tokens": text_tokens.tolist(),
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }
    )
    if verbose:
        print(make_safe(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"))

    return all_segments


def process_decoding_result(
        seek, previous_seek_value, result, segment, timestamp_offset, input_stride,
        time_precision, all_tokens, all_segments, segment_duration, tokenizer, prompt_reset_since):
    tokens = torch.tensor(result.tokens)

    if no_speech_threshold is not None:
        # no voice activity check
        should_skip = result.no_speech_prob > no_speech_threshold
        if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
            # don't skip if the logprob is high enough, despite the no_speech_prob
            should_skip = False

        if should_skip:
            seek += segment.shape[-1]  # fast-forward to the next segment boundary
            return seek, previous_seek_value

    timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
    consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
    if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
        last_slice = 0
        for current_slice in consecutive:
            sliced_tokens = tokens[last_slice:current_slice]
            start_timestamp_position = (
                    sliced_tokens[0].item() - tokenizer.timestamp_begin
            )
            end_timestamp_position = (
                    sliced_tokens[-1].item() - tokenizer.timestamp_begin
            )
            all_segments = add_segment(
                tokenizer, all_segments, seek,
                start=timestamp_offset + start_timestamp_position * time_precision,
                end=timestamp_offset + end_timestamp_position * time_precision,
                text_tokens=sliced_tokens[1:-1],
                result=result,
            )
            last_slice = current_slice
        last_timestamp_position = (
                tokens[last_slice - 1].item() - tokenizer.timestamp_begin
        )
        seek += last_timestamp_position * input_stride
        all_tokens.extend(tokens[: last_slice + 1].tolist())
    else:
        duration = segment_duration
        timestamps = tokens[timestamp_tokens.nonzero().flatten()]
        if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
            # no consecutive timestamps but it has a timestamp; use the last one.
            # single timestamp at the end means no speech after the last timestamp.
            last_timestamp_position = timestamps[-1].item() - tokenizer.timestamp_begin
            duration = last_timestamp_position * time_precision

        all_segments = add_segment(
            tokenizer, all_segments, seek,
            start=timestamp_offset,
            end=timestamp_offset + duration,
            text_tokens=tokens,
            result=result,
        )

        seek += segment.shape[-1]
        all_tokens.extend(tokens.tolist())

    if not condition_on_previous_text or result.temperature > 0.5:
        # do not feed the prompt tokens if a high temperature was used
        prompt_reset_since = len(all_tokens)

    previous_seek_value = seek
    return seek, previous_seek_value, all_tokens, all_segments, prompt_reset_since


def process(mel, tokenizer, input_stride, time_precision, num_frames, dtype, decode_options):
    seek = 0  # 当前处理的语音片段在原始语音信号中的起始位置（单位为帧）

    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    print('******************num_frames:{}'.format(num_frames))
    previous_seek_value = seek

    start_pro_time = time.time()
    all_decode_time = 0
    with tqdm.tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
        while seek < num_frames:
            # 当前处理片段的偏置
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, seek:], N_FRAMES).to(model.device).to(dtype)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            st_time = time.time()

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(segment, decode_options)

            ed_time = time.time()
            all_decode_time += (ed_time - st_time)

            seek, previous_seek_value, all_tokens, all_segments, prompt_reset_since = \
                process_decoding_result(seek, previous_seek_value, result, segment, timestamp_offset, input_stride,
                                        time_precision, all_tokens, all_segments, segment_duration, tokenizer, prompt_reset_since)  # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)

    end_pro_time = time.time()
    all_pro_time = end_pro_time - start_pro_time
    print('process spent {} s. \n'.format(all_pro_time))
    print('decode spent {} s. \n'.format(all_decode_time))

    return all_segments


def postprocess(all_segments):
    # print(all_segments)
    text = ''
    for segment in all_segments:
        text += segment['text']
        text += ' | '

    text = text[:-3]

    dic = dict(
        text=text,
        segments=all_segments
    )

    # json.dump(dic, output_dir, ensure_ascii=False)
    # json.dump(dic, open('./test1.json', mode='w', encoding='utf-8'), ensure_ascii=False)

    return dic


def new_transcrebe(trans_model: "Whisper", audio_input, **decode_options):
    global model
    model = trans_model
    # preprocess
    mel, tokenizer, input_stride, time_precision, num_frames, dtype = \
        preprocess(audio_input, decode_options)

    # process
    all_segments = process(mel, tokenizer, input_stride, time_precision, num_frames, dtype, decode_options)

    # postprocess
    res_dic = postprocess(all_segments)

    return res_dic


def transcribe(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor],
        *,
        verbose: Optional[bool] = None,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool 是否显示正在解码的文本到控制台。如果为 True，则显示所有详细信息,如果为 False，则显示最小的细节; 如果为 None，则不显示任何内容
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    mel = log_mel_spectrogram(audio)

    # 如果没有指定语言，需要检测语言
    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print("Detecting language using up to the first 30 seconds. Use `--language` to specify the language")
            # 分割成固定大小的帧片段
            segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
            # 检测语言类型
            _, probs = model.detect_language(segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(f"Detected language: {LANGUAGES[decode_options['language']].title()}")

    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
        temperatures = [temperature] if isinstance(temperature, (int, float)) else temperature
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            # global all_real_decode_time
            # st1_time = time.time()
            decode_result = model.decode(segment, options)
            # ed1_time = time.time()
            # all_real_decode_time += (ed1_time - st1_time)

            needs_fallback = False
            if compression_ratio_threshold is not None and decode_result.compression_ratio > compression_ratio_threshold:
                needs_fallback = True  # too repetitive
            if logprob_threshold is not None and decode_result.avg_logprob < logprob_threshold:
                needs_fallback = True  # average log probability is too low

            if not needs_fallback:
                break

        return decode_result

    seek = 0  # 当前处理的语音片段在原始语音信号中的起始位置（单位为帧）
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def add_segment(
            *, start: float, end: float, text_tokens: torch.Tensor, result: DecodingResult
    ):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": text_tokens.tolist(),
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if verbose:
            print(make_safe(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"))

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    print('******************num_frames:{}'.format(num_frames))
    previous_seek_value = seek

    start_pro_time = time.time()
    all_decode_time = 0
    with tqdm.tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
        while seek < num_frames:

            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, seek:], N_FRAMES).to(model.device).to(dtype)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            st_time = time.time()
            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(segment)

            ed_time = time.time()
            cost_time = ed_time - st_time
            all_decode_time += cost_time

            tokens = torch.tensor(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                            sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                            sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    add_segment(
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=timestamp_offset + end_timestamp_position * time_precision,
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = timestamps[-1].item() - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                add_segment(
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek

    end_pro_time = time.time()
    all_pro_time = end_pro_time - start_pro_time
    print('process spent {} s. \n'.format(all_pro_time))
    print('decode spent {} s. \n'.format(all_decode_time))
    # print('really decode spent {} s. \n'.format(all_real_decode_time))

    # print('all_tokens***********************************')
    # print(all_tokens)
    #
    # text = ''
    # for segment in all_segments:
    #     text += segment.text
    #     text += ' | '
    # text -= ' | '
    #
    # return dict(
    #     text=text,
    #     segments=all_segments,
    #     language=language
    # )

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens):]),
        segments=all_segments,
        language=language
    )

