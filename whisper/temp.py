def process_DecodingResult(seek, previous_seek_value, result):
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
    return seek, previous_seek_value