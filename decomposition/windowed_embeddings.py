'''Continuous windowed hidden-state extraction.

The model is applied to overlapping windows of input audio. Each window
contributes only its center frames, so every kept frame has at least
`context` seconds of audio on both sides. Windows start on the 20 ms frame
grid, so the center frames of consecutive windows join without gaps or
overlap on one global frame grid.
'''
from pathlib import Path

import numpy as np

import to_vector
from phraser import Store

from . import locations

SAMPLE_RATE = 16000
STRIDE = 320                       # samples per frame (20 ms)
FRAME_SECONDS = STRIDE / SAMPLE_RATE


def compute_windowed_embeddings(audio_filename, model_name, start,
    duration=180.0, layer=9, context=2.0, center=16.0, gpu=True,
    model=None):
    '''Return (embeddings, times) for duration seconds starting at start.
    audio_filename:  path to the 16 kHz audio file
    model_name:      local model name (see locations), Hugging Face id or
                     checkpoint path
    start:           start of the region in seconds, multiple of 0.02
    duration:        length of the region in seconds, multiple of 0.02
    layer:           hidden-state index (0 is the CNN/projection output)
    context:         seconds dropped on each side of a window
    center:          seconds kept from the middle of each window
    gpu:             whether to run the model on GPU
    model:           optional loaded model, skips loading model_name

    embeddings has shape (n_frames, dim); times holds each frame's start
    time in seconds. Requires context seconds of audio before start and
    after the last window's center.
    '''
    context_frames = seconds_to_frames(context, 'context')
    center_frames = seconds_to_frames(center, 'center')
    n_frames = seconds_to_frames(duration, 'duration')
    seconds_to_frames(start, 'start')
    if start < context:
        m = f'start {start} must be at least context {context} seconds'
        raise ValueError(m)
    window_frames = center_frames + 2 * context_frames
    n_windows = -(-n_frames // center_frames)
    load_start = start - context
    load_frames = (n_windows - 1) * center_frames + window_frames
    load_end = load_start + load_frames * FRAME_SECONDS
    audio = to_vector.load_audio(audio_filename, start=load_start,
        end=load_end)
    if model is None:
        path = locations.model_path(model_name)
        model = to_vector.load_model(str(path), gpu=gpu)
    embeddings = _window_centers(audio, model, n_frames, layer,
        context_frames, center_frames, gpu)
    times = start + np.arange(n_frames) * FRAME_SECONDS
    return embeddings, times


def _window_centers(audio, model, n_frames, layer, context_frames,
    center_frames, gpu):
    '''Run the model per window and concatenate the center frames.'''
    window_samples = (center_frames + 2 * context_frames) * STRIDE
    chunks = []
    for first_frame in range(0, n_frames, center_frames):
        offset = first_frame * STRIDE
        window = audio[offset:offset + window_samples]
        outputs = to_vector.audio_to_vector(window, model=model, gpu=gpu)
        hidden_states = outputs.hidden_states[layer]
        if hidden_states.ndim == 3: hidden_states = hidden_states[0]
        end = context_frames + center_frames
        chunks.append(hidden_states[context_frames:end])
    embeddings = np.concatenate(chunks)[:n_frames]
    if len(embeddings) < n_frames:
        m = f'audio too short: got {len(embeddings)} of {n_frames} frames, '
        m += 'the region plus context runs past the end of the file'
        raise ValueError(m)
    return embeddings


def seconds_to_frames(seconds, name='seconds'):
    '''Convert seconds to a frame count; must lie on the 20 ms grid.'''
    frames = round(seconds / FRAME_SECONDS)
    if abs(frames * FRAME_SECONDS - seconds) > 1e-9:
        m = f'{name} {seconds} is not a multiple of {FRAME_SECONDS} seconds'
        raise ValueError(m)
    return frames


def make_output_filename(audio_filename, start, duration, model_name, layer):
    '''Return e.g. fn001234_s-60_dur-180_<model>_l-9.npz.'''
    audio_name = Path(audio_filename).stem
    model = Path(str(model_name)).name
    filename = f'{audio_name}_s-{start:g}_dur-{duration:g}'
    filename += f'_{model}_l-{layer}.npz'
    return filename


def save_windowed_embeddings(embeddings, times, audio_filename, model_name,
    start, duration, layer, context, center, output_dir='.'):
    '''Save embeddings, frame times and extraction settings to an .npz.
    Returns the output path; the filename comes from make_output_filename.
    '''
    filename = make_output_filename(audio_filename, start, duration,
        model_name, layer)
    path = Path(output_dir) / filename
    np.savez(path, embeddings=embeddings, times=times,
        audio_filename=audio_filename, model_name=model_name, start=start,
        duration=duration, layer=layer, context=context, center=center)
    return path


def select_recording(store_path=locations.cgn_awd_lmdb,
    filename_contains='/comp-o/', min_seconds=200):
    '''Return the first audio filename matching filename_contains that is
    at least min_seconds long, or None.
    '''
    store = Store(store_path)
    audios = store.audios.filter(filename__contains=filename_contains)
    for audio in audios:
        if audio.duration / 1000 >= min_seconds: return audio.filename
    return None
