'''PCA for increasing audio durations from a balanced speaker sample.

compute_speaker_pcas uses growing prefixes from one start point;
compute_segment_pcas uses consecutive non-overlapping segments.
'''
from math import ceil
import multiprocessing
from pathlib import Path
import traceback

import numpy as np
import to_vector

from . import locations, pca, selection, windowed_embeddings


def compute_speaker_pcas(output_dir='pca_outputs', n_male=10, n_female=10,
    minutes=range(1, 11), start=10.0, layer=9, context=2.0, center=16.0,
    model_name=locations.wav2vec2_nl1_name, store_path=locations.cgn_awd_lmdb,
    component='o', gpu=True, spawn=True, progress=True):
    '''Save PCA outputs for one distinct recording per selected speaker.
    output_dir:  directory for PCA .npz files; embeddings are never saved
    n_male:      number of distinct male speakers
    n_female:    number of distinct female speakers
    minutes:     increasing extraction durations in minutes
    start:       shared extraction start in seconds
    layer:       model hidden-state index
    context:     context on either side of each window in seconds
    center:      retained center of each window in seconds
    model_name:  local model name, Hugging Face id or checkpoint path
    store_path:  CGN phraser store path
    component:   CGN component letter
    gpu:         whether model inference uses a GPU
    spawn:       run all extraction and PCA in one disposable child process
    progress:    print worker progress in the calling session

    Returns saved paths. Each file includes speaker and gender metadata.
    Recordings are associated with a speaker, but may contain other speakers.
    '''
    durations = [float(value) * 60 for value in minutes]
    _validate_settings(durations, start, context, center, n_male, n_female)
    segments = [(0.0, duration) for duration in durations]
    settings = {'model_name': str(model_name), 'start': start, 'layer': layer,
        'context': context, 'center': center}
    return _run_segments(segments, max(durations), settings, output_dir,
        n_male, n_female, store_path, component, gpu, spawn, progress)


def compute_segment_pcas(output_dir='segment_pca_outputs', n_male=10,
    n_female=10, minutes=range(1, 6), total_minutes=10, start=10.0, layer=9,
    context=2.0, center=16.0, model_name=locations.wav2vec2_nl1_name,
    store_path=locations.cgn_awd_lmdb, component='o', gpu=True, spawn=True,
    progress=True):
    '''Save PCA outputs for consecutive segments of each selected recording.
    output_dir:     directory for PCA .npz files; embeddings are never saved
    n_male:         number of distinct male speakers
    n_female:       number of distinct female speakers
    minutes:        increasing segment lengths in minutes
    total_minutes:  length of the region split into segments
    start:          start of the region and of the first segment in seconds
    layer:          model hidden-state index
    context:        context on either side of each window in seconds
    center:         retained center of each window in seconds
    model_name:     local model name, Hugging Face id or checkpoint path
    store_path:     CGN phraser store path
    component:      CGN component letter
    gpu:            whether model inference uses a GPU
    spawn:          run all extraction and PCA in one disposable child process
    progress:       print worker progress in the calling session

    With the defaults, the region 10-610 s is split into 10-70, 70-130, ...
    (1 minute) up to 10-310, 310-610 (5 minutes). Only full segments are
    kept, so 3 minutes covers 10-550. Recordings are selected as in
    compute_speaker_pcas with the same total duration, so both experiments
    use the same files. Returns saved paths.
    '''
    durations = [float(value) * 60 for value in minutes]
    total_duration = float(total_minutes) * 60
    _validate_settings(durations, start, context, center, n_male, n_female)
    windowed_embeddings.seconds_to_frames(total_duration, 'total_minutes')
    if max(durations) > total_duration:
        raise ValueError('minutes must not exceed total_minutes')
    segments = make_segments(durations, total_duration)
    settings = {'model_name': str(model_name), 'start': start, 'layer': layer,
        'context': context, 'center': center}
    return _run_segments(segments, total_duration, settings, output_dir,
        n_male, n_female, store_path, component, gpu, spawn, progress)


def make_segments(durations, total_duration):
    '''Return (offset, duration) pairs of consecutive full-length segments.
    Offsets are in seconds relative to the region start; a leftover shorter
    than duration at the end of the region is dropped.
    '''
    segments = []
    for duration in durations:
        n_segments = int(total_duration // duration)
        for index in range(n_segments):
            segments.append((index * duration, duration))
    return segments


def _run_segments(segments, selection_duration, settings, output_dir, n_male,
    n_female, store_path, component, gpu, spawn, progress):
    '''Select recordings long enough for selection_duration and save PCAs.'''
    start, center = settings['start'], settings['center']
    n_windows = ceil(selection_duration / center)
    min_seconds = start + n_windows * center + settings['context']
    store = selection.load_store(store_path)
    audios = selection.filter_on_component(selection.load_audios(store),
        component=component)
    audios = selection.filter_on_duration(audios, min_seconds=min_seconds)
    speakers = selection.make_gender_speaker_audio_dict(audios, store)
    selected = _select_recordings(speakers, n_male, n_female)
    if not selected: return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _print_progress if progress else _silent_progress
    report(f'Selected {n_male} male and {n_female} female speakers')
    args = (selected, segments, settings, gpu, output_dir)
    if spawn:
        paths = _run_spawned(_run_experiment, args, report)
        report(f'Complete: {len(paths)} PCA files saved; worker exited')
    else:
        paths = _run_experiment(*args, report=report)
        report(f'Complete: {len(paths)} PCA files saved')
    return paths


def _print_progress(message):
    print(message, flush=True)


def _silent_progress(message):
    pass


def _run_spawned(function, args, report):
    '''Forward progress and reap the child, including on Ctrl-C.'''
    context = multiprocessing.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(writer, function, args))
    try:
        process.start()
        writer.close()
        return _receive_worker(reader, process, report)
    finally:
        if process.pid is not None:
            if process.is_alive(): process.terminate()
            process.join()
        reader.close()
        writer.close()


def _receive_worker(reader, process, report):
    '''Wait for messages without hiding worker errors or abnormal exits.'''
    while True:
        if reader.poll(0.2):
            try:
                kind, value = reader.recv()
            except EOFError:
                process.join()
                message = f'Experiment worker exited without results '
                message += f'(exit code {process.exitcode})'
                raise RuntimeError(message) from None
            if kind == 'progress': report(value)
            elif kind == 'error': raise RuntimeError(value)
            elif kind == 'result':
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f'Worker exit code {process.exitcode}')
                return value
        elif not process.is_alive():
            process.join()
            message = f'Experiment worker exited without results '
            message += f'(exit code {process.exitcode})'
            raise RuntimeError(message)


def _worker(connection, function, args):
    '''Send only progress, saved paths and tracebacks back to the parent.'''
    def report(message):
        connection.send(('progress', message))

    try:
        paths = function(*args, report=report)
        connection.send(('result', paths))
    except Exception:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


def _run_experiment(selected, segments, settings, gpu, output_dir, report):
    '''Load once and reuse each recording's extraction for all segments.'''
    report('Worker started; loading model')
    model_path = locations.model_path(settings['model_name'])
    model = to_vector.load_model(str(model_path), gpu=gpu)
    model.eval()
    paths = []
    for index, recording in enumerate(selected, start=1):
        gender, speaker_id, audio_filename = recording
        report(f'Recording {index}/{len(selected)}: {speaker_id} ({gender})')
        paths.extend(_recording_pcas(recording, segments, settings, model,
            gpu, output_dir, report))
        report(f'Recording {index}/{len(selected)} complete')
    return paths


def _recording_pcas(recording, segments, settings, model, gpu, output_dir,
    report):
    '''Extract up to the last segment end once and slice every segment.'''
    gender, speaker_id, audio_filename = recording

    def window_progress(completed, total):
        report(f'  Extraction: {completed}/{total} windows')

    duration = max(offset + length for offset, length in segments)
    embeddings, _ = windowed_embeddings.compute_windowed_embeddings(
        audio_filename, duration=duration, model=model, gpu=gpu,
        progress=window_progress, **settings)
    paths = []
    for index, (offset, length) in enumerate(segments, start=1):
        first = windowed_embeddings.seconds_to_frames(offset, 'offset')
        n_frames = windowed_embeddings.seconds_to_frames(length, 'duration')
        segment = embeddings[first:first + n_frames]
        path = _save_pca(segment, audio_filename, gender, speaker_id,
            settings['start'] + offset, length, settings, output_dir)
        paths.append(path)
        report(f'  PCA saved: {index}/{len(segments)} segments')
    return paths


def _validate_settings(durations, start, context, center, n_male, n_female):
    '''Reject invalid extraction settings before loading data or models.'''
    for count in (n_male, n_female):
        if not isinstance(count, int) or count < 0:
            raise ValueError('Speaker counts must be nonnegative integers')
    if not durations or any(value <= 0 for value in durations):
        raise ValueError('minutes must contain positive durations')
    if durations != sorted(set(durations)):
        raise ValueError('minutes must be strictly increasing')
    if context <= 0 or center <= 0 or start < context:
        raise ValueError('Require context > 0, center > 0 and start >= context')
    for name, value in (('start', start), ('context', context),
        ('center', center)):
        windowed_embeddings.seconds_to_frames(value, name)
    for duration in durations:
        windowed_embeddings.seconds_to_frames(duration, 'duration')


def _select_recordings(speakers, n_male, n_female):
    '''Choose distinct speakers and filenames in reproducible sorted order.'''
    selected = []
    used = set()
    for gender, count in (('male', n_male), ('female', n_female)):
        found = 0
        for speaker_id, audios in sorted(speakers.get(gender, {}).items()):
            if found == count: break
            filenames = sorted(str(audio.filename) for audio in audios)
            available = [name for name in filenames if name not in used]
            if not available: continue
            filename = available[0]
            used.add(filename)
            selected.append((gender, speaker_id, filename))
            found += 1
        if found < count:
            message = f'Need {count} {gender} speakers with distinct eligible '
            message += f'recordings; selected {found}'
            raise ValueError(message)
    return selected


def _save_pca(embeddings, audio_filename, gender, speaker_id, start,
    duration, settings, output_dir):
    '''Save only PCA results and metadata for one embedding segment.
    start is the segment start; settings['start'] is the extraction start.
    '''
    result = pca.compute_pca(embeddings)
    filename = windowed_embeddings.make_output_filename(audio_filename, start,
        duration, settings['model_name'], settings['layer'])
    path = output_dir / pca.make_pca_filename(filename)
    metadata = {**settings, 'start': start,
        'extraction_start': settings['start']}
    np.savez(path, **result, **metadata, audio_filename=audio_filename,
        duration=duration, speaker_id=speaker_id, gender=gender,
        n_frames=len(embeddings))
    return path
