'''PCA for increasing audio durations from a balanced speaker sample.'''
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
    min_seconds = start + ceil(max(durations) / center) * center + context
    store = selection.load_store(store_path)
    audios = selection.filter_on_component(selection.load_audios(store),
        component=component)
    audios = selection.filter_on_duration(audios, min_seconds=min_seconds)
    speakers = selection.make_gender_speaker_audio_dict(audios, store)
    selected = _select_recordings(speakers, n_male, n_female)
    if not selected: return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {'model_name': str(model_name), 'start': start, 'layer': layer,
        'context': context, 'center': center}
    report = _print_progress if progress else _silent_progress
    report(f'Selected {n_male} male and {n_female} female speakers')
    args = (selected, durations, settings, gpu, output_dir)
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


def _run_experiment(selected, durations, settings, gpu, output_dir, report):
    '''Load once and reuse each recording's longest extraction for all PCAs.'''
    report('Worker started; loading model')
    model_path = locations.model_path(settings['model_name'])
    model = to_vector.load_model(str(model_path), gpu=gpu)
    model.eval()
    paths = []
    for index, recording in enumerate(selected, start=1):
        gender, speaker_id, audio_filename = recording
        report(f'Recording {index}/{len(selected)}: {speaker_id} ({gender})')
        paths.extend(_recording_pcas(recording, durations, settings, model,
            gpu, output_dir, report))
        report(f'Recording {index}/{len(selected)} complete')
    return paths


def _recording_pcas(recording, durations, settings, model, gpu, output_dir,
    report):
    '''Keep only this recording's embeddings in memory.'''
    gender, speaker_id, audio_filename = recording

    def window_progress(completed, total):
        report(f'  Extraction: {completed}/{total} windows')

    embeddings, _ = windowed_embeddings.compute_windowed_embeddings(
        audio_filename, duration=max(durations), model=model, gpu=gpu,
        progress=window_progress, **settings)
    paths = []
    for index, duration in enumerate(durations, start=1):
        n_frames = windowed_embeddings.seconds_to_frames(duration, 'duration')
        path = _save_pca(embeddings[:n_frames], audio_filename, gender,
            speaker_id, duration, settings, output_dir)
        paths.append(path)
        report(f'  PCA saved: {index}/{len(durations)} durations')
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


def _save_pca(embeddings, audio_filename, gender, speaker_id, duration,
    settings, output_dir):
    '''Save only PCA results and metadata for an embedding prefix.'''
    result = pca.compute_pca(embeddings)
    filename = windowed_embeddings.make_output_filename(audio_filename,
        settings['start'], duration, settings['model_name'], settings['layer'])
    path = output_dir / pca.make_pca_filename(filename)
    np.savez(path, **result, **settings, audio_filename=audio_filename,
        duration=duration, speaker_id=speaker_id, gender=gender,
        n_frames=len(embeddings))
    return path
