'''CPU smoke checks for the spawned experiment and prefix PCA outputs.'''
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from decomposition import compute_speaker_pcas, experiment, pca


def fake_experiment(selected, durations, settings, gpu, output_dir, report):
    '''Replace only model inference inside a real spawned worker.'''
    def extract(audio_filename, duration, progress, **kwargs):
        assert duration == 600
        progress(1, 38)
        progress(38, 38)
        values = np.arange(90_000, dtype=float).reshape(30_000, 3)
        return values, np.arange(30_000)

    with patch.object(experiment.to_vector, 'load_model') as load_model:
        with patch.object(experiment.windowed_embeddings,
            'compute_windowed_embeddings', side_effect=extract) as extraction:
            paths = experiment._run_experiment(selected, durations, settings,
                gpu, output_dir, report)
        load_model.assert_called_once_with('model', gpu=False)
        load_model.return_value.eval.assert_called_once()
        assert extraction.call_count == 20
    report(f'Worker PID: {os.getpid()}')
    return paths


def failing_worker(report):
    raise ValueError('test worker failure')


def exiting_worker(report):
    os._exit(7)


def waiting_worker(report):
    import time

    report('ready')
    time.sleep(60)


def test_spawned_prefix_pcas(tmp_path):
    selected = []
    for gender in ('male', 'female'):
        for index in range(10):
            speaker_id = f'{gender}_{index}'
            selected.append((gender, speaker_id, f'{speaker_id}.wav'))
    durations = list(range(60, 601, 60))
    settings = {'model_name': 'model', 'start': 10, 'layer': 9,
        'context': 2, 'center': 16}
    messages = []
    args = (selected, durations, settings, False, tmp_path)
    paths = experiment._run_spawned(fake_experiment, args, messages.append)
    assert len(paths) == len(set(paths)) == 200
    assert len(list(tmp_path.iterdir())) == 200
    assert f'Worker PID: {os.getpid()}' not in messages
    assert any(message.startswith('Worker PID: ') for message in messages)
    assert messages.count('  Extraction: 38/38 windows') == 20
    assert messages.count('  PCA saved: 10/10 durations') == 20
    speaker_ids = {'male': set(), 'female': set()}
    for path in paths:
        result = pca.load_pca(path)
        speaker_ids[result['gender']].add(result['speaker_id'])
        assert 'embeddings' not in result
        assert result['start'] == 10
        assert result['n_frames'] == result['duration'] * 50
    assert len(speaker_ids['male']) == len(speaker_ids['female']) == 10
    first = pca.load_pca(paths[0])
    expected = pca.compute_pca(np.arange(9000).reshape(3000, 3))
    np.testing.assert_allclose(first['covariance'], expected['covariance'])


@pytest.mark.parametrize('function, message', [
    (failing_worker, 'ValueError: test worker failure'),
    (exiting_worker, 'exit code 7')])
def test_worker_errors(function, message):
    with pytest.raises(RuntimeError, match=message):
        experiment._run_spawned(function, (), lambda message: None)


def test_interrupt_reaps_worker():
    children = multiprocessing.active_children()

    def interrupt(message):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        experiment._run_spawned(waiting_worker, (), interrupt)
    assert multiprocessing.active_children() == children


def test_selection_and_inline_mode(tmp_path):
    speakers = {}
    for gender in ('male', 'female'):
        audio = SimpleNamespace(filename=f'{gender}.wav')
        speakers[gender] = {gender: [audio]}
    with patch.object(experiment.selection, 'load_store') as store:
        with patch.object(experiment.selection, 'load_audios', return_value=[]):
            with patch.object(experiment.selection,
                'make_gender_speaker_audio_dict', return_value=speakers):
                with patch.object(experiment, '_run_experiment',
                    return_value=[Path('result.npz')]) as run:
                    paths = compute_speaker_pcas(output_dir=tmp_path,
                        n_male=1, n_female=1, spawn=False, progress=False)
    assert paths == [Path('result.npz')]
    assert len(run.call_args.args[0]) == 2
    store.assert_called_once()
    with pytest.raises(ValueError, match='Need 2 male'):
        experiment._select_recordings(speakers, 2, 1)


def test_window_progress_and_prefix_equivalence():
    module = experiment.windowed_embeddings
    audio = np.arange(8 * module.STRIDE)
    messages = []

    def vector(window, **kwargs):
        frames = len(window) // module.STRIDE
        hidden = np.arange(frames).reshape(1, frames, 1)
        return SimpleNamespace(hidden_states=[hidden])

    with patch.object(module.to_vector, 'audio_to_vector', side_effect=vector):
        longest = module._window_centers(audio, None, 6, 0, 1, 2, False,
            lambda done, total: messages.append((done, total)))
        shortest = module._window_centers(audio, None, 3, 0, 1, 2, False)
    np.testing.assert_array_equal(longest[:3], shortest)
    assert messages == [(1, 3), (2, 3), (3, 3)]
