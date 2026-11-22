'''Checkpoint PCA loop with mocked inference, and the second moment helper.'''
from unittest.mock import MagicMock, patch

import numpy as np

from decomposition import experiment, locations, pca


def test_checkpoint_steps():
    steps = experiment.CHECKPOINT_STEPS
    assert len(steps) == 54
    assert steps == sorted(set(steps))
    assert steps[0] == 1 and steps[-1] == 200_000


def fake_extract(audio_filename, duration, model, gpu, model_name, start,
    layer, context, center):
    rng = np.random.default_rng(int(start))
    n_frames = int(round(duration / 0.02))
    return rng.normal(size=(n_frames, 4)), None


def run(tmp_path, steps):
    loaded = []
    def load_model(path, gpu):
        loaded.append(path)
        return MagicMock()
    with patch.object(experiment.to_vector, 'load_model', load_model), \
        patch.object(locations, 'model_path', lambda name: name), \
        patch.object(experiment.windowed_embeddings,
            'compute_windowed_embeddings', fake_extract):
        paths = experiment.compute_checkpoint_pcas('huibert',
            output_dir=tmp_path, model_version=2, steps=steps, duration=6.0,
            gpu=False, progress=False)
    return paths, loaded


def test_compute_checkpoint_pcas_saves_and_skips(tmp_path):
    paths, loaded = run(tmp_path, [1, 100])
    assert len(paths) == 8 and len(loaded) == 2
    names = sorted(path.name for path in paths)
    expected = 'fn001061_s-300_dur-6_huibert_nl2_checkpoint-100_l-9_pca.npz'
    assert expected in names
    d = pca.load_pca(tmp_path / names[0])
    assert d['gender'] in ('male', 'female') and d['n_frames'] == 300
    assert d['covariance'].shape == (4, 4) and d['mean'].shape == (4,)
    paths, loaded = run(tmp_path, [1, 100, 1000])
    assert len(paths) == 4 and loaded == ['huibert_nl2_checkpoint-1000']


def test_second_moment():
    x = np.random.default_rng(0).normal(loc=2.0, size=(500, 5))
    d = pca.compute_pca(x)
    d['n_frames'] = len(x)
    np.testing.assert_allclose(pca.second_moment(d), x.T @ x / len(x))
