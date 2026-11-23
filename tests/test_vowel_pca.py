import json

import numpy as np
import pytest

from decomposition import vowel_pca


class FakeStorage:
    def __init__(self, payloads):
        self.payloads = payloads

    def load_many(self, metadata):
        return [self.payloads[m] for m in metadata]


class FakeStore:
    def __init__(self, payloads):
        self.payloads = payloads
        self.storage = FakeStorage(payloads)

    def make_echoframe_key(self, kind, model_name, phraser_key, layer,
        collar):
        return phraser_key

    def load_many_metadata(self, keys, keep_missing=False):
        return [key if key in self.payloads else None for key in keys]

    def load_many_frames(self, keys, frame='center', keep_missing=False):
        rows = []
        for key in keys:
            payload = self.payloads.get(key)
            rows.append(None if payload is None else payload[len(payload) // 2])
        return rows


def make_store():
    payloads = {'a': np.arange(6.0).reshape(3, 2),
        'b': np.arange(4.0).reshape(2, 2) + 10}
    return FakeStore(payloads)


def test_center_and_all_frames():
    store = make_store()
    center = vowel_pca.load_center_frames(store, ['a', 'b'])
    assert center.tolist() == [[2.0, 3.0], [12.0, 13.0]]
    payloads = vowel_pca.load_all_frames(store, ['a', 'b'])
    assert [len(p) for p in payloads] == [3, 2]


def test_missing_vowel_raises():
    with pytest.raises(ValueError):
        vowel_pca.load_all_frames(make_store(), ['a', 'x'])


def test_pca_path():
    path = vowel_pca.pca_path('/x/vowels_comp-k_male_A.npz', 'all', '/out')
    assert str(path) == '/out/vowels_comp-k_male_A_all_pca.npz'


def write_set(tmp_path, keys):
    path = tmp_path / 'vowels_x_A.npz'
    np.savez(path, key=np.array(keys), label=np.array(['a'] * len(keys)),
        cell='x_female', set_name='A', settings=json.dumps({'name': 'x'}))
    return path


def make_set_store():
    rng = np.random.default_rng(0)
    payloads = {bytes([i]): rng.normal(size=(i + 1, 3)) for i in range(1, 6)}
    return FakeStore(payloads)


def test_set_pcas_skip_only_matching_outputs(tmp_path):
    keys = [bytes([i]).hex() for i in range(1, 6)]
    set_path = write_set(tmp_path, keys)
    store = make_set_store()
    settings = {'model_name': 'm', 'layer': 6, 'collar_ms': 2000}
    paths = vowel_pca.set_pcas(set_path, store, settings, tmp_path)
    assert len(paths) == 2
    with np.load(paths[1]) as data:
        assert data['n_frames_per_vowel'].tolist() == [2, 3, 4, 5, 6]
    assert vowel_pca.set_pcas(set_path, store, settings, tmp_path) == []
    with pytest.raises(ValueError):
        vowel_pca.set_pcas(set_path, store, {**settings, 'layer': 9},
            tmp_path)
    write_set(tmp_path, keys[:4])
    with pytest.raises(ValueError):
        vowel_pca.set_pcas(set_path, store, settings, tmp_path)
