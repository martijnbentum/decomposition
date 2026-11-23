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
    assert vowel_pca.load_all_frames(store, ['a', 'b']).shape == (5, 2)


def test_missing_vowel_raises():
    with pytest.raises(ValueError):
        vowel_pca.load_all_frames(make_store(), ['a', 'x'])


def test_pca_path():
    path = vowel_pca.pca_path('/x/vowels_comp-k_male_A.npz', 'all', '/out')
    assert str(path) == '/out/vowels_comp-k_male_A_all_pca.npz'
