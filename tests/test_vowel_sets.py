import numpy as np
import pytest

from decomposition import vowel_sets, vowels


def make_records(n_speakers=10, per_speaker=40):
    rows = []
    for s in range(n_speakers):
        gender = 'male' if s % 2 else 'female'
        component = 'k' if s == 0 else 'a'
        for i in range(per_speaker):
            key = f'{s}-{i}'.encode()
            start = 1000.0 * i
            rows.append((key, 'a', start, start + 80, f'f{s}.wav', f'S{s}',
                gender, component))
    return vowels.records_to_arrays(rows)


def test_speaker_cell_gives_disjoint_sets_of_n():
    records = make_records()
    cell = {'name': 'x', 'gender': 'female', 'speaker_id': 'S0',
        'component': 'k', 'n': 15}
    a, b = vowel_sets.sample_cell(records, cell, seed=1)
    assert len(a) == len(b) == 15
    assert not set(a) & set(b)
    assert set(records['speaker_id'][np.concatenate([a, b])]) == {'S0'}


def test_same_seed_gives_same_sets():
    records = make_records()
    cell = {'name': 'x', 'gender': 'female', 'speaker_id': 'S0', 'n': 10}
    first = vowel_sets.sample_cell(records, cell, seed=3)
    second = vowel_sets.sample_cell(records, cell, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(first, second))


def test_group_cell_respects_cap():
    records = make_records(n_speakers=10, per_speaker=40)
    cell = {'name': 'g', 'gender': 'male', 'cap': 0.25, 'n': 40}
    a, b = vowel_sets.sample_cell(records, cell, seed=0)
    assert not set(a) & set(b)
    for indices in (a, b):
        counts = np.unique(records['speaker_id'][indices],
            return_counts=True)[1]
        assert counts.max() <= 10


def test_too_small_pool_raises():
    records = make_records(n_speakers=2, per_speaker=10)
    cell = {'name': 'x', 'gender': 'female', 'speaker_id': 'S0', 'n': 6}
    with pytest.raises(ValueError):
        vowel_sets.sample_cell(records, cell)
