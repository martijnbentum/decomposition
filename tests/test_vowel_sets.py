from collections import Counter

import numpy as np
import pytest

from decomposition import vowel_sets, vowels


def make_records(n_speakers=20, n_recordings=4, per_recording=30):
    '''Speakers alternate gender; speaker s talks in component a or k.'''
    rows = []
    for s in range(n_speakers):
        gender = 'male' if s % 2 else 'female'
        component = 'k' if s % 4 < 2 else 'a'
        for r in range(n_recordings):
            for i in range(per_recording):
                key = f'{s}-{r}-{i}'.encode()
                start = 1000.0 * (i + 2)
                rows.append((key, 'a', start, start + 80, f's{s}r{r}.wav',
                    f'S{s}', gender, component))
    return vowels.records_to_arrays(rows)


def speaker_cell(n=40, seed=1):
    return {'name': 'x', 'gender': 'female', 'speaker_id': 'S0',
        'component': 'k', 'n': n, 'seed': seed}


def group_cell(n=100, seed=0, max_speakers=None):
    return {'name': 'g', 'gender': 'female', 'n': n, 'cap': 0.25,
        'mix': {'a': 0.6, 'k': 0.4}, 'max_speakers': max_speakers or {},
        'seed': seed}


def recordings(records, indices):
    return set(records['audio_filename'][indices])


def test_speaker_cell_splits_recordings():
    records = make_records()
    a, b = vowel_sets.sample_cell(records, speaker_cell())
    assert len(a) == len(b) == 40
    assert not recordings(records, a) & recordings(records, b)
    assert set(records['speaker_id'][np.concatenate([a, b])]) == {'S0'}


def test_same_seed_gives_same_sets():
    records = make_records()
    first = vowel_sets.sample_cell(records, group_cell(seed=3))
    second = vowel_sets.sample_cell(records, group_cell(seed=3))
    assert all(np.array_equal(x, y) for x, y in zip(first, second))


def test_group_cell_mix_cap_and_recordings():
    records = make_records()
    a, b = vowel_sets.sample_cell(records, group_cell())
    assert not recordings(records, a) & recordings(records, b)
    for indices in (a, b):
        assert Counter(records['component'][indices]) == {'a': 60, 'k': 40}
        counts = Counter(records['speaker_id'][indices])
        assert max(counts.values()) <= 25
        assert set(records['gender'][indices]) == {'female'}


def test_group_cell_skips_excluded_and_limits_speakers():
    records = make_records()
    cells = [speaker_cell(n=10), group_cell(n=40, max_speakers={'k': 2})]
    excluded = vowel_sets.group_exclusion(records, cells)
    a, b = vowel_sets.sample_cell(records, cells[1], excluded=excluded)
    both = np.concatenate([a, b])
    assert 'S0' not in set(records['speaker_id'][both])
    k_part = both[records['component'][both] == 'k']
    assert len(set(records['speaker_id'][k_part])) <= 2


def test_too_small_pool_raises():
    records = make_records(n_recordings=2, per_recording=10)
    with pytest.raises(ValueError):
        vowel_sets.sample_cell(records, speaker_cell(n=11))


def test_split_recordings_balances_halves():
    records = make_records(n_speakers=1, n_recordings=6)
    indices = np.arange(len(records['key']))
    rng = np.random.default_rng(0)
    first, second = vowel_sets.split_recordings(records, indices, rng)
    assert len(first) == len(second) == 90
    assert not recordings(records, first) & recordings(records, second)


def test_component_counts_sum_to_n():
    counts = vowel_sets.component_counts(vowel_sets.MIX, 8000)
    assert sum(counts.values()) == 8000
    assert counts['k'] == 640


def test_count_full_frames_matches_grid():
    # unclipped: the grid starts exactly at the vowel start
    assert vowel_sets.count_full_frames(5000, 5029) == 1
    assert vowel_sets.count_full_frames(5000, 5065) == 3
    # clipped at the file start: grid at 0, 20, 40 ms ...
    assert vowel_sets.count_full_frames(1230, 1260) == 0
    assert vowel_sets.count_full_frames(1240, 1265) == 1


def test_full_frame_mask_removes_short_clipped_vowels():
    rows = [(b'a', 'a', 1230.0, 1260.0, 'f.wav', 'S0', 'male', 'a'),
        (b'b', 'a', 5000.0, 5029.0, 'f.wav', 'S0', 'male', 'a'),
        (b'c', 'a', 1230.0, 1400.0, 'f.wav', 'S0', 'male', 'a')]
    mask = vowel_sets.full_frame_mask(vowels.records_to_arrays(rows))
    assert mask.tolist() == [False, True, True]
