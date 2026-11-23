from types import SimpleNamespace

from decomposition import vowels


def make_records():
    rows = [(b'k1', 'a', 1000.0, 1080.0, 'f.wav', 'N1', 'male', 'o'),
        (b'k2', 'ə', 2000.0, 2050.0, 'f.wav', 'N1', 'male', 'a'),
        (b'k3', 'i', 3000.0, 3060.0, 'g.wav', 'N2', 'female', 'o')]
    return vowels.records_to_arrays(rows)


def test_has_context_requires_audio_on_both_sides():
    phone = SimpleNamespace(start=1000, end=1100)
    assert vowels.has_context(phone, 2100, 1000)
    assert not vowels.has_context(phone, 2099, 1000)
    assert not vowels.has_context(SimpleNamespace(start=999, end=1100),
        5000, 1000)


def test_word_overlaps_treats_missing_word_as_overlap():
    assert vowels.word_overlaps(SimpleNamespace(word=None))
    word = SimpleNamespace(overlap=False)
    assert not vowels.word_overlaps(SimpleNamespace(word=word))


def test_merge_stages_sums_aligned_counts():
    stages = [[('Vowel', 10, 4), ('Gender', 4, 3)],
        [('Vowel', 5, 2), ('Gender', 2, 2)]]
    report = vowels.merge_stages(stages)
    assert report.starting_count == 15
    assert report.eligible_count == 5


def test_speaker_report_and_feasibility():
    speakers = vowels.speaker_vowel_report(make_records())
    assert list(speakers) == ['N1', 'N2']
    assert speakers['N1']['n_vowels'] == 2
    assert speakers['N1']['n_recordings'] == 1
    assert round(speakers['N1']['mean_ms']) == 65
    rows = vowels.feasibility(speakers, [1, 20], cap=0.05)
    male_1 = [r for r in rows if r['n'] == 1 and r['gender'] == 'male'][0]
    assert male_1['single_ok']
    male_20 = [r for r in rows if r['n'] == 20 and r['gender'] == 'male'][0]
    assert male_20['cap_per_dataset'] == 1
    assert male_20['group_usable'] == 2
    assert not male_20['group_ok']


def test_suggest_n_uses_mean_duration():
    assert vowels.suggest_n(make_records(), budget_seconds=0.19) == 3


def test_keys_with_trailing_nul_survive_records():
    key = b'\x01\x02\x00'
    rows = [(key, 'a', 1000.0, 1080.0, 'f.wav', 'N1', 'male', 'o')]
    records = vowels.records_to_arrays(rows)
    assert vowels.key_to_bytes(records['key'][0]) == key
