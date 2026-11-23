from types import SimpleNamespace

import numpy as np
import soundfile as sf
from to_vector.audio import load_audio

from decomposition import vowel_embeddings


def make_phone(start, end, audio_duration=600_000):
    audio = SimpleNamespace(duration=audio_duration)
    return SimpleNamespace(start=start, end=end, start_seconds=start / 1000,
        end_seconds=end / 1000, audio=audio)


def test_window_samples_follow_echoframe_collar():
    assert vowel_embeddings.window_samples(make_phone(5000, 5080)) == 65280
    # clipped at the file start
    assert vowel_embeddings.window_samples(make_phone(1000, 1080)) == 49280


def test_group_by_window_never_mixes_lengths():
    phones = [make_phone(5000, 5080) for _ in range(3)]
    phones += [make_phone(9000, 9100) for _ in range(3)]
    phones.append(make_phone(7000, 7050))
    groups = vowel_embeddings.group_by_window(phones, batch_size=2,
        min_group=3)
    assert [(size, len(group)) for size, group in groups] == [(2, 3),
        (2, 3), (1, 1)]
    for size, group in groups[:-1]:
        lengths = {vowel_embeddings.window_samples(p) for p in group}
        assert len(lengths) == 1


def test_grouping_matches_loaded_16khz_lengths(tmp_path):
    path = tmp_path / 'audio.wav'
    sf.write(path, np.zeros(8 * 16000), 16000)
    phones = [make_phone(3000, 3080), make_phone(3059, 3139)]
    groups = vowel_embeddings.group_by_window(phones, batch_size=2,
        min_group=1)
    assert len(groups) == 2
    lengths = []
    for phone in phones:
        _, _, start, end = vowel_embeddings.segment_times(phone, 2000)
        loaded_length = len(load_audio(path, start, end))
        assert vowel_embeddings.window_samples(phone) == loaded_length
        lengths.append(loaded_length)
    assert lengths == [65280, 65279]
