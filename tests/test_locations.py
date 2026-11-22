'''Checkpoint lookup matches the training step exactly.'''
from unittest.mock import patch

import pytest

from decomposition import locations

NAMES = ['checkpoint_1_1', 'checkpoint_1_10', 'checkpoint_1_100',
    'checkpoint_3_1000', 'checkpoint_23_10000', 'checkpoint_229_100000',
    'checkpoint_best', 'checkpoint_last']


@pytest.mark.parametrize('step', [1, 10, 100, 1000, 10000, 100000])
def test_checkpoint_path_matches_exact_step(tmp_path, step):
    for name in NAMES: (tmp_path / name).mkdir()
    roots = {('wav2vec2', 1): tmp_path}
    with patch.dict(locations.model_roots, roots):
        path = locations.checkpoint_path('wav2vec2', 1, step)
    assert path.name.split('_')[-1] == str(step)


def test_checkpoint_path_missing_step(tmp_path):
    for name in NAMES: (tmp_path / name).mkdir()
    roots = {('wav2vec2', 1): tmp_path}
    with patch.dict(locations.model_roots, roots):
        with pytest.raises(ValueError):
            locations.checkpoint_path('wav2vec2', 1, 5)
