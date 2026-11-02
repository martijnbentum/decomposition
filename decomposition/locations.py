'''Paths to CGN phraser stores and pretrained Dutch model checkpoints.

Model names have the form <architecture>_nl<version>_checkpoint-<step>,
e.g. wav2vec2_nl1_checkpoint-200000; model_name_to_path recreates the full
checkpoint path from such a name.
'''
from pathlib import Path
import re

phraser_data = Path('/vol/mlusers/mbentum/phraser/data')
cgn_lmdb = phraser_data / 'cgn_lmdb'
cgn_awd_lmdb = phraser_data / 'cgn_awd_lmdb'

speech_training_root = Path('/vol/mlusers/mbentum/speech_training/models')
pretrained_root = speech_training_root / 'pretrained'
model_roots = {
    ('wav2vec2', 1): pretrained_root / 'wav2vec2_the_first',
    ('wav2vec2', 2): pretrained_root / 'wav2vec2_the_second',
    ('huibert', 1): pretrained_root / 'huibert_the_first/second_iteration',
    ('huibert', 2): pretrained_root / 'huibert_the_second/second_iteration',
}

MODEL_NAME_PATTERN = r'(wav2vec2|huibert)_nl([12])_checkpoint-(\d+)'


def make_model_name(architecture, version=1, checkpoint=200_000):
    '''Return a model name such as wav2vec2_nl1_checkpoint-200000.'''
    return f'{architecture}_nl{version}_checkpoint-{checkpoint}'


def parse_model_name(model_name):
    '''Return (architecture, version, checkpoint) for a model name.'''
    match = re.fullmatch(MODEL_NAME_PATTERN, model_name)
    if match is None:
        m = f'invalid model name {model_name!r}, expected e.g. '
        m += make_model_name('wav2vec2')
        raise ValueError(m)
    return match[1], int(match[2]), int(match[3])


def is_model_name(model_name):
    '''Return whether model_name follows the local model name pattern.'''
    return re.fullmatch(MODEL_NAME_PATTERN, str(model_name)) is not None


def model_name_to_path(model_name):
    '''Return the checkpoint path for a model name.'''
    architecture, version, checkpoint = parse_model_name(model_name)
    return checkpoint_path(architecture, version, checkpoint)


def checkpoint_path(architecture, version, checkpoint):
    '''Return the single checkpoint_* path in the root matching checkpoint.'''
    root = model_roots[(architecture, version)]
    matches = []
    for path in root.glob('checkpoint_*'):
        if f'_{checkpoint}' in path.name: matches.append(path)
    if len(matches) != 1:
        m = f'expected one checkpoint {checkpoint} in {root}, '
        m += f'found {len(matches)}: {matches}'
        raise ValueError(m)
    return matches[0]


def model_path(model_name):
    '''Return the checkpoint path for a local model name; other names
    (Hugging Face ids, paths) are returned unchanged.
    '''
    if is_model_name(model_name): return model_name_to_path(model_name)
    return model_name


wav2vec2_nl1_name = make_model_name('wav2vec2', 1, 200_000)
hubert_nl1_name = make_model_name('huibert', 1, 200_000)
