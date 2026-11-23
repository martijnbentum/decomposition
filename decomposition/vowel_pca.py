'''PCA of the stored vowel embeddings, one per set, model and frame choice.

Step 4 of the vowel PCA experiment. For every vowels_<cell>_<A|B>.npz set
the vowel embeddings are read from the model's echoframe store and two PCAs
are computed:

- center: one vector per vowel, echoframe's center frame (stored frame
  n // 2; with an even count the later of the two middle frames)
- all:    every stored frame of every vowel, i.e. all frames fully inside
  the vowel boundaries, stacked
'''
import json
from pathlib import Path
import time

import echoframe
import numpy as np

from . import locations, pca
from .vowel_embeddings import VOWEL_LAYERS
from .vowel_sets import COLLAR_MS
from .vowels import key_to_bytes

FRAME_CHOICES = ('center', 'all')


def run_vowel_pcas(architecture, sets_dir=locations.vowel_sets_dir,
    store_root=locations.vowel_store_root,
    output_dir=locations.decomposition_root / 'vowel_pca', layer=None,
    collar_ms=COLLAR_MS):
    '''Save center and all-frame PCAs for every vowel set of one model.
    architecture:  'wav2vec2' or 'huibert' (nl1, checkpoint 200000)
    sets_dir:      directory with the vowels_<cell>_<set>.npz files
    store_root:    parent of the echoframe store <store_root>/<model_name>
    output_dir:    PCAs go to <output_dir>/<model_name>/
    layer:         hidden-state layer; None uses VOWEL_LAYERS
    collar_ms:     collar the embeddings were stored with

    Existing outputs are skipped. Returns the saved paths.
    '''
    if layer is None: layer = VOWEL_LAYERS[architecture]
    model_name = locations.make_model_name(architecture, 1, 200_000)
    store = echoframe.Store(str(Path(store_root) / model_name))
    output_dir = Path(output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {'model_name': model_name, 'layer': layer,
        'collar_ms': collar_ms}
    paths = []
    for set_path in sorted(Path(sets_dir).glob('vowels_*.npz')):
        begin = time.time()
        paths.extend(set_pcas(set_path, store, settings, output_dir))
        print(f'{set_path.stem} done in {time.time() - begin:.0f} s',
            flush=True)
    print(f'Complete: {len(paths)} PCA files saved', flush=True)
    return paths


def set_pcas(set_path, store, settings, output_dir):
    '''Compute and save the missing PCAs of one vowel set.'''
    missing = []
    for frames in FRAME_CHOICES:
        path = pca_path(set_path, frames, output_dir)
        if not path.exists(): missing.append((frames, path))
    if not missing: return []
    with np.load(set_path) as data:
        vowel_set = {name: data[name] for name in data.files}
    keys = echoframe_keys(store, vowel_set['key'], settings)
    paths = []
    for frames, path in missing:
        if frames == 'center': matrix = load_center_frames(store, keys)
        else: matrix = load_all_frames(store, keys)
        save_set_pca(matrix, vowel_set, frames, settings, path)
        paths.append(path)
    return paths


def echoframe_keys(store, hex_keys, settings):
    '''Return the hidden-state echoframe keys of the vowels in a set.'''
    keys = []
    for hex_key in hex_keys:
        keys.append(store.make_echoframe_key('hidden_state',
            model_name=settings['model_name'],
            phraser_key=key_to_bytes(hex_key), layer=settings['layer'],
            collar=settings['collar_ms']))
    return keys


def load_center_frames(store, keys):
    '''Return echoframe's center frame per vowel as (n_vowels, dim).'''
    rows = store.load_many_frames(keys, frame='center', keep_missing=True)
    _check_complete(rows, 'center frames')
    return np.stack(rows)


def load_all_frames(store, keys):
    '''Return all stored frames of all vowels as (n_frames, dim).'''
    metadata = store.load_many_metadata(keys, keep_missing=True)
    _check_complete(metadata, 'metadata')
    payloads = store.storage.load_many(metadata)
    _check_complete(payloads, 'payloads')
    return np.concatenate(payloads)


def _check_complete(items, name):
    missing = sum(item is None for item in items)
    if missing:
        raise ValueError(f'{missing} of {len(items)} {name} missing in store')


def pca_path(set_path, frames, output_dir):
    '''Return e.g. <output_dir>/vowels_comp-k_male_A_center_pca.npz.'''
    return Path(output_dir) / f'{Path(set_path).stem}_{frames}_pca.npz'


def save_set_pca(matrix, vowel_set, frames, settings, path):
    '''Compute the PCA of matrix and save it with the set metadata.'''
    result = pca.compute_pca(matrix)
    cell = str(vowel_set['cell'])
    np.savez(path, **result, **settings, cell=cell,
        set_name=str(vowel_set['set_name']), frames=frames,
        gender=cell.split('_')[-1], n_vowels=len(vowel_set['key']),
        n_frames=len(matrix), cell_settings=str(vowel_set['settings']),
        labels=json.dumps(_label_counts(vowel_set['label'])))


def _label_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}
