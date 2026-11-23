'''Store the hidden states of the sampled vowels in echoframe.

Step 3 of the vowel PCA experiment. Every unique vowel in the step 2 sets is
passed to echoframe's compute_embeddings_batch, which runs the model on the
vowel plus a collar on each side and stores the frames fully inside the
vowel. Each model gets its own echoframe store, because a store allows only
one writer and the models run in parallel on separate GPUs.

Both models are group-norm base models that get no attention mask, so zero
padding inside a batch would change the outputs of the shorter windows.
Vowels are therefore grouped by window length and batched within a group;
lengths with fewer than min_group vowels run unbatched.
'''
from collections import defaultdict
from pathlib import Path

import echoframe
from echoframe.batch_segment_features import compute_embeddings_batch
from echoframe.utils_segment_features import segment_times
import numpy as np
from phraser import Store as PhraserStore

from . import locations
from .vowel_sets import COLLAR_MS, count_full_frames, set_paths
from .vowels import key_to_bytes

VOWEL_LAYERS = {'wav2vec2': 6, 'huibert': 9}
PHRASER_SOURCE_ID = 'cgn_lmdb'
SAMPLE_RATE = 16000


def compute_vowel_embeddings(architecture, layer=None,
    sets_dir=locations.vowel_sets_dir, store_root=locations.vowel_store_root,
    phraser_path=locations.cgn_lmdb, collar_ms=COLLAR_MS, batch_size=16,
    min_group=256, gpu=True):
    '''Compute and store hidden states for all vowels in the step 2 sets.
    architecture:  'wav2vec2' or 'huibert' (nl1, checkpoint 200000)
    layer:         hidden-state layer; None uses VOWEL_LAYERS
    sets_dir:      directory with the vowels_<cell>_<set>.npz files
    store_root:    parent directory; the store is <store_root>/<model_name>
    phraser_path:  phraser store the vowel keys come from
    collar_ms:     audio added on both sides of each vowel
    batch_size:    vowels per batch within one window-length group
    min_group:     smaller window-length groups run unbatched
    gpu:           whether the model runs on GPU

    echoframe skips vowels it already stored, so a run can be restarted.
    Returns the echoframe store.
    '''
    if layer is None: layer = VOWEL_LAYERS[architecture]
    model_name = locations.make_model_name(architecture, 1, 200_000)
    phraser_store = PhraserStore(str(phraser_path), eager_query_roots=False)
    phones = load_set_phones(sets_dir, phraser_store)
    store = open_vowel_store(model_name, store_root, phraser_store)
    groups = group_by_window(phones, collar_ms, batch_size, min_group)
    for index, (size, group) in enumerate(groups, start=1):
        print(f'Group {index}/{len(groups)}: {len(group)} vowels, '
            f'batch size {size}', flush=True)
        compute_embeddings_batch(group, [layer], model_name,
            store, collar=collar_ms, gpu=gpu, batch_size=size)
    return store


def load_set_keys(sets_dir=locations.vowel_sets_dir):
    '''Return the sorted unique phraser keys of the manifest's sets.'''
    keys = set()
    for path in set_paths(sets_dir):
        with np.load(path) as data:
            keys.update(key_to_bytes(key) for key in data['key'])
    return sorted(keys)


def load_set_phones(sets_dir, phraser_store):
    '''Load the phones of all vowel set files from the phraser store.'''
    keys = load_set_keys(sets_dir)
    print(f'Loading {len(keys)} vowels', flush=True)
    phones = []
    for index, key in enumerate(keys, start=1):
        phones.append(phraser_store.load(key))
        if index % 20_000 == 0: print(f'  {index} loaded', flush=True)
    return phones


def open_vowel_store(model_name, store_root, phraser_store):
    '''Open <store_root>/<model_name>, attach phraser, register the model.'''
    store = echoframe.Store(str(Path(store_root) / model_name))
    store.attach_phraser_store(PHRASER_SOURCE_ID, phraser_store)
    if store.load_model_metadata(model_name) is None:
        path = str(locations.model_path(model_name))
        store.register_model(model_name, local_path=path)
    return store


def window_samples(phone, collar_ms=COLLAR_MS):
    '''Return the model input length in samples, as echoframe cuts it.'''
    _, _, start, end = segment_times(phone, collar_ms)
    return int((end - start) * SAMPLE_RATE)


def group_by_window(phones, collar_ms=COLLAR_MS, batch_size=16,
    min_group=256):
    '''Return (batch_size, phones) pairs, one per equal window length.
    Lengths with fewer than min_group phones are pooled into one final
    group with batch size 1, so no batch ever holds padded windows.
    '''
    by_length = defaultdict(list)
    for phone in phones:
        by_length[window_samples(phone, collar_ms)].append(phone)
    groups, rest = [], []
    for length in sorted(by_length):
        group = by_length[length]
        if len(group) >= min_group: groups.append((batch_size, group))
        else: rest.extend(group)
    if rest: groups.append((1, rest))
    return groups


def check_vowel_embeddings(architecture, n_check=50, layer=None,
    sets_dir=locations.vowel_sets_dir, store_root=locations.vowel_store_root,
    collar_ms=COLLAR_MS, seed=0):
    '''Check stored vowels: frame count, dimension and centre frame.
    Loads n_check random vowels from the sets and compares the number of
    stored frames with count_full_frames. Returns a list of problems.
    '''
    if layer is None: layer = VOWEL_LAYERS[architecture]
    model_name = locations.make_model_name(architecture, 1, 200_000)
    store = echoframe.Store(str(Path(store_root) / model_name))
    rows = _set_rows(sets_dir)
    n_check = min(n_check, len(rows))
    picks = np.random.default_rng(seed).choice(len(rows), n_check,
        replace=False)
    problems = []
    for index in picks:
        key, start, end = rows[index]
        embedding = store.phraser_key_to_embedding(key, model_name, layer,
            collar=collar_ms)
        expected = count_full_frames(start, end, collar_ms)
        shape = embedding.data.shape
        if shape[0] != expected or shape[1] != 768:
            problems.append((key.hex(), start, end, shape, expected))
    return problems


def _set_rows(sets_dir):
    '''Return (key, start, end) for every vowel in the manifest's sets.'''
    rows = {}
    for path in set_paths(sets_dir):
        with np.load(path) as data:
            for key, start, end in zip(data['key'], data['start'],
                data['end']):
                rows[key_to_bytes(key)] = (float(start), float(end))
    return [(key, start, end) for key, (start, end) in sorted(rows.items())]
