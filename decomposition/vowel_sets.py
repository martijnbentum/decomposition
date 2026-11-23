'''Sample two disjoint vowel datasets (A and B) per experiment cell.

Step 2 of the vowel PCA experiment; it reads the vowel_candidates.npz
written by vowels.run_vowel_report. A cell is either one speaker
(optionally restricted to one CGN component) or a gender group with a
per-speaker cap. Vowels are drawn uniformly over tokens, so label and
duration frequencies follow the corpus. A and B never share a vowel token;
in group cells the same speaker may contribute to both.

echoframe stores only the frames fully inside a segment, on a 20 ms grid of
25 ms frames that starts at max(0, start - collar). A vowel near the file
start can then contain no full frame, which makes echoframe raise; such
vowels are removed from every pool before sampling.
'''
from collections import Counter
import json
from pathlib import Path

from frame.frames import Frame
import numpy as np

from .vowels import RECORD_FIELDS

N_VOWELS = 8000
CAP = 0.05
COLLAR_MS = 2000
STRIDE = 0.02
FIELD = 0.025
VOWEL_CELLS = [
    {'name': 'comp-k_female', 'gender': 'female', 'speaker_id': 'N02005',
        'component': 'k', 'n': N_VOWELS},
    {'name': 'comp-k_male', 'gender': 'male', 'speaker_id': 'N02003',
        'component': 'k', 'n': N_VOWELS},
    {'name': 'comp-a_female', 'gender': 'female', 'speaker_id': 'N01247',
        'component': 'a', 'n': N_VOWELS},
    {'name': 'comp-a_male', 'gender': 'male', 'speaker_id': 'N01238',
        'component': 'a', 'n': N_VOWELS},
    {'name': f'group-{N_VOWELS}_female', 'gender': 'female', 'cap': CAP,
        'n': N_VOWELS},
    {'name': f'group-{N_VOWELS}_male', 'gender': 'male', 'cap': CAP,
        'n': N_VOWELS},
    {'name': f'group-{2 * N_VOWELS}_female', 'gender': 'female', 'cap': CAP,
        'n': 2 * N_VOWELS},
    {'name': f'group-{2 * N_VOWELS}_male', 'gender': 'male', 'cap': CAP,
        'n': 2 * N_VOWELS}]


def load_candidates(filename):
    '''Return the vowel candidate arrays as a dict.'''
    with np.load(filename) as data:
        return {name: data[name] for name in RECORD_FIELDS}


def count_full_frames(start_ms, end_ms, collar_ms=COLLAR_MS):
    '''Return the number of frames echoframe would store for a vowel.'''
    start, end = start_ms / 1000, end_ms / 1000
    grid_start = max(0.0, start - collar_ms / 1000)
    first = max(0, int((start - grid_start) / STRIDE) - 1)
    last = int((end - grid_start) / STRIDE) + 1
    count = 0
    for index in range(first, last + 1):
        frame = Frame(index, STRIDE, FIELD, grid_start, None)
        if frame.overlap_percentage(start, end) >= 100: count += 1
    return count


def full_frame_mask(records, collar_ms=COLLAR_MS):
    '''Return a bool array: the vowel contains at least one full frame.
    Vowels of at least stride + field (45 ms) always do; only shorter
    vowels are checked.
    '''
    durations = records['end'] - records['start']
    mask = np.ones(len(durations), dtype=bool)
    short = np.flatnonzero(durations < 1000 * (STRIDE + FIELD))
    for index in short:
        start, end = records['start'][index], records['end'][index]
        mask[index] = count_full_frames(start, end, collar_ms) > 0
    return mask


def cell_pool(records, cell, eligible=None):
    '''Return indices of the eligible candidates that belong to a cell.
    eligible is an optional bool array, e.g. from full_frame_mask.
    '''
    mask = records['gender'] == cell['gender']
    if eligible is not None: mask &= eligible
    if 'speaker_id' in cell: mask &= records['speaker_id'] == cell['speaker_id']
    if 'component' in cell: mask &= records['component'] == cell['component']
    return np.flatnonzero(mask)


def sample_cell(records, cell, seed=0, eligible=None):
    '''Return (a_indices, b_indices), disjoint and each of length cell['n'].
    records:  candidate arrays from load_candidates
    cell:     dict with name, gender, n and either speaker_id (plus an
              optional component) or cap (maximum share per speaker)
    seed:     seed of the shuffle; the same seed gives the same sets
    eligible: optional bool array restricting the candidates

    The pool is shuffled once; A takes the first eligible tokens and B
    continues from where A stopped. Indices are returned sorted.
    '''
    pool = cell_pool(records, cell, eligible)
    order = np.random.default_rng(seed).permutation(pool)
    if 'cap' in cell:
        speakers = records['speaker_id']
        limit = int(cell['cap'] * cell['n'])
        a, rest = draw_capped(order, speakers, cell['n'], limit)
        b, _ = draw_capped(rest, speakers, cell['n'], limit)
    else:
        a, b = order[:cell['n']], order[cell['n']:2 * cell['n']]
    if len(a) < cell['n'] or len(b) < cell['n']:
        message = f"cell {cell['name']}: need 2 x {cell['n']} vowels, "
        message += f'got {len(a)} + {len(b)} from a pool of {len(pool)}'
        raise ValueError(message)
    return np.sort(a), np.sort(b)


def draw_capped(order, speakers, n, limit):
    '''Take up to n indices in order, at most limit per speaker.
    Returns (taken, rest); rest keeps the order of the unused indices.
    '''
    counts = Counter()
    taken, rest = [], []
    for index in order:
        speaker = speakers[index]
        if len(taken) < n and counts[speaker] < limit:
            counts[speaker] += 1
            taken.append(index)
        else: rest.append(index)
    return np.array(taken, dtype=int), np.array(rest, dtype=int)


def set_summary(records, indices):
    '''Return counts and durations describing one sampled set.'''
    durations = records['end'][indices] - records['start'][indices]
    speakers = Counter(records['speaker_id'][indices].tolist())
    return {'n_vowels': len(indices),
        'vowel_seconds': float(durations.sum() / 1000),
        'mean_ms': float(durations.mean()),
        'n_speakers': len(speakers),
        'max_per_speaker': max(speakers.values()),
        'n_recordings': len(set(records['audio_filename'][indices])),
        'labels': dict(Counter(records['label'][indices].tolist())
            .most_common())}


def save_set(records, indices, cell, set_name, seed, output_dir):
    '''Save one set's vowel records and cell settings; return the path.'''
    path = Path(output_dir) / f"vowels_{cell['name']}_{set_name}.npz"
    subset = {name: records[name][indices] for name in RECORD_FIELDS}
    np.savez(path, **subset, cell=cell['name'], set_name=set_name,
        seed=seed, settings=json.dumps(cell))
    return path


def run_vowel_sampling(candidates_filename, output_dir, cells=VOWEL_CELLS,
    seed=0):
    '''Sample, save and summarise sets A and B for every cell.
    candidates_filename:  vowel_candidates.npz from step 1
    output_dir:           receives vowels_<cell>_<A|B>.npz and
                          vowel_sets.json
    cells:                cell definitions, see sample_cell
    seed:                 base seed; cell i uses seed + i

    Returns the summary dict.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_candidates(candidates_filename)
    eligible = full_frame_mask(records)
    n_removed = int((~eligible).sum())
    print(f'{n_removed} vowels without a full frame removed', flush=True)
    summary = {'candidates': str(candidates_filename), 'seed': seed,
        'collar_ms': COLLAR_MS, 'no_full_frame_removed': n_removed,
        'cells': {}}
    for offset, cell in enumerate(cells):
        cell_seed = seed + offset
        a, b = sample_cell(records, cell, cell_seed, eligible)
        cell_summary = {'settings': cell, 'seed': cell_seed,
            'pool': len(cell_pool(records, cell, eligible))}
        for set_name, indices in (('A', a), ('B', b)):
            save_set(records, indices, cell, set_name, cell_seed, output_dir)
            cell_summary[set_name] = set_summary(records, indices)
        summary['cells'][cell['name']] = cell_summary
        print(format_cell(cell['name'], cell_summary), flush=True)
    with open(output_dir / 'vowel_sets.json', 'w') as fout:
        json.dump(summary, fout, indent=1, ensure_ascii=False)
    return summary


def format_cell(name, cell_summary):
    '''Return one line per set of a cell summary.'''
    lines = [f"{name} (pool {cell_summary['pool']:,})"]
    for set_name in ('A', 'B'):
        s = cell_summary[set_name]
        line = f"  {set_name}: {s['n_vowels']} vowels "
        line += f"{s['vowel_seconds']:.1f} s mean {s['mean_ms']:.0f} ms, "
        line += f"{s['n_speakers']} speakers (max {s['max_per_speaker']}), "
        line += f"{s['n_recordings']} recordings"
        lines.append(line)
    return '\n'.join(lines)
