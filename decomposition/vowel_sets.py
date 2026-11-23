'''Sample two disjoint vowel datasets (A and B) per experiment cell.

Step 2 of the vowel PCA experiment; it reads the vowel_candidates.npz
written by vowels.run_vowel_report. A and B never share a recording, so no
vowel in A shares its collar window with a vowel in B.

- speaker cell: one speaker in one CGN component. The speaker's recordings
  are split into two halves and each set is drawn uniformly from its half.
- group cell: one gender, a fixed CGN component mix (MIX) that is the same
  for both genders, and at most cap * n vowels per speaker per set. Each
  component's recordings are split into two halves. Comp-k is limited to
  MAX_SPEAKERS['k'] speakers, the number of female comp-k speakers. Group
  cells exclude the speakers of the speaker cells and every recording in
  which those speakers occur. Speakers can occur in both A and B (for
  example the interviewer of comp-b); the summary reports how many do.

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

from . import locations
from .vowels import RECORD_FIELDS

N_VOWELS = 8000
CAP = 0.05
COLLAR_MS = 2000
STRIDE = 0.02
FIELD = 0.025
MIX = {'a': 0.40, 'f': 0.14, 'h': 0.12, 'g': 0.11, 'b': 0.10, 'k': 0.08,
    'j': 0.03, 'l': 0.02}
MAX_SPEAKERS = {'k': 7}


def speaker_cell(name, gender, speaker_id, component, seed):
    '''Return a speaker cell definition with N_VOWELS per set.'''
    return {'name': name, 'gender': gender, 'speaker_id': speaker_id,
        'component': component, 'n': N_VOWELS, 'seed': seed}


def group_cell(gender, n, seed):
    '''Return a group cell definition with the shared component mix.'''
    return {'name': f'group-{n}_{gender}', 'gender': gender, 'n': n,
        'cap': CAP, 'mix': MIX, 'max_speakers': MAX_SPEAKERS, 'seed': seed}


VOWEL_CELLS = [speaker_cell('comp-k_female', 'female', 'N02005', 'k', 0),
    speaker_cell('comp-k_male', 'male', 'N02003', 'k', 1),
    speaker_cell('comp-a_female', 'female', 'N01247', 'a', 2),
    speaker_cell('comp-a_male', 'male', 'N01238', 'a', 3),
    group_cell('female', N_VOWELS, 4), group_cell('male', N_VOWELS, 5),
    group_cell('female', 2 * N_VOWELS, 6), group_cell('male', 2 * N_VOWELS, 7)]


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


def group_exclusion(records, cells):
    '''Return a bool array: vowels of speaker-cell speakers or recordings.'''
    speaker_ids = [c['speaker_id'] for c in cells if 'speaker_id' in c]
    speakers = np.isin(records['speaker_id'], speaker_ids)
    recordings = np.unique(records['audio_filename'][speakers])
    return speakers | np.isin(records['audio_filename'], recordings)


def cell_pool(records, cell, eligible=None, excluded=None):
    '''Return indices of the candidates that belong to a cell.
    eligible:  optional bool array, e.g. from full_frame_mask
    excluded:  optional bool array, applied to group cells only
    '''
    mask = records['gender'] == cell['gender']
    if eligible is not None: mask &= eligible
    if 'speaker_id' in cell:
        mask &= records['speaker_id'] == cell['speaker_id']
        mask &= records['component'] == cell['component']
    else:
        mask &= np.isin(records['component'], list(cell['mix']))
        if excluded is not None: mask &= ~excluded
    return np.flatnonzero(mask)


def sample_cell(records, cell, eligible=None, excluded=None):
    '''Return (a_indices, b_indices) of cell['n'] vowels each.
    records:   candidate arrays from load_candidates
    cell:      speaker_cell or group_cell definition
    eligible:  optional bool array restricting the candidates
    excluded:  optional bool array of vowels group cells may not use

    A and B share no recording. The same cell['seed'] gives the same sets.
    Indices are returned sorted.
    '''
    pool = cell_pool(records, cell, eligible, excluded)
    rng = np.random.default_rng(cell['seed'])
    if 'speaker_id' in cell: halves = split_recordings(records, pool, rng)
    else: halves = split_group(records, cell, pool, rng)
    sets = []
    for name, half in zip('AB', halves):
        if 'speaker_id' in cell: indices = draw_uniform(half, cell['n'], rng)
        else: indices = draw_group(records, cell, half, rng)
        if len(indices) < cell['n']:
            message = f"cell {cell['name']} set {name}: need {cell['n']} "
            message += f'vowels, got {len(indices)} from a pool of {len(pool)}'
            raise ValueError(message)
        sets.append(np.sort(indices))
    return sets[0], sets[1]


def split_recordings(records, indices, rng):
    '''Split indices into two halves that share no recording.
    Recordings go, largest first, to the half with fewer vowels; recordings
    of equal size are taken in random order and the halves are swapped at
    random, so neither half systematically gets the largest recording.
    '''
    filenames = records['audio_filename'][indices]
    names, inverse, counts = np.unique(filenames, return_inverse=True,
        return_counts=True)
    order = np.lexsort((rng.permutation(len(names)), -counts))
    half_of = np.zeros(len(names), dtype=int)
    totals = [0, 0]
    for name_index in order:
        half = 0 if totals[0] <= totals[1] else 1
        half_of[name_index] = half
        totals[half] += counts[name_index]
    in_first = half_of[inverse] == 0
    halves = [indices[in_first], indices[~in_first]]
    if rng.random() < 0.5: halves.reverse()
    return halves


def split_group(records, cell, pool, rng):
    '''Return two halves, each a {component: indices} dict.'''
    halves = ({}, {})
    for component in cell['mix']:
        indices = pool[records['component'][pool] == component]
        n_speakers = cell['max_speakers'].get(component)
        if n_speakers:
            indices = keep_speakers(records, indices, n_speakers, rng)
        first, second = split_recordings(records, indices, rng)
        halves[0][component], halves[1][component] = first, second
    return halves


def keep_speakers(records, indices, n_speakers, rng):
    '''Return the indices of n_speakers speakers chosen at random.'''
    speakers = np.unique(records['speaker_id'][indices])
    if len(speakers) <= n_speakers: return indices
    chosen = rng.choice(speakers, n_speakers, replace=False)
    return indices[np.isin(records['speaker_id'][indices], chosen)]


def component_counts(mix, n):
    '''Return {component: count} with the counts summing to n.'''
    counts = {component: round(share * n) for component, share in mix.items()}
    largest = max(mix, key=mix.get)
    counts[largest] += n - sum(counts.values())
    return counts


def draw_uniform(indices, n, rng):
    '''Return n indices drawn without replacement, or all if fewer.'''
    if len(indices) <= n: return indices
    return rng.choice(indices, n, replace=False)


def draw_group(records, cell, half, rng):
    '''Draw the component counts from one half under the speaker cap.'''
    limit = int(cell['cap'] * cell['n'])
    speakers = records['speaker_id']
    counts = Counter()
    taken = []
    for component, n in component_counts(cell['mix'], cell['n']).items():
        order = rng.permutation(half[component])
        chosen = draw_capped(order, speakers, n, limit, counts)
        if len(chosen) < n:
            message = f"cell {cell['name']} component {component}: need {n}"
            message += f' vowels, got {len(chosen)}'
            raise ValueError(message)
        taken.append(chosen)
    return np.concatenate(taken)


def draw_capped(order, speakers, n, limit, counts=None):
    '''Take up to n indices in order, at most limit per speaker.
    counts holds the vowels already taken per speaker and is updated.
    '''
    if counts is None: counts = Counter()
    taken = []
    for index in order:
        if len(taken) == n: break
        if counts[speakers[index]] < limit:
            counts[speakers[index]] += 1
            taken.append(index)
    return np.array(taken, dtype=int)


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
        'components': dict(Counter(records['component'][indices].tolist())),
        'labels': dict(Counter(records['label'][indices].tolist())
            .most_common())}


def shared_counts(records, a, b):
    '''Return the number of speakers and recordings in both A and B.'''
    counts = {}
    for field, name in (('speaker_id', 'speakers'),
        ('audio_filename', 'recordings')):
        shared = set(records[field][a]) & set(records[field][b])
        counts[f'shared_{name}'] = len(shared)
    return counts


def save_set(records, indices, cell, set_name, output_dir):
    '''Save one set's vowel records and cell settings; return the path.'''
    path = Path(output_dir) / f"vowels_{cell['name']}_{set_name}.npz"
    subset = {name: records[name][indices] for name in RECORD_FIELDS}
    np.savez(path, **subset, cell=cell['name'], set_name=set_name,
        seed=cell['seed'], settings=json.dumps(cell))
    return path


def run_vowel_sampling(candidates_filename=None,
    output_dir=locations.vowel_sets_dir, cells=VOWEL_CELLS):
    '''Sample, save and summarise sets A and B for every cell.
    candidates_filename:  vowel_candidates.npz from step 1; None uses
                          locations.vowel_report_dir
    output_dir:           receives vowels_<cell>_<A|B>.npz and
                          vowel_sets.json
    cells:                speaker_cell and group_cell definitions

    Returns the summary dict.
    '''
    if candidates_filename is None:
        candidates_filename = locations.vowel_candidates_filename
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_candidates(candidates_filename)
    eligible = full_frame_mask(records)
    excluded = group_exclusion(records, cells)
    n_removed = int((~eligible).sum())
    print(f'{n_removed} vowels without a full frame removed', flush=True)
    summary = {'candidates': str(candidates_filename),
        'collar_ms': COLLAR_MS, 'no_full_frame_removed': n_removed,
        'group_excluded': int(excluded.sum()), 'cells': {}}
    for cell in cells:
        a, b = sample_cell(records, cell, eligible, excluded)
        pool = cell_pool(records, cell, eligible, excluded)
        cell_summary = {'settings': cell, 'pool': len(pool)}
        cell_summary.update(shared_counts(records, a, b))
        for set_name, indices in (('A', a), ('B', b)):
            save_set(records, indices, cell, set_name, output_dir)
            cell_summary[set_name] = set_summary(records, indices)
        summary['cells'][cell['name']] = cell_summary
        print(format_cell(cell['name'], cell_summary), flush=True)
    with open(output_dir / 'vowel_sets.json', 'w') as fout:
        json.dump(summary, fout, indent=1, ensure_ascii=False)
    return summary


def format_cell(name, cell_summary):
    '''Return one line per set of a cell summary.'''
    lines = [f"{name} (pool {cell_summary['pool']:,}, shared speakers "
        f"{cell_summary['shared_speakers']}, shared recordings "
        f"{cell_summary['shared_recordings']})"]
    for set_name in ('A', 'B'):
        s = cell_summary[set_name]
        line = f"  {set_name}: {s['n_vowels']} vowels "
        line += f"{s['vowel_seconds']:.1f} s mean {s['mean_ms']:.0f} ms, "
        line += f"{s['n_speakers']} speakers (max {s['max_per_speaker']}), "
        line += f"{s['n_recordings']} recordings"
        lines.append(line)
    return '\n'.join(lines)
