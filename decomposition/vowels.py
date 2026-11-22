'''Collect eligible CGN vowel tokens and report what each speaker offers.

Step 1 of the vowel PCA experiment; no model is needed. A vowel is a phone
whose IPA type is 'vowel' (schwa included). It must come from a speaker of
known gender, carry the within_range duration flag, have no recorded timing
concern, sit in a word that does not overlap another speaker, and have at
least min_context_ms of audio before and after it inside the file.

The report shows the loss per filter, the eligible vowel time per speaker
(also per CGN component), label frequencies, and for candidate dataset
sizes N whether one speaker per gender can fill two disjoint datasets or
whether a gender group can under a per-speaker cap.
'''
from collections import Counter
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
from phraser import Store
from phraser.sampling import SelectionReport, cgn
from phraser.sampling._selection import TokenFilters, filter_candidates

from . import locations

# nl recordings; c and d are 8 kHz telephone speech, m is of poor quality
COMPONENTS = 'abefghijkl'
DIPHTHONGS = ('ɛi', 'œy', 'ɑu', 'ui')
GENDERS = ('female', 'male')
RECORD_FIELDS = ('key', 'label', 'start', 'end', 'audio_filename',
    'speaker_id', 'gender', 'component')

_worker_store = None


def collect_vowel_candidates(store_path=locations.cgn_lmdb,
    components=COMPONENTS, variant='nl', exclude_diphthongs=True,
    min_context_ms=1000, processes=16, progress=True):
    '''Return (SelectionReport, records) for all eligible vowel tokens.
    store_path:          phraser store, WebMAUS cgn_lmdb by default
    components:          CGN component letters to include
    variant:             'nl' or 'vl'
    exclude_diphthongs:  drop the labels in DIPHTHONGS
    min_context_ms:      audio required before and after the vowel
    processes:           worker processes, each with its own Store
    progress:            print progress every 100 recordings

    records is a dict of equal-length numpy arrays named RECORD_FIELDS;
    start and end are in milliseconds, key holds the phraser phone keys.
    '''
    store = Store(store_path, eager_query_roots=False)
    audios = cgn.select_audios(store=store, components=components,
        variant=variant)
    keys = [audio.key for audio in audios]
    settings = (exclude_diphthongs, min_context_ms)
    tasks = [(key, settings) for key in keys]
    context = multiprocessing.get_context('spawn')
    results = []
    begin = time.time()
    with context.Pool(processes, _init_worker, (str(store_path),)) as pool:
        for result in pool.imap_unordered(_audio_task, tasks, chunksize=4):
            results.append(result)
            if progress and len(results) % 100 == 0:
                elapsed = time.time() - begin
                print(f'{len(results)}/{len(keys)} recordings, '
                    f'{elapsed:.0f} s', flush=True)
    return _merge_results(results)


def _init_worker(store_path):
    global _worker_store
    _worker_store = Store(store_path, eager_query_roots=False)


def _audio_task(task):
    '''Load one recording in a worker and collect its eligible vowels.'''
    key, (exclude_diphthongs, min_context_ms) = task
    audio = _worker_store.load(key)
    return audio_vowel_candidates(audio, exclude_diphthongs, min_context_ms)


def audio_vowel_candidates(audio, exclude_diphthongs=True,
    min_context_ms=1000):
    '''Return (stages, records) for the vowels of one recording.
    stages is a list of (condition, before, after) tuples; records is a
    list of tuples ordered as RECORD_FIELDS.
    '''
    phones = audio.phones
    report = SelectionReport('phone tokens', len(phones), None)
    vowels = _filter_vowels(phones, report, audio, exclude_diphthongs,
        min_context_ms)
    stages = [(s.condition, s.before, s.after) for s in report.stages]
    component = Path(audio.filename).parent.parent.name[-1]
    speakers = {speaker.key: speaker for speaker in audio.speakers}
    records = []
    for vowel in vowels:
        speaker = speakers[vowel.speaker_key]
        records.append((vowel.key, vowel.label, vowel.start, vowel.end,
            str(audio.filename), speaker.name, speaker.gender(), component))
    return stages, records


def _filter_vowels(phones, report, audio, exclude_diphthongs,
    min_context_ms):
    '''Apply the vowel filters in order and record each loss in report.'''
    genders = {s.key: s.gender() for s in audio.speakers}
    vowels = filter_candidates(phones, report, 'Vowel',
        lambda phone: phone.type == 'vowel')
    if exclude_diphthongs:
        vowels = filter_candidates(vowels, report, 'Not a diphthong',
            lambda phone: phone.label not in DIPHTHONGS)
    vowels = filter_candidates(vowels, report, 'Speaker male or female',
        lambda phone: genders.get(phone.speaker_key) in GENDERS)
    filters = TokenFilters(exclude_timing_uncertain=True,
        exclude_overlap=True)
    vowels = filters.apply(vowels, report, overlap_test=word_overlaps)
    condition = f'At least {min_context_ms} ms of audio on both sides'
    return filter_candidates(vowels, report, condition,
        lambda phone: has_context(phone, audio.duration, min_context_ms))


def word_overlaps(phone):
    '''Whether the phone's word overlaps another speaker; True if no word.'''
    word = phone.word
    if word is None: return True
    return word.overlap


def has_context(phone, audio_duration, min_context_ms):
    '''Whether min_context_ms of audio lies before and after the phone.'''
    if phone.start < min_context_ms: return False
    return phone.end + min_context_ms <= audio_duration


def _merge_results(results):
    '''Sum per-recording stage counts and stack the records.'''
    stage_lists = [stages for stages, _ in results]
    report = merge_stages(stage_lists)
    rows = []
    for _, records in results:
        rows.extend(records)
    rows.sort(key=lambda row: row[0])
    return report, records_to_arrays(rows)


def merge_stages(stage_lists):
    '''Return one SelectionReport summing aligned per-recording stages.'''
    stage_lists = [stages for stages in stage_lists if stages]
    if not stage_lists: return SelectionReport('vowel tokens', 0, None)
    conditions = [condition for condition, _, _ in stage_lists[0]]
    before = [0] * len(conditions)
    after = [0] * len(conditions)
    for stages in stage_lists:
        if [condition for condition, _, _ in stages] != conditions:
            raise ValueError('recordings report different filter stages')
        for index, (_, n_before, n_after) in enumerate(stages):
            before[index] += n_before
            after[index] += n_after
    report = SelectionReport('phone tokens', before[0], None)
    for condition, n_before, n_after in zip(conditions, before, after):
        report.add_stage(condition, n_before, n_after)
    return report


def records_to_arrays(rows):
    '''Return a dict of numpy arrays, one per RECORD_FIELDS name.'''
    columns = list(zip(*rows)) if rows else [()] * len(RECORD_FIELDS)
    arrays = {}
    for name, column in zip(RECORD_FIELDS, columns):
        if name in ('start', 'end'): arrays[name] = np.array(column, float)
        elif name == 'key': arrays[name] = np.array(column, dtype=bytes)
        else: arrays[name] = np.array(column, dtype=str)
    return arrays


def speaker_vowel_report(records):
    '''Return {speaker_id: summary} sorted by eligible vowel count.
    Each summary holds gender, n_vowels, vowel_seconds, mean_ms,
    n_recordings and vowel_seconds per component.
    '''
    durations = records['end'] - records['start']
    summaries = {}
    for index, speaker_id in enumerate(records['speaker_id']):
        summary = summaries.setdefault(str(speaker_id), _empty_summary(
            records['gender'][index]))
        summary['n_vowels'] += 1
        summary['vowel_seconds'] += durations[index] / 1000
        component = str(records['component'][index])
        components = summary['component_seconds']
        components[component] = components.get(component, 0)
        components[component] += durations[index] / 1000
        summary['recordings'].add(str(records['audio_filename'][index]))
    for summary in summaries.values():
        summary['mean_ms'] = 1000 * summary['vowel_seconds']
        summary['mean_ms'] /= summary['n_vowels']
        summary['n_recordings'] = len(summary.pop('recordings'))
    order = sorted(summaries, key=lambda s: -summaries[s]['n_vowels'])
    return {speaker_id: summaries[speaker_id] for speaker_id in order}


def _empty_summary(gender):
    return {'gender': str(gender), 'n_vowels': 0, 'vowel_seconds': 0.0,
        'component_seconds': {}, 'recordings': set()}


def label_counts(records):
    '''Return {'all': Counter, 'female': Counter, 'male': Counter}.'''
    counts = {'all': Counter(records['label'].tolist())}
    for gender in GENDERS:
        labels = records['label'][records['gender'] == gender]
        counts[gender] = Counter(labels.tolist())
    return counts


def feasibility(speakers, n_values, cap=0.05):
    '''Return one row per dataset size N for both routes and genders.
    speakers:  output of speaker_vowel_report
    n_values:  candidate numbers of vowels per dataset
    cap:       maximum share of a grouped dataset from one speaker

    Single speaker: the best speaker needs 2N vowels (datasets A and B).
    Grouped: each speaker gives at most floor(cap * N) vowels to each
    dataset, so at most twice that to A and B together; the gender group
    needs 2N in total.
    '''
    rows = []
    for n in n_values:
        per_dataset = int(cap * n)
        for gender in GENDERS:
            counts = [s['n_vowels'] for s in speakers.values()
                if s['gender'] == gender]
            best = max(counts, default=0)
            usable = sum(min(count, 2 * per_dataset) for count in counts)
            full = sum(1 for count in counts if count >= 2 * per_dataset)
            rows.append({'n': n, 'gender': gender, 'best_speaker': best,
                'single_ok': best >= 2 * n, 'cap_per_dataset': per_dataset,
                'group_usable': usable, 'group_ok': usable >= 2 * n,
                'speakers_at_cap': full, 'n_speakers': len(counts)})
    return rows


def suggest_n(records, budget_seconds=300):
    '''Return the vowel count matching budget_seconds at the mean duration.'''
    durations = records['end'] - records['start']
    if len(durations) == 0: return 0
    return int(round(budget_seconds * 1000 / durations.mean()))


def run_vowel_report(output_dir, store_path=locations.cgn_lmdb,
    components=COMPONENTS, exclude_diphthongs=True, min_context_ms=1000,
    n_values=(1000, 2000, 3000, 4000, 5000), cap=0.05, processes=16):
    '''Collect candidates, then save and print the step 1 report.
    output_dir:  receives vowel_candidates.npz and vowel_report.json
    other parameters follow collect_vowel_candidates and feasibility;
    the suggested N (300 s at the pool's mean duration) is added to
    n_values.
    Returns the report dict.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection, records = collect_vowel_candidates(store_path, components,
        'nl', exclude_diphthongs, min_context_ms, processes)
    np.savez(output_dir / 'vowel_candidates.npz', **records)
    speakers = speaker_vowel_report(records)
    suggested = suggest_n(records)
    n_values = sorted(set(n_values) | {suggested})
    report = {'store_path': str(store_path), 'components': components,
        'exclude_diphthongs': exclude_diphthongs,
        'min_context_ms': min_context_ms, 'cap': cap,
        'selection': str(selection), 'n_eligible': len(records['key']),
        'suggested_n': suggested,
        'labels': {k: dict(v.most_common()) for k, v in
            label_counts(records).items()},
        'feasibility': feasibility(speakers, n_values, cap),
        'speakers': speakers}
    with open(output_dir / 'vowel_report.json', 'w') as fout:
        json.dump(report, fout, indent=1, ensure_ascii=False)
    print(format_report(report), flush=True)
    return report


def format_report(report, n_top=10):
    '''Return a readable summary of a run_vowel_report dict.'''
    lines = [report['selection'], '']
    lines.append(f"Eligible vowels: {report['n_eligible']:,}")
    lines.append(f"Suggested N (300 s / mean): {report['suggested_n']:,}")
    lines.append('')
    lines.append('Labels (all): ' + _format_counts(report['labels']['all']))
    lines.append('')
    for gender in GENDERS:
        lines.append(f'Top {n_top} {gender} speakers:')
        lines.extend(_top_speakers(report['speakers'], gender, n_top))
        lines.append('')
    lines.append('Feasibility (single: best speaker >= 2N; group: capped '
        'total >= 2N)')
    for row in report['feasibility']:
        line = f"N={row['n']:>5} {row['gender']:>6}: "
        line += f"best {row['best_speaker']:>6} single "
        line += f"{'yes' if row['single_ok'] else 'no ':>3} | "
        line += f"cap {row['cap_per_dataset']:>4} usable "
        line += f"{row['group_usable']:>7} group "
        line += f"{'yes' if row['group_ok'] else 'no':>3} "
        line += f"({row['speakers_at_cap']}/{row['n_speakers']} at cap)"
        lines.append(line)
    return '\n'.join(lines)


def _format_counts(counts):
    return ', '.join(f'{label} {count:,}' for label, count in counts.items())


def _top_speakers(speakers, gender, n_top):
    '''Lines for the speakers of one gender with the most vowels.'''
    lines = []
    for speaker_id, s in speakers.items():
        if s['gender'] != gender: continue
        components = sorted(s['component_seconds'].items())
        mix = ' '.join(f'{c}:{seconds:.0f}' for c, seconds in components)
        line = f"  {speaker_id} {s['n_vowels']:>6} vowels "
        line += f"{s['vowel_seconds']:>7.1f} s mean {s['mean_ms']:.0f} ms "
        line += f"{s['n_recordings']} rec [{mix}]"
        lines.append(line)
        if len(lines) == n_top: break
    return lines
