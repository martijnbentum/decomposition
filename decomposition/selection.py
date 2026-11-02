'''Select CGN recordings from the phraser store.

Audio durations in the store are in milliseconds; the functions here take
seconds. Components are CGN component letters such as 'o' (read speech),
matched against the comp-<letter> directory in the audio filename.
'''
from phraser import Store

from . import locations


def load_store(store_path=locations.cgn_awd_lmdb):
    '''Return the CGN phraser store.'''
    return Store(store_path)


def load_audios(store=None):
    '''Return all audios in the store as a list.'''
    if store is None: store = load_store()
    return list(store.audios.all())


def filter_on_component(audios, component='o'):
    '''Return the audios from CGN component, e.g. 'o' for comp-o.'''
    pattern = f'/comp-{component}/'
    return [audio for audio in audios if pattern in audio.filename]


def filter_on_duration(audios, min_seconds=0, max_seconds=None):
    '''Return the audios with min_seconds <= duration <= max_seconds.'''
    selected = []
    for audio in audios:
        seconds = audio.duration / 1000
        if seconds < min_seconds: continue
        if max_seconds is not None and seconds > max_seconds: continue
        selected.append(audio)
    return selected


def make_gender_speaker_audio_dict(audios, store=None):
    '''Return a dict gender -> speaker_id -> list of audios.
    audios:    audios to include; speakers not in any of them are skipped
    store:     phraser store, loaded with load_store if None

    speaker_id is the CGN speaker name (e.g. N01234); gender is 'female',
    'male' or 'unknown'.
    '''
    if store is None: store = load_store()
    audio_set = set(audios)
    d = {}
    for speaker in store.speakers.all():
        speaker_audios = [a for a in speaker.audios if a in audio_set]
        if not speaker_audios: continue
        speakers = d.setdefault(speaker.gender(), {})
        speakers[speaker.name] = speaker_audios
    return d
