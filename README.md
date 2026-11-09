# decomposition

Continuous hidden-state extraction for speech audio. A model (wav2vec2,
HuBERT, ...) is applied to overlapping windows of a stretch of audio and
only the center frames of each window are kept, so every frame has context
on both sides. Windows start on the 20 ms frame grid, so the center frames
join into one continuous, frame-aligned matrix.

## Install

```bash
uv venv .venv --python 3.12
uv sync
```

## Example

```python
from decomposition import locations
from decomposition import compute_windowed_embeddings
from decomposition import save_windowed_embeddings, select_recording

audio_filename = select_recording()   # first comp-o recording >= 200 s
model_name = locations.wav2vec2_nl1_name
embeddings, times = compute_windowed_embeddings(audio_filename, model_name,
    start=60.0, duration=180.0, layer=9, context=2.0, center=16.0)
path = save_windowed_embeddings(embeddings, times, audio_filename,
    model_name, start=60.0, duration=180.0, layer=9, context=2.0,
    center=16.0, output_dir='.')
# e.g. fn001234_s-60_dur-180_wav2vec2_nl1_checkpoint-200000_l-9.npz
```

`embeddings` has shape `(n_frames, dim)`; `times` holds each frame's start
time in seconds. `start`, `duration`, `context` and `center` must be
multiples of 0.02 s.

`locations.py` holds the CGN phraser store paths and the pretrained Dutch
checkpoint roots; `locations.model_name_to_path(name)` recreates the full
checkpoint path from a name such as `wav2vec2_nl1_checkpoint-200000`.

## Balanced speaker PCA experiment

```python
from decomposition import compute_speaker_pcas

paths = compute_speaker_pcas(output_dir='pca_outputs', n_male=10,
    n_female=10, minutes=range(1, 11), start=10.0, layer=9,
    context=2.0, center=16.0, gpu=True, spawn=True, progress=True)
```

Selects one distinct recording per speaker, using CGN component `o` by
default. The defaults require recordings of at least 620 seconds, including
window padding and context. Selection uses sorted speaker IDs and filenames;
recordings may contain additional speakers. Missing speaker quotas raise an
error before model loading.

By default, one spawned worker loads the model once and runs the whole
experiment. Each recording's longest stretch is extracted once in evaluation
mode; shorter durations use prefixes on the same window grid. Only one
recording's embeddings are retained at a time. Only PCA results and
extraction metadata (including speaker ID and gender) are saved: 200 `.npz`
files with the default settings. Existing files with matching names are
overwritten. Use `pca.load_pca(path)` to load a result. Corpus paths and model
checkpoints must be accessible locally.

Progress is forwarded to the calling session, including IPython: model
loading, recording number, completed extraction windows, and saved PCA
durations. Set `progress=False` to suppress these reports. Worker failures
raise an error with the child traceback; Ctrl-C terminates and joins the
worker. Files already saved remain available after failure or interruption.

The call returns saved paths after the worker exits, releasing its CUDA
context and GPU allocations. CUDA allocations previously created in the
parent session are unaffected. Set `spawn=False` for execution in the calling
process, which does not provide this GPU cleanup guarantee. This wrapper
works with the existing `to_vector` inference API and does not require its
newer per-call spawning API.

In IPython, import and call the function as shown above. In a script, protect
the call so the spawned worker does not start another experiment:

```python
from decomposition import compute_speaker_pcas

if __name__ == '__main__':
    paths = compute_speaker_pcas()
```
