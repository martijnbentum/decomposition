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
