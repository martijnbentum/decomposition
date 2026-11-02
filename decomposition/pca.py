'''Vanilla PCA on windowed embeddings.

Uses scikit-learn's PCA with all components kept; for tall data (many more
frames than dimensions) it eigendecomposes the covariance matrix. Results are
computed in float64. Eigenvectors are stored as columns, so
eigenvectors[:, i] belongs to eigenvalues[i], sorted by descending eigenvalue.
'''
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def compute_pca(embeddings):
    '''Return a dict with covariance, eigenvalues, eigenvectors (columns),
    explained_variance_ratio and mean for embeddings (n_frames, dim).
    '''
    pca = PCA().fit(embeddings.astype(np.float64))
    return {'covariance': pca.get_covariance(),
        'eigenvalues': pca.explained_variance_,
        'eigenvectors': pca.components_.T,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'mean': pca.mean_}


def make_pca_filename(embeddings_filename):
    '''Return e.g. fn001234_s-60_dur-180_<model>_l-9_pca.npz.'''
    return f'{Path(embeddings_filename).stem}_pca.npz'


def pca_embeddings_file(embeddings_filename, output_dir='.'):
    '''Compute PCA on a saved embeddings .npz and save the result.
    embeddings_filename:   .npz written by save_windowed_embeddings
    output_dir:            directory for the _pca.npz output

    The output also holds the source filename and extraction settings.
    Returns the output path.
    '''
    d = np.load(embeddings_filename)
    result = compute_pca(d['embeddings'])
    path = Path(output_dir) / make_pca_filename(embeddings_filename)
    np.savez(path, **result, source=str(embeddings_filename),
        audio_filename=d['audio_filename'], model_name=d['model_name'],
        start=d['start'], duration=d['duration'], layer=d['layer'],
        context=d['context'], center=d['center'])
    return path


def load_pca(pca_filename):
    '''Return a dict with the arrays and settings of a _pca.npz file.
    Scalar settings (model_name, start, layer, ...) are returned as plain
    Python values instead of 0-d arrays.
    '''
    d = {}
    with np.load(pca_filename) as data:
        for name in data.files:
            value = data[name]
            if value.ndim == 0: value = value.item()
            d[name] = value
    return d
