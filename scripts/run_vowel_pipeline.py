'''Run one step of the vowel PCA experiment; outputs go below vowel_root.

usage: run_vowel_pipeline.py report [phraser_store_path]
       run_vowel_pipeline.py sets
       run_vowel_pipeline.py embed|check|pca wav2vec2|huibert

Run embed on a GPU host, one model per GPU (CUDA_VISIBLE_DEVICES); the
other steps need only a CPU.
'''
import sys

from decomposition import locations, vowel_embeddings, vowel_pca
from decomposition import vowel_sets, vowels


def main(step, argument=None):
    if step == 'report':
        store_path = argument or locations.cgn_lmdb
        vowels.run_vowel_report(locations.vowel_report_dir, store_path,
            n_values=(4000, 8000, 16000))
    elif step == 'sets': vowel_sets.run_vowel_sampling()
    elif step == 'embed':
        vowel_embeddings.compute_vowel_embeddings(argument)
        check(argument)
    elif step == 'check': check(argument)
    elif step == 'pca': vowel_pca.run_vowel_pcas(argument)
    else: raise ValueError(f'unknown step {step!r}\n{__doc__}')


def check(architecture):
    '''Compare the stored frame counts of 200 random vowels.'''
    problems = vowel_embeddings.check_vowel_embeddings(architecture, 200)
    print(f'check: {len(problems)} problems {problems}', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
