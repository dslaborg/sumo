import argparse
import datetime
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path
from sys import path

import numpy as np

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[1]))
from sumo.data import Subject
from sumo.data.data import create_subjects, create_train_test, create_train_val, create_cv_folds
from helper.a7_parallel_evaluation import split_evaluation


def get_args():
    parser = argparse.ArgumentParser(description='Create dataset splits, evaluate them on A7 algorithm and choose split'
                                                 'with median performance of A7',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_input_dir = Path(__file__).absolute().parents[1] / 'input'
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output_dir = Path(__file__).absolute().parents[1] / 'output' / f'datasets_{date_time}'

    parser.add_argument('-i', '--input', type=str, default=default_input_dir,
                        help='Directory with stored input files as generated by the MODA file MODA02_genEEGVectBlock.m')
    parser.add_argument('-n', '--n_datasets', type=int, default=25, help='Number of test datasets to create and save')
    parser.add_argument('-t', '--test', type=float, default=0.2,
                        help='Proportion of the data that is used as test data (0-1)')
    parser.add_argument('-o', '--output', type=str, default=default_output_dir,
                        help='Output directory to save generated data splits in')

    return parser.parse_args()


def create_random_splits():
    eeg_files = [str(input_dir / 'EEGVect_p1.mat'), str(input_dir / 'EEGVect_p2.mat')]
    spindle_files = [str(input_dir / 'GCVect_exp_p1.mat'), str(input_dir / 'GCVect_exp_p2.mat')]
    block_files = [str(input_dir / '6_segListSrcDataLoc_p1.txt'), str(input_dir / '7_segListSrcDataLoc_p2.txt')]

    subjects = create_subjects(eeg_files, spindle_files, block_files)

    for i in range(args.n_datasets):
        subjects_train, subjects_test = create_train_test(subjects, args.test)
        output_dict = {'train': subjects_train.tolist(), 'test': subjects_test.tolist()}
        with open(output_dir / f'split_{i:02d}.pickle', 'wb') as output_file:
            pickle.dump(output_dict, output_file)


def choose_split():
    input_files = sorted(glob(str(output_dir / 'split_*.pickle')))

    f1s = []
    with ProcessPoolExecutor() as pool:
        for f1_mean in pool.map(split_evaluation, input_files):
            f1s.append(f1_mean)

    final_split_idx = np.where(f1s == np.median(f1s))[0][0]
    shutil.copy(input_files[final_split_idx], output_dir / 'final_split.pickle')

    return final_split_idx, f1s


if __name__ == '__main__':
    args = get_args()

    assert 0 <= args.test <= 1

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    create_random_splits()  # create n_datasets splits as described in our paper
    split_idx, f1s = choose_split()  # evaluate A7 on the test data per split and choose split with median performance

    with open(output_dir / 'final_split.pickle', 'rb') as final_split_path:
        final_split = pickle.load(final_split_path)

    # create hold-out validation set, to demonstrate create_train_val function
    subjects_train, subjects_val = create_train_val(final_split['train'], 0.1)
    output_dict = {'train': subjects_train, 'val': subjects_val, 'test': final_split['test']}
    with open(output_dir / 'final_split_val.pickle', 'wb') as output_path:
        pickle.dump(output_dict, output_path)

    # create cross validation folds, to demonstrate create_cv_folds function
    _, val_folds = create_cv_folds(np.array(final_split['train'], dtype=Subject), 6)
    output_dict = {}
    for i, val_fold in enumerate(val_folds):
        output_dict[f'fold_{i}'] = val_fold
    output_dict['test'] = final_split['test']
    with open(output_dir / 'final_split_cv.pickle', 'wb') as output_path:
        pickle.dump(output_dict, output_path)
