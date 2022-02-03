import pickle
from typing import Iterable, List

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.io import loadmat
from scipy.stats import zscore
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import ConcatDataset, Dataset, DataLoader

from sumo.config import Config


def spindle_vect_to_indices(x):
    """
    Convert a spindle vector to indices.

    Convert a vector of zeros and ones, indicating the existence of a spindle for each sample, to a vector of start
    and stop samples for each spindle. A spindle is defined as one or more consecutive ones in the input spindle
    vector.

    Parameters
    ----------
    x : ndarray
        The input spindle vector consisting of ones and zeros; format (n_samples,).

    Returns
    -------
    spindle_indices : ndarray
        The start index (included) and stop index (excluded) for each contained spindle; format (n_spindles, 2).
    """

    diff = np.diff(np.r_[0, x, 0])  # be able to detect spindles at the start and end of vector
    return np.c_[np.argwhere(diff == 1), np.argwhere(diff == -1)]


class Subject(Dataset):
    """
    Class representing a single subject of the MODA study with recorded eeg data and corresponding annotated spindles.

    Parameters
    ----------
    data_vectors : List
        List containing the recorded eeg data of this subject in 115 second blocks as ndarray; format (n_blocks, 11500).
    spindle_vectors : List
        List containing the spindle vector for each of the given 115 second blocks as ndarray; format (n_blocks, 11500).
    phase : int
        The cohort or phase of this subject, either zero for the younger or one for the older cohort.
    patient_id : str
        The unique patient ID of this subject, as used in the MASS dataset.
    """

    def __init__(self, data_vectors: List, spindle_vectors: List, phase: int, patient_id: str) -> None:
        super(Subject, self).__init__()
        assert len(data_vectors) == len(spindle_vectors)

        self.data = data_vectors
        self.spindles = spindle_vectors
        self.phase = phase
        self.patient_id = patient_id

    def __len__(self) -> int:
        """
        Return the size of this dataset, defined by the amount of blocks given.

        Returns
        -------
        len : int
            The size of this dataset.
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the appropriate element for the given index.

        Parameters
        ----------
        idx : int
            Index of the element to be returned.

        Returns
        -------
        data : ndarray
            The block of 115 second long eeg data at the given index; format (11500,).
        spindle : ndarray
            The corresponding spindle vector for the block of eeg data; format (11500,).
        """

        data = self.data[idx]
        spindle = self.spindles[idx]

        return data, spindle


class MODADataset(ConcatDataset[Subject]):
    """
    Class representing a dataset consisting of multiple MODA subjects.

    Implemented as a ConcatDataset, which concatenates the elements (blocks) of each element (subject) to one dataset.

    Parameters
    ----------
    subjects : Iterable[Subject]
        The subjects contained in this dataset.
    preprocessing : bool
        If the data should be preprocessed using the z-score before returned.
    """

    def __init__(self, subjects: Iterable[Subject], preprocessing: bool) -> None:
        super(MODADataset, self).__init__(subjects)

        self.preprocessing = preprocessing

    def preprocess(self, data):
        """
        Preprocess the given data in case `self.preprocessing` is True.

        Parameters
        ----------
        data : ndarray
            The block of eeg data to be preprocessed; format (11500,).

        Returns
        -------
        processed_data : ndarray
            The processed data in case `self.preprocessing` is True, `data` otherwise; format (11500,).
        """

        return zscore(data) if self.preprocessing else data

    def __getitem__(self, idx):
        """
        Return the block of eeg data at the given index as a tensor.

        Parameters
        ----------
        idx : int
            Index of the element to be returned.

        Returns
        -------
        data : torch.Tensor
            The block of 115 second long eeg data at the given index; format (11500,).
        spindle : torch.Tensor
            The corresponding spindle vector for the block of eeg data; format (11500,).
        """

        data, spindle = super(MODADataset, self).__getitem__(idx)

        data = self.preprocess(data)

        return torch.from_numpy(data).float(), torch.from_numpy(spindle).long()


class MODADataModule(pl.LightningDataModule):
    """
    Class representing a data module (as used by pytorch lightning) for the MODA study.

    Parameters
    ----------
    config : sumo.config.Config
        The config to be used by the data module, containing e.g. the batch size.
    """

    def __init__(self, config: Config):
        super(MODADataModule, self).__init__()

        self.config = config
        self.preprocessing = config.preprocessing
        self.batch_size = config.batch_size

        self.subjects_train = None
        self.subjects_val = None
        self.subjects_test = None

    def prepare_data(self):
        with open(self.config.data_file, 'rb') as data_file:
            subjects = pickle.load(data_file)  # load the configured file containing the subjects
        self.subjects_train = subjects[self.config.train_split_name]
        self.subjects_val = subjects[self.config.val_split_name]

        if self.config.test_split_name is not None:
            self.subjects_test = subjects[self.config.test_split_name]

    def setup(self, stage=None):
        pass

    def subjects_to_data_loader(self, subjects, shuffle=False):
        """
        Return a data loader for the given subjects following the given configuration.

        Parameters
        ----------
        subjects : List
            The subjects to be included in the data loader.
        shuffle : bool
            If the data loader should shuffle the given subjects.

        Returns
        -------
        dataloader : torch.utils.data.DataLoader
            The data loader containing the given subjects.
        """

        dataset = MODADataset(subjects, preprocessing=self.preprocessing)
        # use num_workers=0 as this is for some reason faster than using multiple workers
        return DataLoader(dataset, self.batch_size, shuffle, num_workers=0, pin_memory=True)

    def train_dataloader(self):
        return self.subjects_to_data_loader(self.subjects_train, shuffle=True)

    def val_dataloader(self):
        return self.subjects_to_data_loader(self.subjects_val)

    def test_dataloader(self):
        assert self.subjects_test is not None
        assert len(self.subjects_test) > 0

        if type(self.subjects_test[0]) is list:  # multiple test datasets
            return [self.subjects_to_data_loader(test_phase) for test_phase in self.subjects_test]
        else:
            return self.subjects_to_data_loader(self.subjects_test)


class MODADataModuleCV(MODADataModule):
    """
    Class representing a data module (as used by pytorch lightning) for the MODA study when using cross validation.

    Parameters
    ----------
    config : sumo.config.Config
        The config to be used by the data module, containing e.g. the batch size.
    fold_idx : int
        The index of the fold to be used as validation data in this data module.
    """

    def __init__(self, config: Config, fold_idx: int):
        super(MODADataModuleCV, self).__init__(config)

        self.fold_idx = fold_idx

    def prepare_data(self):
        with open(self.config.data_file, 'rb') as data_file:
            subjects = pickle.load(data_file)
        cv_folds = self.config.cross_validation
        k = len(cv_folds)

        # use the fold at fold_idx as validation data and the remaining folds as training data
        self.subjects_train = sum([subjects[cv_folds[i]] for i in range(k) if i != self.fold_idx], [])
        self.subjects_val = subjects[cv_folds[self.fold_idx]]

        if self.config.test_split_name is not None:
            self.subjects_test = subjects[self.config.test_split_name]


def split_and_remove_nan(x):
    """
    Helper function to transform a vector as returned by the MODA project in one vector per block.

    Splits the data/spindle expert consensus vector as generated by the `MODA02_genEEGVectBlock.m` file
    at https://github.com/klacourse/MODA_GC in a list of 115 second vectors. Removes the "missing" block of
    one subject, which contains only NaN values.

    Parameters
    ----------
    x : ndarray
        The eeg data or spindle vector of all blocks of all subjects as created by the code in the MODA_GC
        repository; there is one data and one spindle vector for each cohort; format (k*11500+k,) with k
        the number of blocks in the cohort.

    Returns
    -------
    x_split : ndarray
        The input data split up in 115 second blocks, with "empty" blocks removed; format (k, 11500)
        with k the number of blocks in the cohort (k+1 for cohort 1 due to the empty, filtered block).
    """

    split_up = np.split(x, np.where(np.isnan(x))[0])
    return [arr[~np.isnan(arr)] for arr in split_up if ~np.all(np.isnan(arr))]


def create_subjects(eeg_files, spindle_files, block_files):
    """
    Loads the MODA data and spindle vector and creates Subject objects.

    Using the Matlab files of the eeg data, annotated spindles and used blocks creates Subject objects
    for each cohort.

    Parameters
    ----------
    eeg_files : List
        The paths to the Matlab files containing the eeg data vectors as produced by the `MODA02_genEEGVectBlock.m`
        file at https://github.com/klacourse/MODA_GC.
    spindle_files : List
        The paths to the Matlab files containing the annotated spindle vectors as produced by the
        `MODA02_genEEGVectBlock.m` file at https://github.com/klacourse/MODA_GC.
    block_files : List
        The paths to the .txt files containing descriptions of the used blocks as produced by the
        `MODA02_genEEGVectBlock.m` file at https://github.com/klacourse/MODA_GC.

    Returns
    -------
    subjects : List
        The Subject objects created using the given input data, with one ndarray for each cohort.
    """

    subjects = []
    for idx, (eeg_file, spindle_file, block_file) in enumerate(zip(eeg_files, spindle_files, block_files)):
        eeg_data = loadmat(eeg_file)['EEGvector']
        spindle_vector = loadmat(spindle_file)['GCVect']

        # split the data and spindle vectors into blocks
        data_vectors = split_and_remove_nan(eeg_data)
        spindle_vectors = split_and_remove_nan(spindle_vector)

        # load all the used subject ids
        subject_per_block = np.loadtxt(block_file, str, skiprows=1, usecols=1)
        # get the first index of each subject in the list of blocks and sort them ascending
        _, unique_indices = np.unique(subject_per_block, return_index=True)
        unique_indices.sort()

        # group the blocks by subject
        data_vectors_per_subject = np.split(data_vectors, unique_indices[1:], axis=0)  # type: ignore
        spindle_vectors_per_subject = np.split(spindle_vectors, unique_indices[1:], axis=0)  # type: ignore

        subjects.append(np.array(
            [Subject(subject_data_vector, subject_spindle_vector, phase=idx, patient_id=subject_per_block[block_idx])
             for subject_data_vector, subject_spindle_vector, block_idx in
             zip(data_vectors_per_subject, spindle_vectors_per_subject, unique_indices)], dtype=Subject))

    return subjects


def create_train_test(subjects, test_size, only_small_subjects_in_test=True):
    """
    Create a random distribution of the subjects in train and test data.

    The split is always done on a subject level, the proportion of test data can be configured and optionally only
    "small" subjects (those with only three blocks) are used for the test set. The test set contains the same
    (absolute) amount of subjects for each cohort.

    Parameters
    ----------
    subjects : List
        The Subject objects with one ndarray per cohort.
    test_size : float
        The proportion of the data to be used as test data, between 0 and 1.
    only_small_subjects_in_test : bool
        If only the subjects with 3 (or fewer) blocks should be used for the test data; True by default.

    Returns
    -------
    subjects_train_all : ndarray
        All the training subjects, combined over all cohorts and shuffled; format (n_subjects - n_test,).
    subjects_test_all : ndarray
        The test subjects, with one row per cohort; format (n_cohorts, n_test_per_phase).
    """

    assert 0 <= test_size <= 1, 'the test_size has to be between 0 and 1 (both inclusive)'

    n_subjects = sum([len(s) for s in subjects])
    n_test = int(n_subjects * test_size)
    n_test_per_phase = n_test // len(subjects)  # absolute size of test data equal in all cohorts

    subjects_train_all = np.empty(0, dtype=Subject)
    subjects_test_all = np.empty((len(subjects), n_test_per_phase), dtype=Subject)

    for phase, subjects_phase in enumerate(subjects):
        if only_small_subjects_in_test:
            # use only the subjects with 3 (or 2 for one subject) blocks for the test data
            mask = np.array([len(s) for s in subjects_phase]) <= 3
        else:
            # use all subjects for the test data
            mask = np.ones_like(subjects_phase, dtype=bool)

        subjects_train, subjects_test = train_test_split(subjects_phase[mask], test_size=n_test_per_phase)
        if only_small_subjects_in_test:
            subjects_train = np.r_[subjects_train, subjects_phase[~mask]]

        subjects_train_all = np.r_[subjects_train_all, subjects_train]
        subjects_test_all[phase] = subjects_test

    np.random.default_rng().shuffle(subjects_train_all)
    return subjects_train_all, subjects_test_all


def create_train_val(subjects, val_size):
    """
    Splits the given (non-test) data into train and validation data, stratified by the cohort.

    Parameters
    ----------
    subjects : ndarray
        All subjects excluding the test data; format (n_train_val,).
    val_size : float
        The proportion of the (non-test) data to be used as validation data, between 0 and 1.

    Returns
    -------
    subjects_train : ndarray
        The subjects used for training; format (n_train,).
    subjects_val : ndarray
        The subjects used for validation; format (n_val,).
    """

    phases = [s.phase for s in subjects]
    subjects_train, subjects_val = train_test_split(subjects, test_size=val_size, stratify=phases)

    return subjects_train, subjects_val


def create_train_val_test(subjects, test_size, val_size, only_small_subjects=True):
    """
    Splits the given subjects into train, validation and test data.

    First the test data is created, creating one subset for each cohort, where each subset has the same size and the
    combined size is `test_size` of the size of all subjects. In case `only_small_subjects` is True, only the subjects
    with three (or fewer) blocks are used for the test data. Afterwards the remaining non-test data is split, so that
    `val_size` of the remaining subjects are used as validation data and the rest as train data.

    Parameters
    ----------
    subjects : List
        All subjects, with one ndarray per cohort.
    test_size : float
        The proportion of the data to be used as test data, between 0 and 1.
    val_size : float
        The proportion of the (remaining non-test) data to be used as validation data, between 0 and 1.
    only_small_subjects : bool
        If only the subjects with 3 (or fewer) blocks should be used for the test data; True by default.

    Returns
    -------
    subjects_train : ndarray
        The subjects used for training; format (n_train,).
    subjects_val : ndarray
        The subjects used for validation; format (n_val,).
    subjects_test : ndarray
        The subjects used for testing; format (n_cohorts, n_test_per_phase).
    """

    subjects_train, subjects_test = create_train_test(subjects, test_size, only_small_subjects)
    subjects_train, subjects_val = create_train_val(subjects_train, val_size)

    return subjects_train, subjects_val, subjects_test


def create_cv_folds(subjects, k):
    """
    Creates a split of the given data following a cross validation procedure.

    Splits the subjects into `k` folds, so that each fold has the same amount of subjects with 10 and with 3 blocks.
    Additionally, the folds are stratified by the cohorts, so that the proportion of the cohorts is similar across the
    folds. Afterwards two lists are created with `k` ndarrays, where the `test_folds` list contains the i-th fold and
    the `train_folds` list contains all but the i-th fold (1<=i<=k).

    Parameters
    ----------
    subjects : ndarray
        All subjects excluding the test data; format (n_train_val,).
    k : int
        The number of validation folds to be used.

    Returns
    -------
    train_folds : List
        A list of size k with one ndarray for each fold, where each ndarray contains the training data for this step of
        the cross validation (so all folds but the one in the corresponding ndarray in `test_folds`).
    test_folds : List
        A list of size k with one ndarray for each fold, where each ndarray contains the validation/test data for this
        step of the cross validation (so exactly one fold).
    """

    # split data into subjects with 10 and 3 (or fewer) blocks
    mask = np.array([len(s) for s in subjects]) <= 3
    subjects1 = subjects[mask]
    subjects2 = subjects[~mask]

    # generate folds for each split, stratified by the cohort
    kfold = StratifiedKFold(n_splits=k, shuffle=True)
    generator1 = kfold.split(subjects1, [s.phase for s in subjects1])
    generator2 = kfold.split(subjects2, [s.phase for s in subjects2])

    train_folds, test_folds = [], []
    for ((train_fold_idx1, test_fold_idx1), (train_fold_idx2, test_fold_idx2)) in zip(generator1, generator2):
        # combine the two corresponding splits
        train_fold = np.concatenate((subjects1[train_fold_idx1], subjects2[train_fold_idx2]))
        test_fold = np.concatenate((subjects1[test_fold_idx1], subjects2[test_fold_idx2]))

        train_folds.append(train_fold)
        test_folds.append(test_fold)

    return train_folds, test_folds
