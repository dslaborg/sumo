import argparse
from pathlib import Path
from sys import path
from typing import Union

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import zscore
from torch.utils.data import Dataset, DataLoader

from a7.butter_filter import butter_bandpass_filter, downsample

# append root dir to python path so that we find `sumo`
path.insert(0, str(Path(__file__).absolute().parents[1]))
from sumo.config import Config
from sumo.model import SUMO


def get_model(path: Union[str, Path]):
    path = Path(path)

    model_file = path if path.is_file() else get_best_model(path)
    if gpu:
        model_checkpoint = torch.load(model_file)
    else:
        model_checkpoint = torch.load(model_file, map_location='cpu')

    model = SUMO(config)
    model.load_state_dict(model_checkpoint['state_dict'])

    return model


def get_best_model(experiment_path: Path, sort_by_loss: bool = False):
    models_path = experiment_path / 'models'
    models = list(models_path.glob('epoch=*.ckpt'))

    regex = r'.*val_loss=(0\.[0-9]+).*\.ckpt' if sort_by_loss else r'.*val_f1_mean=(0\.[0-9]+).*\.ckpt'
    regex_results = [re.search(regex, str(m)) for m in models]

    models_score = np.array([float(r.group(1)) for r in regex_results])
    model_idx = np.argmin(models_score) if sort_by_loss else np.argmax(models_score)

    return models[model_idx]


class SimpleDataset(Dataset):
    def __init__(self, data_vectors):
        super(SimpleDataset, self).__init__()

        self.data = data_vectors

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def preprocess(data):
        return zscore(data)

    def __getitem__(self, idx):
        data = self.preprocess(self.data[idx])
        return torch.from_numpy(data).float(), torch.zeros(0)


def get_args():
    # synthetic input data
    default_data_path = (Path(__file__).absolute().parents[1] / 'input' / 'eeg-sample.npy').__str__()
    default_model_path = (Path(__file__).absolute().parents[1] / 'output' / 'final.ckpt').__str__()

    parser = argparse.ArgumentParser(description='Evaluate a UTime model on any given eeg data')
    parser.add_argument('-d', '--data_path', type=str, default=default_data_path, help='Path to input data, given in \
    .pickle or .npy format as a dict with the channel name as key and the eeg data as value')
    parser.add_argument('-sr', '--sample_rate', type=float, default=100.0,
                        help='Rate with which the given data was sampled')
    parser.add_argument('-m', '--model_path', type=str, default=default_model_path,
                        help='Path to the model checkpoint used for evaluating')
    parser.add_argument('-g', '--gpu', action='store_true', default=False,
                        help='If a GPU should be used')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    data_path = args.data_path
    sr = args.sample_rate
    model_path = args.model_path
    gpu = args.gpu

    resample_rate = 100.0

    config = Config('predict', create_dirs=False)

    data = np.load(data_path, allow_pickle=True)
    if type(data) is np.ndarray:
        data = data.tolist()

    channels = data.keys()
    eegs = list(data.values())

    eegs = [downsample(butter_bandpass_filter(x, 0.3, 30.0, sr, 10), sr, resample_rate) for x in eegs]

    dataset = SimpleDataset(eegs)
    dataloader = DataLoader(dataset)

    model = get_model(model_path)

    trainer = pl.Trainer(gpus=int(gpu), num_sanity_val_steps=0, logger=False)
    predictions = trainer.predict(model, dataloader)

    result = dict(zip(channels, [prediction[0].numpy() for prediction in predictions]))
    output_dir = Path(__file__).absolute().parents[1] / 'output'
    np.save(output_dir / 'result.npy', result)
