import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from sumo.config import Config
from sumo.evaluation.performance_analysis import PerformanceEvaluation, metric_scores
from .model_parts import Decoder, DoubleConv, Encoder


class SUMO(pl.LightningModule):
    """
    Model definition of the Slim U-Net trained on MODA (SUMO).

    The model is an adaption of the U-Net architecture for sleep spindle detection, as described in our paper:
        Lars Kaulen, Justus T.C. Schwabedal, Jules Schneider, Philipp Ritter and Stephan Bialonski.
        Advanced sleep spindle identification with neural networks. Sci Rep 12, 7686 (2022).
        https://doi.org/10.1038/s41598-022-11210-y

    U-Net architecture is defined in:
        Olaf Ronneberger, Philipp Fischer and Thomas Brox.
        U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI (2015).
        https://doi.org/10.1007/978-3-319-24574-4_28
    """

    def __init__(self, config: Config):
        super(SUMO, self).__init__()

        self.save_hyperparameters()

        # save configured training classes and properties/parameters
        self.moving_avg_size = config.moving_avg_size
        self.early_stopping = config.early_stopping
        self.criterion = config.loss(**config.loss_params)
        self.optimizer = config.optimizer
        self.optimizer_params = config.optimizer_params
        self.lr_scheduler = config.lr_scheduler
        self.lr_scheduler_params = config.lr_scheduler_params

        # save configured validation properties/parameters
        self.overlap_thresholds = config.overlap_thresholds

        # configure the model using the configured properties/parameters
        model_params = config.convolution_params.copy()
        model_params['activation'] = config.activation()

        depth = config.depth
        chs = config.channel_size
        pools = config.pools
        self.pools = pools
        n_classes = config.n_classes

        # create encoders, doubling the number of used channels with each encoder
        self.inc = DoubleConv(1, chs, **model_params)
        self.encoders = nn.ModuleList(
            [Encoder(chs * 2 ** i, chs * 2 ** (i + 1), pools[i], **model_params) for i in range(depth)])

        # decoders don't use dilation
        model_params['dilation'] = 1
        # create decoders, halving the number of used channels with each decoder
        self.decoders = nn.ModuleList(
            [Decoder(chs * 2 ** (i + 1), chs * 2 ** i, pools[i], **model_params) for i in reversed(range(depth))])

        # create convolution for dense segmentation using as many channels as classes to be classified
        self.dense = nn.Sequential(
            nn.Conv1d(chs, n_classes, kernel_size=1, padding='same'),  # type: ignore
            model_params['activation']
        )

    def dense_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the U-part of the U-Net architecture, i.e. the encoders and decoders.

        First performs an extrapolation on the input data using mirroring, to expand the input data to L elements per
        observation (L>=K), where L is the necessary number of elements to downsample (or rather maxpool) the data
        without "losing" elements due to missing padding.
        Afterwards the outputs of the encoders and decoders are calculated, where each encoder output is concatenated to
        the appropriate decoder input (skip connections).
        The previously extrapolated samples are cropped again and with these outputs the dense segmentation is
        calculated and returned, providing a value for each class per element.

        Parameters
        ----------
        x : torch.Tensor
            The input data in format [N,K] where N is the batch size and K the number of elements in each observation.

        Returns
        -------
        logits : torch.Tensor
            The calculated logits in format [N,C,K] where N is the batch size, C the number of classes and K the number
            of elements in each observation.
        """

        n_samples = x.shape[1]

        # add the filters/channels as second dimension
        x = x.unsqueeze(1)

        # extrapolate to the necessary number of elements by calculating the width of the lowest layer (without
        # rounding), rounding it up and calculating the output width using this rounded up value
        extrapolation = int(np.ceil(n_samples / np.prod(self.pools)) * np.prod(self.pools) - n_samples)
        # extrapolation is done using the reflection/mirroring mode
        x = F.pad(x, (extrapolation // 2, extrapolation // 2 + extrapolation % 2), mode='reflect')

        # save outputs of all encoders except the last one
        features_enc = []
        x = self.inc(x)
        for enc in self.encoders:
            features_enc.append(x)
            x = enc(x)

        # reverse the encoder outputs (inplace)
        features_enc.reverse()
        # calculate output of all decoders, using the encoder outputs as skip connections
        for dec, x_enc in zip(self.decoders, features_enc):
            x = dec(x, x_enc)

        # remove the previously added/extrapolated elements again
        x = self.crop(x, n_samples)
        # calculate the dense segmentation using the last convolution
        logits = self.dense(x)

        return logits

    def postprocessing(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Average the calculated logits using a (non-trainable) sliding window approach.

        The previously calculated logits (see `dense_logits`) are averaged using a sliding window with step size one and
        the configured window width.

        Parameters
        ----------
        logits : torch.Tensor
            The values as calculated by `dense_logits` in format [N,C,K] where N is the batch size, C the number of
            classes and K the number of elements in each observation.

        Returns
        -------
        averaged_logits : torch.Tensor
            The averaged results in format [N,C,K] where N is the batch size, C the number of classes and K the number
            of elements in each observation.
        """

        # zero padding before moving average
        s = self.moving_avg_size - 1
        logits = F.pad(logits, (s // 2, s // 2 + s % 2), mode='constant', value=0)

        return F.avg_pool1d(logits, self.moving_avg_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines dense logit calculation and subsequent moving average.

        Parameters
        ----------
        x : torch.Tensor
            The input data in format [N,K] where N is the batch size and K the number of elements in each observation.

        Returns
        -------
        averaged_logits : torch.Tensor
            The averaged results in format [N,C,K] where N is the batch size, C the number of classes and K the number
            of elements in each observation.
        """

        # calculate the dense logits
        logits = self.dense_logits(x)
        # return the averaged logits
        return self.postprocessing(logits)

    def get_batch_results(self, logits: torch.Tensor, mask: torch.Tensor):
        """
        Calculate total number of detected, gold standard and true positive spindles in the given batch.

        Parameters
        ----------
        logits : torch.Tensor
            The values as calculated by `dense_logits` in format [N,C,K] where N is the batch size, C the number of
            classes and K the number of elements in each observation.
        mask : torch.Tensor
            The ground truth values in format [N,K] where N is the batch size and K the number of elements in each
            observation.
        Returns
        -------
        results : dict
            Dictionary containing the detected, gold standard and true positive (one value per overlap threshold)
            spindles in the given batch.
        """

        # apply averaging to calculate actual predictions
        averaged_logits = self.postprocessing(logits)

        # transform predictions to spindle vectors using softmax and transfer them to numpy arrays
        spindles_pred_batch = F.softmax(averaged_logits, dim=1).argmax(dim=1).long().detach().cpu().numpy()
        spindles_gs_batch = mask.long().detach().cpu().numpy()

        n_spindles_detected, n_spindles_gs, n_true_positives = 0, 0, np.zeros_like(self.overlap_thresholds, dtype=int)
        for spindles_pred, spindles_gs in zip(spindles_pred_batch, spindles_gs_batch):
            res = PerformanceEvaluation(spindles_pred, spindles_gs, self.overlap_thresholds).evaluate_performance()
            n_spindles_detected += res[0]
            n_spindles_gs += res[1]
            n_true_positives += res[2]

        return {'detected': n_spindles_detected, 'gold_standard': n_spindles_gs, 'tp': n_true_positives}

    def training_step(self, batch, batch_idx):
        data, mask = batch
        # training is performed on dense logits, therefore the (training) loss is calculated using these
        logits = self.dense_logits(data)

        loss = self.criterion(logits, mask)
        self.log('loss/train', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        data, mask = batch
        # return the predicted spindles after postprocessing and transformation by softmax, by applying argmax
        return F.softmax(self(data), dim=1).argmax(dim=1).long()

    def validation_step(self, batch, batch_idx):
        data, mask = batch
        # training is performed on dense logits, therefore the (validation) loss is calculated using these
        logits = self.dense_logits(data)

        loss = self.criterion(logits, mask)
        self.log('loss/val', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return self.get_batch_results(logits, mask)

    def validation_epoch_end(self, outputs):
        n_spindles_detected = sum([x['detected'] for x in outputs])
        n_spindles_gs = sum([x['gold_standard'] for x in outputs])
        n_true_positives = np.array([x['tp'] for x in outputs]).sum(axis=0)

        # calculate precision, recall and f1 over the validation dataset
        precision, recall, f1 = metric_scores(n_spindles_detected, n_spindles_gs, n_true_positives)
        # log the values for an overlap threshold of 20%
        self.log('metrics/val_precision', precision[4])
        self.log('metrics/val_recall', recall[4])
        self.log('metrics/val_f1', f1[4])
        # log the F1 score averaged over the overlap thresholds
        self.log('metrics/val_f1_mean', f1.mean())

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        data, mask = batch
        # training is performed on dense logits, therefore the (test) loss is calculated using these
        logits = self.dense_logits(data)

        loss = self.criterion(logits, mask)
        self.log('loss/test', loss)

        return self.get_batch_results(logits, mask)

    def test_epoch_end(self, outputs):
        n_spindles_detected, n_spindles_gs, n_true_positives = 0, 0, np.zeros_like(self.overlap_thresholds, dtype=int)

        # for each testset sum the detected, gold standard and true positive spindles
        for dataloader_outputs in outputs:
            n_spindles_detected += sum([x['detected'] for x in dataloader_outputs])  # type: ignore
            n_spindles_gs += sum([x['gold_standard'] for x in dataloader_outputs])  # type: ignore
            n_true_positives += np.array([x['tp'] for x in dataloader_outputs]).sum(axis=0)  # type: ignore

        # calculate precision, recall and f1 over the combined test datasets
        precision, recall, f1 = metric_scores(n_spindles_detected, n_spindles_gs, n_true_positives)
        # log the values for an overlap threshold of 20%
        self.log('metrics/test_precision', precision[4])
        self.log('metrics/test_recall', recall[4])
        self.log('metrics/test_f1', f1[4])
        # log the F1 score averaged over the overlap thresholds
        self.log('metrics/test_f1_mean', f1.mean())

    def configure_optimizers(self):
        # use the configured optimizer
        optimizer = self.optimizer(self.parameters(), **self.optimizer_params)
        if self.lr_scheduler is None:
            return optimizer
        else:
            # in case a learning rate scheduler is configured, it is returned together with the optimizer
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_params)
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': lr_scheduler, 'monitor': 'metrics/val_f1_mean'}}

    def configure_callbacks(self):
        callbacks = []
        if self.lr_scheduler is not None:
            # if a learning rate scheduler is configured, add a LearningRateMonitor, which logs the used learning rates
            callbacks.append(LearningRateMonitor())
        if self.early_stopping is not None:
            # add EarlyStopping if it is configured, using the given number of epochs as patience
            callbacks.append(EarlyStopping(monitor='metrics/val_f1_mean', patience=self.early_stopping, mode='max'))
        return callbacks

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super(SUMO, self).get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    @staticmethod
    def crop(x, output_samples):
        """
        Crops x to the given length (in last dimension), removing an equal amount of samples at the front and the end.

        Parameters
        ----------
        x : torch.Tensor
            Tensor which should be cropped in last dimension, shape [N,C,K].
        output_samples : int
            Number of samples L (in last dimension) to crop x to, L<=K

        Returns
        -------
        x : torch.Tensor
            The cropped tensor, shape [N,C,L]
        """

        len_x = x.shape[-1]
        assert output_samples <= len_x, f'x should contain at least output_samples, but contains {len_x}'

        if len_x == output_samples:
            # no cropping necessary
            return x
        else:
            diff = len_x - output_samples
            crop_dims = [diff // 2, diff // 2 + diff % 2]

            if crop_dims[1] == 0:
                return x[:, :, crop_dims[0]:]
            else:
                return x[:, :, crop_dims[0]:-crop_dims[1]]
