import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedDiceLoss(nn.Module):
    """
    Compute the generalised Dice loss defined in:
        Sudre, C. et al. (2017) Generalised Dice overlap as a deep learning loss function for highly unbalanced
        segmentations. DLMIA 2017. https://arxiv.org/pdf/1707.03237.pdf
    Adapted from:
        https://github.com/Project-MONAI/MONAI/blob/0.5.2/monai/losses/dice.py#L216

    Parameters
    ----------
    reduction : {'mean', 'sum', 'none'}, optional
        Specifies the reduction to apply to the output. The sum of the output will be divided by the number of
        elements in the output ('mean'), the output will be summed ('sum') or no reduction will be applied ('none').
        Default is 'mean'.
    smooth : float, optional
        A small constant added to the numerator and denominator to avoid zero and nan.
    use_weight: bool, optional
        When true, use class weights as originally proposed by Sudre et al.
    softmax : bool, optional
        When True, apply a softmax function to the prediction.
    """

    def __init__(self, reduction: str = 'mean', smooth: float = 1e-5, use_weight: bool = True, softmax: bool = True):
        super(GeneralizedDiceLoss, self).__init__()

        self.reduction = reduction
        self.smooth = smooth
        self.use_weight = use_weight
        self.softmax = softmax

    @staticmethod
    def get_onehot_encoding(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        shp_x = y_pred.shape
        shp_y = y_true.shape

        with torch.no_grad():
            y_true = y_true.long()

            if shp_x == shp_y:
                return y_true  # y_true is already in one hot encoding
            else:
                y_onehot = F.one_hot(y_true, num_classes=shp_x[1])  # one hot encoding in format [N,K,C]
                y_onehot = y_onehot.permute(0, 2, 1)  # transform to format [N,C,K]

                return y_onehot

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward pass of the generalized dice loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            The predicted logits (or probabilities if `self.softmax` is False) in shape [N,C,K] where N is the batch
            size, C the number of classes and K the number of elements/steps in each observation.
        y_true : torch.Tensor
            The target labels in shape [N,K] or as one hot encoded vector in format [N,C,K].

        Returns
        -------
        loss : torch.Tensor
            The calculated loss as scalar or in shape [N] if `self.reduction` is 'none'.

        Raises
        ------
        ValueError
            If `self.reduction` is not one of {'mean', 'sum', 'none'}.
        """

        if self.softmax:
            y_pred = F.softmax(y_pred, dim=1)

        y_onehot = self.get_onehot_encoding(y_pred, y_true)

        # calculate intersection and union and sum them over the K steps in each observation; shape [N,C]
        intersection = (y_pred * y_onehot).sum(dim=2)
        union = (y_pred + y_onehot).sum(dim=2)

        if self.use_weight:
            # class weights using the inverse of each label volume; shape [N,C]
            w = 1 / y_onehot.sum(dim=2)**2
            # if one class doesn't contain any labels, its weight is set to 1.0 (as if it would contain exactly one
            # label)
            w[torch.isinf(w)] = 1.0

            # apply the weights on intersection and union and sum over the classes; shape [N]
            intersection = (w * intersection).sum(dim=1)
            union = (w * union).sum(dim=1)
        else:
            # sum intersection and union over the classes; shape [N]
            intersection = intersection.sum(dim=1)
            union = union.sum(dim=1)

        # calculate dice coefficient and generalized dice loss using a small number to prevent zero/nan; shape [N]
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        gdl = 1.0 - dice

        if self.reduction == 'mean':
            gdl = gdl.mean()  # average over the batch and channel; scalar
        elif self.reduction == 'sum':
            gdl = gdl.sum()  # sum over the batch and channel; scalar
        elif self.reduction == 'none':
            pass  # unmodified losses per batch; shape [N]
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return gdl
