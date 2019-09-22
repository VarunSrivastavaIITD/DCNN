import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch as th
from sklearn.metrics import f1_score, precision_score, recall_score


class YOLOloss(nn.Module):
    def __init__(
        self,
        bce_weight=1,
        mse_weight=0.1,
        gci_misclassfication_weight=0.1,
        zero_misclassification_weight=0.1,
        threshold=None,
    ):
        super().__init__()
        self.register_buffer("bce_weight", th.tensor(bce_weight, dtype=th.float32))
        self.register_buffer("mse_weight", th.tensor(mse_weight, dtype=th.float32))
        self.register_buffer(
            "gci_misclassification_weight",
            th.tensor(gci_misclassfication_weight, dtype=th.float32),
        )
        self.register_buffer(
            "zero_misclassification_weight",
            th.tensor(zero_misclassification_weight, dtype=th.float32),
        )
        self.register_buffer("threshold", th.tensor(threshold, dtype=th.float32))

    def compute_loss(
        self,
        input,
        target,
        bce_weight=None,
        mse_weight=None,
        gci_misclass_weight=None,
        zero_misclass_weight=None,
        threshold=None,
    ):

        bce_weight = self.bce_weight.item() if bce_weight is None else bce_weight
        mse_weight = self.mse_weight.item() if mse_weight is None else mse_weight
        gci_misclass_weight = (
            self.gci_misclassification_weight.item()
            if gci_misclass_weight is None
            else gci_misclass_weight
        )
        zero_misclass_weight = (
            self.zero_misclassification_weight.item()
            if zero_misclass_weight is None
            else zero_misclass_weight
        )
        threshold = self.threshold.item() if threshold is None else threshold

        peak_distance_target = target[:, 0]
        peak_indicator_target = target[:, 1]
        loss_bce = F.binary_cross_entropy_with_logits(
            input[:, 1], peak_indicator_target
        )
        loss_mse = F.mse_loss(
            input[:, 0] * peak_indicator_target,
            peak_distance_target * peak_indicator_target,
        )

        out = F.sigmoid(input[:, 1])

        gci_misclass = (peak_indicator_target) * (1 - out) ** 2
        gci_misclass = gci_misclass.mean()

        zero_misclass = (1 - peak_indicator_target) * out ** 2
        zero_misclass = zero_misclass.mean()

        classifier_target = target[:, 1].detach().cpu().numpy().ravel()
        prediction = np.zeros_like(classifier_target)
        tind = input[:, 1].detach().cpu().numpy().ravel() > threshold
        prediction[tind] = 1
        f1 = f1_score(classifier_target, prediction)
        pr = precision_score(classifier_target, prediction)
        re = recall_score(classifier_target, prediction)

        net_loss = (
            bce_weight * loss_bce
            + mse_weight * loss_mse
            + gci_misclass_weight * gci_misclass
            + zero_misclass_weight * zero_misclass
        )

        return (
            net_loss,
            {
                "BCE": loss_bce.item(),
                "MSE": loss_mse.item(),
                "GCI_Misclassification": gci_misclass.item(),
                "Zero_Misclassification": zero_misclass.item(),
                "f1_score": f1,
                "precision_score": pr,
                "recall_score": re,
            },
        )

    def forward(self, input, target):
        net_loss, _ = self.compute_loss(
            input,
            target,
            self.bce_weight,
            self.mse_weight,
            self.misclassification_weight,
            self.zero_misclassification_weight,
            self.threshold,
        )

        return net_loss
