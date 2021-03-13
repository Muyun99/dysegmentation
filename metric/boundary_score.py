# BFScore的fast版本
import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryScore_fast(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def one_hot(self, label, n_classes, requires_grad=True):
        """Return One Hot Label"""
        one_hot_label = torch.eye(
            n_classes, device=self.device, requires_grad=requires_grad)[label]
        one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

        return one_hot_label

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # P = torch.sum(pred_b * gt_b_ext, dim=2)  / (torch.sum(pred_b, dim=2))
        # R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2))

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        return BF1
