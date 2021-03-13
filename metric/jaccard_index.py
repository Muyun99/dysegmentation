import numpy as np
import torch.nn as nn

class jaccard_index(nn.Module):
    def __init__(self):
        super(jaccard_index, self).__init__()

    def intersect_and_union(self, pred_label, label, num_classes):
        """Calculate intersection and Union.
        Args:
            pred_label (ndarray): Prediction segmentation map
            label (ndarray): Ground truth segmentation map
            num_classes (int): Number of categories
            ignore_index (int): Index that will be ignored in evaluation.
         Returns:
             ndarray: The intersection of prediction and ground truth histogram
                 on all classes
             ndarray: The union of prediction and ground truth histogram on all
                 classes
             ndarray: The prediction histogram on all classes.
             ndarray: The ground truth histogram on all classes.
        """

        intersect = pred_label[pred_label == label]
        area_intersect, _ = np.histogram(
            intersect, bins=np.arange(num_classes + 1))
        area_pred_label, _ = np.histogram(
            pred_label, bins=np.arange(num_classes + 1))
        area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
        area_union = area_pred_label + area_label - area_intersect

        return area_intersect, area_union, area_pred_label, area_label

    def mean_iou(self, gt, pred, num_classes):
        """Calculate Intersection and Union (IoU)
        Args:
            results (list[ndarray]): List of prediction segmentation maps
            gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
            num_classes (int): Number of categories
            ignore_index (int): Index that will be ignored in evaluation.
            nan_to_num (int, optional): If specified, NaN values will be replaced
                by the numbers defined by the user. Default: None.
         Returns:
             float: Overall accuracy on all images.
             ndarray: Per category accuracy, shape (num_classes, )
             ndarray: Per category IoU, shape (num_classes, )
        """

        total_area_intersect = np.zeros((num_classes,), dtype=np.float)
        total_area_union = np.zeros((num_classes,), dtype=np.float)
        total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
        total_area_label = np.zeros((num_classes,), dtype=np.float)
        area_intersect, area_union, area_pred_label, area_label = \
            self.intersect_and_union(pred, gt, num_classes)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
        all_acc = total_area_intersect.sum() / total_area_label.sum()
        acc = total_area_intersect / total_area_label
        iou = total_area_intersect / total_area_union
        return all_acc, acc, iou

    def forward(self, mask_gt, mask_pred):
        all_acc, acc, miou = self.mean_iou(gt=mask_gt, pred=mask_pred, num_classes=1)
        return miou[0]
