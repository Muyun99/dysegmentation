import torch
import torch.nn.functional as F
from torch.autograd import Function
from .boundary_score import BoundaryScore_fast
from .jaccard_index import jaccard_index


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def eval_net_unet_dice(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.to(device=device)
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_loss = 0

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_classes > 1:
            total_loss += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            total_loss += dice_coeff(pred, true_masks).item()

    net.train()
    return total_loss / n_val


def eval_net_unet_bfscore(net, loader, device):
    bfscore_func = BoundaryScore_fast()
    net.to(device=device)
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_loss = 0

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        pred = torch.sigmoid(mask_pred)
        # pred[pred > 0.5] = 1
        pred = (pred > 0.5).float()
        bfscore = bfscore_func(pred=pred.cpu(), gt=true_masks.cpu())
        total_loss += bfscore.mean().numpy()

    net.train()
    return total_loss / n_val


def eval_net_unet_metric(net, loader, device):
    bfscore_func = BoundaryScore_fast()
    miou_func = jaccard_index()

    net.to(device=device)
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_dice = 0
    total_miou = 0
    total_bfscore = 0

    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        pred = torch.sigmoid(mask_pred)
        pred = (pred > 0.5).float()

        total_dice += dice_coeff(pred, true_masks).item()
        total_miou += miou_func(mask_gt=true_masks.cpu(), mask_pred=pred.cpu()).item()
        total_bfscore += bfscore_func(pred=pred.cpu(), gt=true_masks.cpu()).mean().numpy()

    net.train()
    return total_dice / n_val, total_miou / n_val, total_bfscore / n_val


def eval_net_unet_miou(net, loader, device):
    miou_func = jaccard_index()
    net.to(device=device)
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_loss = 0

    # with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        pred = torch.sigmoid(mask_pred)
        pred = (pred > 0.5).float()
        total_loss += miou_func(mask_gt=true_masks.cpu(), mask_pred=pred.cpu()).item()

    net.train()
    return total_loss / n_val


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def eval_net_cls(net, loader, device):
    net.to(device=device)
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_loss = 0
    total_acc = 0

    for batch in loader:
        batch_len = len(batch['image'])
        imgs, true_masks, label = batch['image'], batch['mask'], batch['label']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)
        label = label.to(device)

        with torch.no_grad():
            mask_pred, weight_score = net(imgs, true_masks)
            total_loss += F.cross_entropy(weight_score, label).item()
            _acc = accuracy(weight_score, label)
            total_acc += _acc[0].item() * batch_len
    net.train()
    return total_loss / n_val, total_acc / n_val
