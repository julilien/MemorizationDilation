import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from label_smoothing import LabelSmoothingLossCanonical, LabelRelaxationLoss

CROSS_ENTROPY_TAG = "CrossEntropy"
LABEL_SMOOTHING_TAG = "LS"
LABEL_RELAXATION_TAG = "LR"
MSE_TAG = "MSE"


def set_seed(seed=666):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.sep_decay:
        wd_term = 0
    else:
        wd_term = args.weight_decay

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9,
                  'lr': args.lr,
                  'weight_decay': wd_term  # args.weight_decay
                  }
    elif args.optimizer == 'Adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08,
            'lr': args.lr,
            'weight_decay': wd_term  # args.weight_decay
        }
    elif args.optimizer == 'LBFGS':
        optimizer_function = optim.LBFGS
        kwargs = {'lr': args.lr,
                  'history_size': args.history_size,
                  'line_search_fn': 'strong_wolfe'
                  }

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.patience,
            gamma=args.gamma
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    return scheduler


def make_criterion(args, num_classes, is_binary=False):
    if args.loss == CROSS_ENTROPY_TAG:
        if is_binary:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.loss == MSE_TAG:
        criterion = nn.MSELoss()
    elif args.loss == LABEL_SMOOTHING_TAG:
        criterion = LabelSmoothingLossCanonical(args.ls_alpha)
    elif args.loss == LABEL_RELAXATION_TAG:
        criterion = LabelRelaxationLoss(args.lr_alpha, num_classes=num_classes)

    return criterion


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in parameters])


def print_and_save(text_str, file_stream):
    print(text_str)
    print(text_str, file=file_stream)


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_binary_accuracy(output, target):
    """Computes the binary accuracy."""
    y_pred = torch.squeeze(output > 0.5)
    y_true = torch.squeeze(target > 0.5)
    return (y_true == y_pred).sum().item() * 100. / y_true.size(0)
