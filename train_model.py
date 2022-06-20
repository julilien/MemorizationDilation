import sys
import logging

import models
from utils import *
from args import parse_train_args
from datasets import make_reproducible_dataset


def loss_compute(args, model, criterion, outputs, targets):
    if args.loss in [CROSS_ENTROPY_TAG, LABEL_SMOOTHING_TAG, LABEL_RELAXATION_TAG]:
        loss = criterion(outputs[0], targets)
    elif args.loss == MSE_TAG:
        loss = criterion(outputs[0], nn.functional.one_hot(targets).type(torch.FloatTensor).to(args.device))

    # Now decide whether to add weight decay on last weights and last features
    if args.sep_decay:
        # Find features and weights
        features = outputs[1]
        w = model.fc.weight
        b = model.fc.bias
        lamb = args.weight_decay / 2
        lamb_feature = args.feature_decay_rate / 2
        loss += lamb * (torch.sum(w ** 2) + torch.sum(b ** 2)) + lamb_feature * torch.sum(features ** 2)

    return loss


def trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile, num_classes):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    print_and_save('\nTraining Epoch: [%d | %d] LR: %f' % (epoch_id + 1, args.epochs, scheduler.get_last_lr()[-1]),
                   logfile)
    for batch_idx, (inputs, targets) in enumerate(trainloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        model.train()
        outputs = model(inputs)

        if args.sep_decay:
            loss = loss_compute(args, model, criterion, outputs, targets)
        else:
            if args.loss in [CROSS_ENTROPY_TAG, LABEL_SMOOTHING_TAG, LABEL_RELAXATION_TAG]:
                loss = criterion(outputs[0], targets)
            elif args.loss == MSE_TAG:
                loss = criterion(outputs[0],
                                 nn.functional.one_hot(targets, num_classes).type(torch.FloatTensor).to(args.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        model.eval()
        outputs = model(inputs)
        prec1, prec5 = compute_accuracy(outputs[0].detach().data, targets.detach().data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if batch_idx % 10 == 0:
            print_and_save('[epoch: %d] (%d/%d) | Loss: %.4f | top1: %.4f | top5: %.4f ' %
                           (epoch_id + 1, batch_idx + 1, len(trainloader), losses.avg, top1.avg, top5.avg), logfile)

    scheduler.step()

    return losses.avg


def train(args, model, trainloader, num_classes):
    criterion = make_criterion(args, num_classes)
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    logfile = open('%s/train_log.txt' % (args.save_path), 'w')

    if os.path.exists(
            os.path.join(args.save_path, "epoch_" + str(args.epochs).zfill(3) + ".pth")) and not args.force_retrain:
        logging.info("Model already exists, loading this model...")
        model.load_state_dict(torch.load(os.path.join(args.save_path, "epoch_" + str(args.epochs).zfill(3) + ".pth")))
    else:
        print_and_save('# of model parameters: ' + str(count_network_parameters(model)), logfile)
        print_and_save('--------------------- Training -------------------------------', logfile)
        for epoch_id in range(args.epochs):
            trainer(args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile, num_classes)

            torch.save(model.state_dict(), args.save_path + "/epoch_" + str(epoch_id + 1).zfill(3) + ".pth")

    logfile.close()


def main():
    args = parse_train_args()

    set_seed(seed=args.seed)

    if args.optimizer == 'LBFGS':
        sys.exit('Support for training with 1st order methods!')

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, _, _, num_classes = make_reproducible_dataset(args, args.save_path, label_noise=args.label_noise)

    if args.model == "MLP":
        model = models.__dict__[args.model](hidden=args.width, depth=args.depth, fc_bias=args.bias,
                                            num_classes=num_classes).to(device)
    else:
        model = models.__dict__[args.model](num_classes=num_classes, fc_bias=args.bias, ETF_fc=args.ETF_fc,
                                            fixdim=args.fixdim, SOTA=args.SOTA).to(device)

    train(args, model, trainloader, num_classes)


if __name__ == "__main__":
    main()
