import argparse
import logging
import math

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.suggest.basic_variant import BasicVariantGenerator

import models
from train_model import trainer
from train_simple_model import evaluate_dataloader
from utils import *
from args import parse_ho_args, dump_args_dict
from datasets import make_reproducible_dataset


def train_ho_run(config, args, trainloader, test_val_loader, num_classes, val_prefix, save_model=False, test_run=False):
    # Pass config parameters to args
    fused_run_args = argparse.Namespace()
    fused_run_args.__dict__ = args.__dict__.copy()
    if not test_run:
        fused_run_args.epochs = int(args.epochs * (1. - args.val_split_prop))

    for key in config.keys():
        fused_run_args.__dict__[key] = config[key]

    if fused_run_args.model == "MLP":
        input_dim = 3072
        model = models.__dict__[fused_run_args.model](hidden=fused_run_args.width, depth=fused_run_args.depth,
                                                      fc_bias=fused_run_args.bias, num_classes=num_classes,
                                                      input_dim=input_dim).to(fused_run_args.device)
    else:
        model = models.__dict__[fused_run_args.model](num_classes=num_classes, fc_bias=fused_run_args.bias,
                                                      ETF_fc=fused_run_args.ETF_fc,
                                                      fixdim=fused_run_args.fixdim, SOTA=fused_run_args.SOTA).to(
            fused_run_args.device)

    criterion = make_criterion(fused_run_args, num_classes)
    optimizer = make_optimizer(fused_run_args, model)
    scheduler = make_scheduler(fused_run_args, optimizer)

    logfile = open('%s/train_log.txt' % (fused_run_args.save_path), 'w')

    if os.path.exists(
            os.path.join(args.save_path, "epoch_" + str(args.epochs).zfill(3) + ".pth")) and not args.force_retrain:
        logging.info("Model already exists, loading this model...")
        model.load_state_dict(torch.load(os.path.join(args.save_path, "epoch_" + str(args.epochs).zfill(3) + ".pth")))
    else:
        print_and_save('# of model parameters: ' + str(count_network_parameters(model)), logfile)
        print_and_save('--------------------- Training -------------------------------', logfile)

        for epoch_id in range(fused_run_args.epochs):
            loss = trainer(fused_run_args, model, trainloader, epoch_id, criterion, optimizer, scheduler, logfile,
                           num_classes=num_classes)
            if math.isnan(loss):
                print_and_save('NaN loss encountered, stopping training...', logfile)
                break

            if save_model:
                torch.save(model.state_dict(),
                           os.path.join(fused_run_args.save_path, "epoch_" + str(epoch_id + 1).zfill(3) + ".pth"))

    test_val_acc = evaluate_dataloader(test_val_loader, model, fused_run_args, criterion, logfile, is_binary=False,
                                       prefix=val_prefix, num_classes=num_classes, return_top1=True)

    logfile.close()

    acc_str = "{}_acc".format(val_prefix)
    if not test_run:
        reported_metrics = {acc_str: test_val_acc}
        tune.report(**reported_metrics)
    logging.info("{}: {}".format(acc_str, test_val_acc))


def main():
    args, config = parse_ho_args(return_config=True)
    assert args.val_split_prop is not None and args.val_split_prop > 0 and args.val_split_prop < 1
    assert args.use_ho_uid is True, "This script is only for HO experiments - mark the UID by the respective flag."
    assert args.ho is not None, "Hyperparameter optimization method must be specified."

    # Initialize ray
    ray.init(_memory=int(config["RESOURCES"]["HEAP_MEMORY_IN_GB"]) * 1024 * 1024 * 1024,
             object_store_memory=1 * 1024 * 1024 * 1024,
             _redis_max_memory=1 * 1024 * 1024 * 1024, _temp_dir=config["PATHS"]["TMP_PATH"])
    logging.debug("Available resources to ray: {}".format(ray.available_resources()))

    set_seed(seed=args.seed)

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    trainloader, valloader, testloader, num_classes = make_reproducible_dataset(args, None,
                                                                                val_split_prop=args.val_split_prop,
                                                                                label_noise=0.0)

    # Define search space
    ho_config = {"lr": tune.loguniform(1e-5, 0.5), "gamma": tune.choice([0.01, 0.1, 0.5])}
    param_columns = ["lr", "gamma"]
    if args.loss == LABEL_SMOOTHING_TAG:
        ho_config["ls_alpha"] = tune.uniform(0.01, 0.25)
        param_columns.append("ls_alpha")
    elif args.loss == LABEL_RELAXATION_TAG:
        ho_config["lr_alpha"] = tune.uniform(0.01, 0.25)
        param_columns.append("lr_alpha")

    # Specify hyperparameter optimization parameters
    reporter = CLIReporter(parameter_columns=param_columns, metric_columns=["val_acc", "training_iteration"],
                           max_report_frequency=30)
    scheduler = AsyncHyperBandScheduler(metric="val_acc", mode="max")

    if args.ho == "bayes_opt":
        algo = SkOptSearch(metric="val_acc", mode="max")
        algo = ConcurrencyLimiter(algo, max_concurrent=int(config["RESOURCES"]["MAXIMUM_CONCURRENT_JOBS"]))
    else:
        # Random search
        algo = BasicVariantGenerator()

    # Conduct hyperparameter optimization
    result = tune.run(
        tune.with_parameters(train_ho_run, args=args, trainloader=trainloader, test_val_loader=valloader,
                             num_classes=num_classes, val_prefix="val"),
        config=ho_config, progress_reporter=reporter, checkpoint_at_end=True,
        num_samples=args.num_ho_runs, resources_per_trial={"cpu": int(config["RESOURCES"]["CPUS_PER_NODE"]),
                                                           "gpu": float(config["RESOURCES"]["GPUS_PER_NODE"])},
        local_dir=os.path.join(args.save_path, "tune_results"), scheduler=scheduler, search_alg=algo,
        sync_config=tune.SyncConfig(syncer=None), name=args.uid, resume="AUTO")
    best_trial = result.get_best_trial("val_acc", "max", "last")

    # Train final model on best trial
    trainloader, _, testloader, num_classes = make_reproducible_dataset(args, args.save_path, val_split_prop=None,
                                                                        label_noise=args.label_noise)
    best_args = argparse.Namespace()
    best_args.__dict__ = args.__dict__.copy()
    for key in best_trial.config.keys():
        best_args.__dict__[key] = best_trial.config[key]
    train_ho_run(best_trial.config, best_args, trainloader, testloader, num_classes=num_classes,
                 val_prefix="test", save_model=True, test_run=True)

    dump_args_dict(best_args)


if __name__ == "__main__":
    main()
