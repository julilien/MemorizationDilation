import argparse
import logging

from models import MLP
from train_simple_model import train
from utils import *
from args import dump_args_dict, parse_ho_args
from datasets import make_selected_dataset, construct_preloaded_dataset, TRAIN_DATA_FN, TRAIN_TARGETS_FN, \
    TRAIN_ORIG_TARGETS_FN, SELECTED_LABELS_FN
from viz.eval_simple_model_viz import evaluate_model_visually


def construct_simple_model(args, num_classes, device):
    if args.model == "MLP":
        return MLP(hidden=args.width, depth=args.depth, fc_bias=args.bias, num_classes=num_classes,
                   penultimate_layer_features=num_classes, final_activation=args.act_fn,
                   use_bn=args.use_bn, use_layer_norm=args.use_layer_norm).to(device)
    else:
        raise ValueError("Non supported model {}.".format(args.model))


def main():
    args = parse_ho_args()
    assert args.val_split_prop is not None and args.val_split_prop > 0 and args.val_split_prop < 1
    set_seed(seed=args.seed)

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    selected_labels = [i for i in range(args.classes)]

    # Define search space
    search_space = {"lr": ([-11.51, -0.693], lambda x: np.exp(np.random.uniform(x[0], x[1]))),
                    "gamma": ([0.01, 0.1, 0.5], lambda x: np.random.choice(x, 1))
                    }
    if args.loss == LABEL_SMOOTHING_TAG:
        search_space["ls_alpha"] = ([0.01, 0.25], lambda x: np.random.uniform(x[0], x[1]))
    elif args.loss == LABEL_RELAXATION_TAG:
        search_space["lr_alpha"] = ([0.01, 0.25], lambda x: np.random.uniform(x[0], x[1]))

    # Perform a random search for hyperparameter determination
    num_ho_runs = args.num_ho_runs
    val_results = np.zeros(num_ho_runs)
    max_params = None
    max_val_acc = 0.
    search_history = []
    logfile = open('%s/ho_log.txt' % (args.save_path), 'w')

    def fuse_args(base_args, new_params):
        fused_run_args = argparse.Namespace()
        fused_run_args.__dict__ = base_args.__dict__.copy()
        fused_run_args.epochs = int(base_args.epochs * (1. - base_args.val_split_prop))

        for key in new_params.keys():
            fused_run_args.__dict__[key] = float(new_params[key])
        return fused_run_args

    if args.continue_ho:
        # Read and parse previous history
        ho_base_paths = [args.continue_ho_base_path]
        selected_path = None
        matching_base_path = None
        for ho_base_path in ho_base_paths:
            clear_uid = args.uid[:-len(args.uid.split('_')[-1]) - 1]
            tmp_path = os.path.expanduser("~") + os.path.join(os.path.join(ho_base_path, clear_uid), "ho_log.txt")
            if os.path.exists(tmp_path):
                selected_path = tmp_path
                matching_base_path = os.path.expanduser("~") + os.path.join(ho_base_path, clear_uid)
                break

        if selected_path is None:
            logging.error("Could not find any previous history.")
            exit(1)
        assert matching_base_path is not None

        with open(selected_path, 'r') as f:
            raw_history_str = f.readlines()[0][12:-1]

        # Parse list of tuples from string
        search_history = eval(raw_history_str)

        print("Found previous history: {}".format(search_history), file=logfile)
        logfile.flush()

        # Determine current maximum
        for i in range(len(search_history)):
            elem = search_history[i]
            if elem[1] > max_val_acc:
                max_val_acc = elem[1]
                val_results[i] = elem[1]
                max_params = fuse_args(args, elem[0])

        # Read in the data from previous runs
        train_data = torch.load(os.path.join(matching_base_path, TRAIN_DATA_FN))
        train_targets = torch.load(os.path.join(matching_base_path, TRAIN_TARGETS_FN))
        train_original_targets = torch.load(os.path.join(matching_base_path, TRAIN_ORIG_TARGETS_FN))
        selected_labels = torch.load(os.path.join(matching_base_path, SELECTED_LABELS_FN))

        # Save to new directory
        torch.save(train_data, os.path.join(args.save_path, TRAIN_DATA_FN))
        torch.save(train_targets, os.path.join(args.save_path, TRAIN_TARGETS_FN))
        torch.save(train_original_targets, os.path.join(args.save_path, TRAIN_ORIG_TARGETS_FN))
        torch.save(selected_labels, os.path.join(args.save_path, SELECTED_LABELS_FN))

        trainloader, valloader, testloader, num_classes = construct_preloaded_dataset(train_data, train_targets,
                                                                                      args.dataset,
                                                                                      args.data_dir, args.batch_size,
                                                                                      selected_labels=selected_labels,
                                                                                      args=args, shuffle_val_data=True)
    else:
        # Construct dataset
        trainloader, valloader, _, num_classes = make_selected_dataset(args, args.dataset, args.data_dir,
                                                                       args.batch_size, args.sample_size,
                                                                       val_split_prop=args.val_split_prop,
                                                                       label_noise=args.label_noise,
                                                                       selected_labels=selected_labels,
                                                                       shuffle_val_data=True)

    logging.debug("Training data for HO #: {}".format(len(trainloader) * args.batch_size))
    if valloader is not None:
        logging.debug("Validation data for HO #: {}".format(len(valloader) * args.batch_size))

    for i in range(len(search_history), num_ho_runs):
        # Sample next hyperparameter combination
        logging.info("Performing hyperparameter optimization run #{}".format(i))

        # Specify arguments
        ho_run_args = argparse.Namespace()
        ho_run_args.__dict__ = args.__dict__.copy()
        ho_run_args.epochs = int(args.epochs * (1. - args.val_split_prop))

        curr_space = {}
        for key in search_space.keys():
            ho_run_args.__dict__[key] = float(search_space[key][1](search_space[key][0]))
            curr_space[key] = ho_run_args.__dict__[key]
        logging.debug("Testing parameters: {}".format(curr_space))

        # Construct model
        model = construct_simple_model(ho_run_args, num_classes, device)
        val_acc = train(ho_run_args, model, trainloader, None, valloader, num_classes, val_prefix="val",
                        force_retrain=True)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_params = ho_run_args

        logging.info("HO run #{} finished with val_acc={}".format(i, val_acc))
        search_history.append((curr_space, val_acc))
        val_results[i] = val_acc

    # Set epochs back to the original value
    print("HO history: {}".format(search_history), file=logfile)

    logging.info("HO history: {}".format(search_history))
    max_params.epochs = args.epochs
    best_args = max_params
    logging.info("Best args: {}".format(best_args))
    print("====", file=logfile)
    print("Best args: {}".format(best_args), file=logfile)
    logfile.close()

    model = construct_simple_model(best_args, num_classes, device)

    train_targets = torch.load(os.path.join(best_args.save_path, TRAIN_TARGETS_FN))
    train_data = torch.load(os.path.join(best_args.save_path, TRAIN_DATA_FN))
    trainloader, _, testloader, num_classes = construct_preloaded_dataset(train_data, train_targets, best_args.dataset,
                                                                          best_args.data_dir, best_args.batch_size,
                                                                          selected_labels=selected_labels)
    logging.debug("Training data for final model #: {}".format(len(trainloader) * best_args.batch_size))
    logging.debug("Test data for final model #: {}".format(len(testloader) * best_args.batch_size))

    train(best_args, model, trainloader, None, testloader, num_classes, val_prefix="test")
    evaluate_model_visually(best_args, model, base_path=best_args.save_path)

    dump_args_dict(best_args)


if __name__ == "__main__":
    main()
