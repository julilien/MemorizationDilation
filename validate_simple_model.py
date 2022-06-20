import sys
import pickle
import logging

from models import MLP
from utils import *
from args import parse_train_args
from datasets import construct_preloaded_dataset, TRAIN_DATA_FN, TRAIN_TARGETS_FN

from validate_NC import FCFeatures, FCOutputs, eval_model


def main():
    args = parse_train_args()
    delete_model = args.delete_model
    if delete_model:
        logging.warning("Be warned: delete model is set to True")

    if args.val_split_prop == 0.0:
        args.val_split_prop = None

    args.load_path = args.save_path

    set_seed(seed=args.seed)

    if args.load_path is None:
        sys.exit('Need to input the path to a pre-trained model!')

    device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    args.device = device

    train_targets = torch.load(os.path.join(args.load_path, TRAIN_TARGETS_FN))
    train_data = torch.load(os.path.join(args.load_path, TRAIN_DATA_FN))
    selected_labels = torch.load(os.path.join(args.load_path, "selected_labels.pt"))

    trainloader, _, testloader, num_classes = construct_preloaded_dataset(train_data, train_targets, args.dataset,
                                                                          args.data_dir, args.batch_size,
                                                                          selected_labels=selected_labels,
                                                                          four_class_problem=args.fourclass_problem)

    if args.fourclass_problem:
        args.classes = num_classes

    if args.model == "MLP":
        if args.fourclass_twofeatures:
            num_penultimate_features = 2
        else:
            num_penultimate_features = args.classes

        model = MLP(hidden=args.width, depth=args.depth, fc_bias=args.bias,
                    num_classes=args.classes, penultimate_layer_features=num_penultimate_features,
                    final_activation=args.act_fn,
                    use_bn=args.use_bn).to(device)
    else:
        raise ValueError("Other models than MLP are not supported yet!")

    fc_features = FCFeatures()
    model.fc.register_forward_pre_hook(fc_features)

    fc_postsoftmax = FCOutputs()
    model.fc.register_forward_hook(fc_postsoftmax)

    info_dict = {
        'collapse_metric': [],
        'collapse_metric_post': [],
        'ETF_metric': [],
        'WH_relation_metric': [],
        'Wh_b_relation_metric': [],
        'W': [],
        'b': [],
        'H': [],
        'mu_G_train': [],
        'mu_G_post_train': [],
        'train_acc1': [],
        'train_acc{}'.format(num_classes): [],
        'test_acc1': [],
        'test_acc{}'.format(num_classes): [],
        'ece_metric_train': [],
        'ece_metric_test': [],

        # Additional metrics
        'Sigma_W': [],
        'Sigma_W_post': [],
        'Sigma_B': [],
        'Sigma_B_post': []
    }

    logfile = open('%s/test_log.txt' % (args.load_path), 'w')
    model_path = os.path.join(args.load_path, 'epoch_' + str(args.epochs).zfill(3) + '.pth')
    eval_model(args, model, model_path, info_dict, fc_features, fc_postsoftmax, trainloader, testloader, args.epochs,
               logfile, num_eval_classes=num_classes)

    with open(os.path.join(args.load_path, 'info.pkl'), 'wb') as f:
        pickle.dump(info_dict, f)

    # Delete model afterwards
    if delete_model and os.path.exists(model_path):
        os.remove(model_path)
        logging.info("Removed model.")


if __name__ == "__main__":
    main()
