import os
import pickle

from utils import LABEL_SMOOTHING_TAG, LABEL_RELAXATION_TAG


def generate_uid(args):
    """
    Generates a unique identifier for a model experiment.
    """
    if not args.use_largescale_uid:
        wd_str = "wd{}-{}".format(str(args.weight_decay), str(args.feature_decay_rate))
        if args.sep_decay:
            wd_str += "sep"

        if args.use_bn:
            bn_str = "with_bn"
        else:
            bn_str = "wo_bn"

        loss_str = args.loss
        if loss_str == LABEL_SMOOTHING_TAG:
            loss_str += str(args.ls_alpha)
        elif loss_str == LABEL_RELAXATION_TAG:
            loss_str += str(args.lr_alpha)

        uid = '_'.join(
            [args.model, args.dataset, "c" + str(args.classes), "ln" + str(args.label_noise), loss_str, args.act_fn,
             "d" + str(args.depth), "w" + str(args.width), wd_str, bn_str, "s" + str(args.seed)])
    else:
        loss_str = args.loss
        if not args.use_ho_uid:
            if loss_str == LABEL_SMOOTHING_TAG:
                loss_str += str(args.ls_alpha)
            elif loss_str == LABEL_RELAXATION_TAG:
                loss_str += str(args.lr_alpha)

        uid = '_'.join([args.dataset, "s" + str(args.seed), loss_str, args.model, "wd{}".format(args.weight_decay)])

        if not args.use_ho_uid:
            uid += '_lr' + str(args.lr)

    if args.use_ho_uid:
        uid += "_" + args.ho

        if args.label_noise > 0.0:
            uid += "_ln" + str(args.label_noise)

    if "continue_ho" in args.__dict__.keys() and args.continue_ho:
        uid += "_" + str(args.num_ho_runs) + "c"

    if args.fourclass_problem:
        uid += "_4c" + str(args.fc_noise_degree)
        if args.fourclass_twofeatures:
            uid += "_4c2d"

    return uid


def retrieve_model_dict(dict_id, base_path, model_name, alt_base_path=None):
    def retrieve_metrics(model_dir, model_name):
        with open(os.path.join(os.path.join(model_dir, model_name), dict_id + '.pkl'), 'rb') as f:
            return pickle.load(f)

    if alt_base_path is None:
        return retrieve_metrics(base_path, model_name)
    else:
        try:
            return retrieve_metrics(base_path, model_name)
        except FileNotFoundError:
            return retrieve_metrics(alt_base_path, model_name)
