import numpy as np


def is_fourclass_exp_run(args):
    return "fourclass_problem" in args.__dict__ and args.fourclass_problem


def determine_memorization(args, train_features, train_labels, corr_mask, metrics_dict, uncorr_means):
    memorization_dists = np.zeros(2, dtype=float)
    for i in range(memorization_dists.shape[0]):
        num_elements = 0

        # uncorr_means is training mean of images that look like 1 and labeled as 1
        # We look at the original class features
        if args.label_noise > 0.0:
            # Take the elements that look like the class but are labeled the class + 1 % 2
            tmp_elements = train_features[corr_mask][train_labels[corr_mask][:, 1] == i]
            for j in range(tmp_elements.shape[0]):
                memorization_dists[i] += np.linalg.norm(tmp_elements[j] - uncorr_means[i])
            num_elements += tmp_elements.shape[0]

            # Take the elements that look like the class but are labeled either 2 or 3 (3 for class 0, 2 for class 1)
            tmp_elements = train_features[corr_mask][train_labels[corr_mask][:, 1] == (i + 2)]
            for j in range(tmp_elements.shape[0]):
                memorization_dists[i] += np.linalg.norm(tmp_elements[j] - uncorr_means[i])
            num_elements += tmp_elements.shape[0]

        # Take the uncorrupted labels that have the same class as the class we are looking at
        tmp_elements = train_features[corr_mask][train_labels[corr_mask][:, 1] == (i + 2)]
        for j in range(tmp_elements.shape[0]):
            memorization_dists[i] += np.linalg.norm(tmp_elements[j] - uncorr_means[i])

    metrics_dict["corr_test_collapse_dist_mean"] = float(np.mean(memorization_dists))
    metrics_dict["pointwise_distances_test"] = memorization_dists
