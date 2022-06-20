from dataset_utils import CIFAR10_STR, FASHION_MNIST_STR, MNIST_STR, SVHN_STR, CIFAR100_STR

MNIST_TRAIN_SAMPLES = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
MNIST_TEST_SAMPLES = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)
CIFAR10_TRAIN_SAMPLES = 10 * (5000,)
CIFAR10_TEST_SAMPLES = 10 * (1000,)
CIFAR100_TRAIN_SAMPLES = 100 * (500,)
CIFAR100_TEST_SAMPLES = 100 * (100,)

SVHN_TRAIN_SAMPLES = (4948, 13861, 10585, 8497, 7458, 6882, 5727, 5595, 5045, 4659)
SVHN_TEST_SAMPLES = (1744, 5099, 4149, 2882, 2523, 2384, 1977, 2019, 1660, 1595)
FASHION_MNIST_TRAIN_SAMPLES = 10 * (6000,)
FASHION_MNIST_TEST_SAMPLES = 10 * (1000,)


def create_4c_tuple(args):
    if args.dataset == MNIST_STR:
        MNIST_TRAIN_SAMPLES_4C = list(MNIST_TRAIN_SAMPLES[:2] + MNIST_TRAIN_SAMPLES[:2])
        MNIST_TRAIN_SAMPLES_4C[0] = int(MNIST_TRAIN_SAMPLES_4C[0] * (1. - args.fc_noise_degree))
        MNIST_TRAIN_SAMPLES_4C[1] = int(MNIST_TRAIN_SAMPLES_4C[1] * (1. - args.fc_noise_degree))
        MNIST_TRAIN_SAMPLES_4C[2] = int(MNIST_TRAIN_SAMPLES_4C[2] * args.fc_noise_degree)
        MNIST_TRAIN_SAMPLES_4C[3] = int(MNIST_TRAIN_SAMPLES_4C[3] * args.fc_noise_degree)

        return tuple(MNIST_TRAIN_SAMPLES_4C), MNIST_TEST_SAMPLES[:2]
    elif args.dataset == CIFAR10_STR:
        CIFAR10_TRAIN_SAMPLES_4C = list(CIFAR10_TRAIN_SAMPLES[:2] + CIFAR10_TRAIN_SAMPLES[:2])
        CIFAR10_TRAIN_SAMPLES_4C[0] = int(CIFAR10_TRAIN_SAMPLES_4C[0] * (1. - args.fc_noise_degree))
        CIFAR10_TRAIN_SAMPLES_4C[1] = int(CIFAR10_TRAIN_SAMPLES_4C[1] * (1. - args.fc_noise_degree))
        CIFAR10_TRAIN_SAMPLES_4C[2] = int(CIFAR10_TRAIN_SAMPLES_4C[2] * args.fc_noise_degree)
        CIFAR10_TRAIN_SAMPLES_4C[3] = int(CIFAR10_TRAIN_SAMPLES_4C[3] * args.fc_noise_degree)

        return tuple(CIFAR10_TRAIN_SAMPLES_4C), CIFAR10_TEST_SAMPLES[:2]
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))


def get_train_test_samples_per_dataset_2_or_4c(args):
    if args.dataset == MNIST_STR:
        if args.fourclass_problem:
            return create_4c_tuple(args)
        else:
            return MNIST_TRAIN_SAMPLES[:args.classes], MNIST_TEST_SAMPLES[:args.classes]
    elif args.dataset == FASHION_MNIST_STR:
        return FASHION_MNIST_TRAIN_SAMPLES[:args.classes], FASHION_MNIST_TEST_SAMPLES[:args.classes]
    elif args.dataset == CIFAR10_STR:
        if args.fourclass_problem:
            return create_4c_tuple(args)
        else:
            return CIFAR10_TRAIN_SAMPLES[:args.classes], CIFAR10_TEST_SAMPLES[:args.classes]
    elif args.dataset == SVHN_STR:
        return SVHN_TRAIN_SAMPLES[:args.classes], SVHN_TEST_SAMPLES[:args.classes]
    elif args.dataset == CIFAR100_STR:
        return CIFAR100_TRAIN_SAMPLES[:args.classes], CIFAR100_TEST_SAMPLES[:args.classes]
    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))
