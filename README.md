# Memorization-Dilation: Modeling Neural Collapse Under Noise

This repository contains an implementation of _Memorization-Dilation: Modeling Neural Collapse Under Noise_ published at ICLR 2023. Cite this work as follows:

```
@inproceedings{DBLP:conf/iclr/NguyenLLHK23,
  author       = {Duc Anh Nguyen and
                  Ron Levie and
                  Julian Lienen and
                  Eyke H{\"{u}}llermeier and
                  Gitta Kutyniok},
  title        = {Memorization-Dilation: Modeling Neural Collapse Under Noise},
  booktitle    = {The Eleventh International Conference on Learning Representations,
                  {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  publisher    = {OpenReview.net},
  year         = {2023}
}
```

## Foreword

As mentioned in the paper, our implementation builds on top of the official implementation of [1], namely from [this repository](https://github.com/tding1/Neural-Collapse). Thus, we re-used large parts of their code, for which we gratefully thank the authors for their work.

## Requirements

To install all required packages, you need to run
~~~
pip install -r requirements.txt
~~~

The code has been tested using Python 3.6, 3.8 and 3.9 on Ubuntu 18.* and Ubuntu 20.* systems. We trained our models on machines with Nvidia GPUs (we tested CUDA 10.1, 11.1 and 11.6). We recommend to use [Python virtual environments](https://docs.python.org/3/tutorial/venv.html) to get a clean Python environment for the execution without any dependency problems.

## Configuration

In the file `config/config.ini`, one can find parameters that are used throughout the project. These include path-related specifications (e.g., where to store results), logging properties and resource constraints for hyperparameter optimization runs.

## Datasets

All datasets are downloaded automatically, there is no need to store them explicitly. All datasets are publicly available and will be stored to `~/data/` by default. One may change this default location of datasets in `args.py` through the argument `--data_dir`.

## Training and Evaluation

### Training (Conventional Noise Experiments)

To train simple models for the conventional noise setting as presented in the paper, one can simply call:

~~~python
python train_simple_model.py --loss LS --model MLP --dataset svhn --batch_size 16 --weight_decay 0.001 --depth 9 --width 2048 --label_noise 0.2 --seed 42 --classes 2 --epochs 200 --use_bn --act_fn relu
~~~

Note that `--help` passed as argument provides a comprehensive overview over all hyperparameters.

For hyperparameter optimization, `train_simple_model_ho.py` provide the implementation for a Skopt hyperparameter optimization (based on Ray) as described in the paper.

### Training (Latent Noise Class Experiments)

For the latent noise class experiments as shown in the appendix, one has to add the parameter `--fourclass_problem` to the previously introduced command. By default, this works for binary classification problems. Moreover, `--fc_noise_degree` allows for specifying the noise degree, i.e., the fraction of class instances assigned to the new latent class.

### Training (Large-Scale Experiments)

To execute the large-scale experiments as included in the appendix, one can call the model training routine as described in [the original repository](https://github.com/tding1/Neural-Collapse), which is offered in `train_model.py`. To conduct the hyperparameter optimization runs, one has to call `train_model_ho.py`. The search space parameters can be specified therein.

### Training (Misc)

Furthermore, we support to train models as in [the original repository](https://github.com/tding1/Neural-Collapse).

### Evaluation (Conventional and Latent Noise Class Experiments)

To evaluate simple models as used in the memorization-dilation experiments, one can simply call as an example:

~~~python
python validate_simple_model.py --loss LS --model MLP --dataset svhn --batch_size 16 --weight_decay 0.001 --depth 9 --width 2048 --label_noise 0.2 --seed 42 --classes 2 --epochs 200 --use_bn --act_fn relu
~~~

While the conventional label noise setting requires no specific parameter to be set, one again has to add the parameter `--fourclass_problem` as in the training script for the latent class noise experiments.

The resulting (NC) metrics and further information is then stored in the file `info.pkl` in the run directory. As before, we also still provide all implementations as in [the original repository](https://github.com/tding1/Neural-Collapse).

### Evaluation (Large-Scale Experiments)

The evaluation of the large-scale experiments can be realized by

~~~python
python validate_NC.py --gpu_id 0 --dataset <identifier> --batch_size 256 --load_path <path to the uid name>
~~~
with the corresponding parameters of the training run.


## References

[1]: Zhu et al. A Geometric Analysis of Neural Collapse with Unconstrained Features. arXiv:2105.02375, 2021.
