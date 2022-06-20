import torch.nn as nn
from torch.nn import Sigmoid
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multilayer perceptron with a parameterizable number and size of hidden layers, a configurable fully connected
    (linear) classification head, exchangeable activation functions and normalization techniques.
    """

    def __init__(self, hidden, depth=6, fc_bias=True, num_classes=10, penultimate_layer_features=2,
                 final_activation="relu", use_bn=True, use_layer_norm=False, input_dim=3072):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden)]
        if use_bn:
            layers += [nn.BatchNorm1d(num_features=hidden)]
        if use_layer_norm:
            layers += [nn.LayerNorm(hidden)]
        # layers += [nn.ReLU()]
        if final_activation == "relu":
            layers += [nn.ReLU()]
        elif final_activation == "sigmoid":
            layers += [Sigmoid()]

        for i in range(depth - 2):
            layers += [nn.Linear(hidden, hidden)]
            if use_bn:
                layers += [nn.BatchNorm1d(num_features=hidden)]
            if use_layer_norm:
                layers += [nn.LayerNorm(hidden)]
            # layers += [nn.ReLU()]
            if final_activation == "relu":
                layers += [nn.ReLU()]
            elif final_activation == "sigmoid":
                layers += [Sigmoid()]

        if final_activation == "relu":
            final_activation = nn.ReLU()
        elif final_activation == "sigmoid":
            final_activation = Sigmoid()
        else:
            final_activation = None

        layers += [nn.Linear(hidden, penultimate_layer_features)]

        if final_activation is not None:
            if use_bn:
                layers += [nn.BatchNorm1d(num_features=penultimate_layer_features)]
            if use_layer_norm:
                layers += [nn.LayerNorm(penultimate_layer_features)]

            layers += [final_activation]

        self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(penultimate_layer_features, num_classes, bias=fc_bias)
        self.fc_activation = None
        if num_classes == 1:
            # In this case, we use a sigmoid activation that is combined with binary cross-entropy
            self.fc_activation = Sigmoid()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        features = F.normalize(x)
        x = self.fc(x)
        if self.fc_activation is not None:
            x = self.fc_activation(x)
        return x, features
