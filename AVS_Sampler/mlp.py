import torch.nn as nn

class FullyConnectedMLP(nn.Module):
    def __init__(self, input_shape, hiddens, output_shape):
        assert isinstance(hiddens, list)
        super().__init__()
        self.input_shape = (input_shape,)
        self.output_shape = (output_shape,)
        self.hiddens = hiddens
        model = []
        # Stack Dense layers with ReLU activation.
        # Note that you do not have to add relu after the last dense layer
        in_features = self.input_shape[0]
        for out_features in self.hiddens:
            model += [nn.Linear(in_features, out_features)]
#             model += [nn.ReLU()]
            model += [nn.Tanh()]
            in_features = out_features
        out_features = self.output_shape[0]
        model += [nn.Linear(in_features, out_features)]
        self.net = nn.Sequential(*model)
        
    def forward(self, x):
        # apply network that was defined in __init__ and return the output
        return self.net(x)