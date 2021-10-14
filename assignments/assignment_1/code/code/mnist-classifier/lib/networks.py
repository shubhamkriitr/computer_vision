import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'mlp'

        # Define the network layers in order.
        # Input is 28 * 28.
        # Output is 10 values (one per class).
        # Multiple linear layers each followed by a ReLU non-linearity (apart from the last).
        num_d = 784
        mode = "1"
        if mode == "0":
            print(f"Using single layer")
            self.layers = nn.Sequential(
                nn.Linear(num_d, 10, bias=True, dtype=torch.float32),
                # nn.Softmax() -- removed as not mentioned in tutorial
            )
        else:
            print(f"Using Multiple layers (MLP)")
            self.layers = nn.Sequential(
                nn.Linear(num_d, 32, bias=True, dtype=torch.float32),
                nn.ReLU(),
                nn.Linear(32, 10, bias=True, dtype=torch.float32),
                # nn.Softmax() -- removed as not mentioned in tutorial
            )
    
    def forward(self, batch):
        # Flatten the batch for MLP.
        b = batch.size(0)
        batch = batch.view(b, -1)
        # Process batch using the layers.
        x = self.layers(batch)
        return x


class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'conv'

        # Define the network layers in order.
        # Input is 28x28, with one channel.
        # Multiple Conv2d and MaxPool2d layers each followed by a ReLU non-linearity (apart from the last).
        # Needs to end with AdaptiveMaxPool2d(1) to reduce everything to a 1x1 image.
        raise NotImplementedError()
        self.layers = nn.Sequential(
            nn.AdaptiveMaxPool2d(1)
            # TODO
        )
        # Linear classification layer.
        # Output is 10 values (one per class).
        self.classifier = nn.Sequential(
            # TODO
        )
    
    def forward(self, batch):
        # Add channel dimension for conv.
        b = batch.size(0)
        batch = batch.unsqueeze(1)
        # Process batch using the layers.
        x = self.layers(batch)
        x = self.classifier(x.view(b, -1))
        return x
