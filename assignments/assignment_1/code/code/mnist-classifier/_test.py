from lib.dataset import MNISTDataset

train_data, test_data = MNISTDataset("train"), MNISTDataset("test")

train_data.__getitem__(1)
x = 1


# TEST
# TEST
# TEST

class ConvClassifier2(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'conv'

        # Define the network layers in order.
        # Input is 28x28, with one channel.
        # Multiple Conv2d and MaxPool2d layers each followed by a ReLU non-linearity (apart from the last).
        # Needs to end with AdaptiveMaxPool2d(1) to reduce everything to a 1x1 image.
        
        self.l_c1 = nn.Conv2d(1, 8, (3, 3))
        self.l_r1 = nn.ReLU()
        self.l_m1 = nn.MaxPool2d((2, 2), 2)

        self.l_c2 = nn.Conv2d(8, 16, (3, 3))
        self.l_r2 = nn.ReLU()
        self.l_m2 = nn.MaxPool2d((2, 2), 2)

        self.l_c3 = nn.Conv2d(16, 32, (3, 3))
        self.l_r3 = nn.ReLU()
        self.l_m3 = nn.AdaptiveMaxPool2d(1)
        
        # Linear classification layer.
        # Output is 10 values (one per class).
        self.classifier = nn.Sequential(
            nn.Linear(4096, 10)
        )
    
    def forward(self, batch):
        # Add channel dimension for conv.
        b = batch.size(0)
        batch = batch.unsqueeze(1)
        # Process batch using the layers.
        
        x_ = self.l_c1(batch)
        x_ = self.l_r1(x_)
        x_ = self.l_m1(x_)

        x_ = self.l_c2(x_)
        x_ = self.l_r2(x_)
        x_ = self.l_m2(x_)

        

        x_ = self.l_c3(x_)
        x_ = self.l_r3(x_)
        x_ = self.l_m3(x_)

        x = self.classifier(x_.view(b, -1))
        return x