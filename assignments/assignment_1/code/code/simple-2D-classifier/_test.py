from lib.dataset import Simple2DDataset

import os

if __name__ == "__main__":
    ds = Simple2DDataset()

"""
[ (name, param.data )for name, param in net.named_parameters() if param.requires_grad]

[Epoch 04] Loss: 0.7035
[Epoch 04] Acc.: 50.0000%

https://datascience.stackexchange.com/questions/14027/counting-the-number-of-layers-in-a-neural-network

Once this is done, you should be able to run the train.py script. What accuracy do you achieve
with the linear classifier? Is this an expected result? Justify your answer.


Switch to the new network by uncommenting L83 in train.py. What accuracy does this network
obtain? Why are the results better compared to the previous classifier?
"""
