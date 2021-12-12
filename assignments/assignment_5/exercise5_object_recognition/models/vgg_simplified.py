import torch
import torch.nn as nn
import math

class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        # todo:A construct the simplified VGG network blocks
        # input shape: [bs, 3, 32, 32]
        # layers and output feature shape for each block:
        # # conv_block1 (Conv2d, ReLU, MaxPool2d) --> [bs, 64, 16, 16]
        # # conv_block2 (Conv2d, ReLU, MaxPool2d) --> [bs, 128, 8, 8]
        # # conv_block3 (Conv2d, ReLU, MaxPool2d) --> [bs, 256, 4, 4]
        # # conv_block4 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 2, 2]
        # # conv_block5 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 1, 1]
        # # classifier (Linear, ReLU, Dropout2d, Linear) --> [bs, 10] (final output)

        # hint: stack layers in each block with nn.Sequential, e.x.:
        # # self.conv_block1 = nn.Sequential(
        # #     layer1,
        # #     layer2,
        # #     layer3,
        # #     ...)

        all_conv_block_layers = []
        self.classifier_layers = []

        model_configuration = [
            # filter, stride, in_channels, out_channels, padding
            ((3, 3), 1, 3, 64, (1, 1)),
            ((3, 3), 1, 64, 128, (1, 1)),
            ((3, 3), 1, 128, 256, (1, 1)),
            ((3, 3), 1, 256, 512, (1, 1)),
            ((3, 3), 1, 512, 512, (1, 1)),

        ]

        # conv - relu -maxpool

        for conf in model_configuration:
            filter_, stride_, in_channnels_, out_channels_, padding_ = conf
            conv_layer =  nn.Conv2d(in_channels=in_channnels_, out_channels=out_channels_,
                kernel_size=filter_, stride=stride_, padding=padding_)
            relu_layer = nn.ReLU()
            maxpool_layer = nn.MaxPool2d(kernel_size=2)
            for layer in [conv_layer, relu_layer, maxpool_layer]:
                all_conv_block_layers.append(layer)
        
        layer_idx = 0
        self.conv_block1 = nn.Sequential(*all_conv_block_layers[layer_idx:layer_idx+3])
        layer_idx += 3
        self.conv_block2 = nn.Sequential(*all_conv_block_layers[layer_idx:layer_idx+3])
        layer_idx += 3
        self.conv_block3 = nn.Sequential(*all_conv_block_layers[layer_idx:layer_idx+3])
        layer_idx += 3
        self.conv_block4 = nn.Sequential(*all_conv_block_layers[layer_idx:layer_idx+3])
        layer_idx += 3
        self.conv_block5 = nn.Sequential(*all_conv_block_layers[layer_idx:layer_idx+3])
        del layer_idx


        # Linear, ReLU, Dropout2d, Linear
        linear_1 = nn.Linear(512, self.fc_layer)
        relu_1 = nn.ReLU()
        dropout_1 = nn.Dropout2d(p=0.5)
        linear_2 = nn.Linear(self.fc_layer, self.classes)

        classifier_layers = [linear_1, relu_1, dropout_1, linear_2]
        self.classifier = nn.Sequential(*classifier_layers)




        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print("Initializing: {}".format(m))
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """
        score = None
        # todo
        

        return score

