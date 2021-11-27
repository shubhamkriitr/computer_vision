import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        
        extraction_layers = []

        configuration_ = [
            # filter, stride, in_channels, out_channels
            ((3, 3), 1, 3, 8),
            ((3, 3), 1, 8, 8),
            ((5, 5), 2, 8, 16),
            ((3, 3), 1, 16, 16),
            ((3, 3), 1, 16, 16),
            ((5, 5), 2, 16, 32),
            ((3, 3), 1, 32, 32),
            ((3, 3), 1, 32, 32)
        ]

        for conf in configuration_:
            filter_, stride_, in_channnels_, out_channels_ = conf
            conv_layer =  nn.Conv2d(in_channels=in_channnels_, out_channels=out_channels_,
                kernel_size=filter_, stride=stride_)
            bn_layer = nn.BatchNorm2d(num_features=out_channels_)
            relu_layer = nn.ReLU()
            for layer in [conv_layer, bn_layer, relu_layer]:
                extraction_layers.append(layer)

        final_conv_layer = nn.Conv2d(32, 32, (3, 3), 1)
        extraction_layers.append(final_conv_layer)

        self.module_list = nn.ModuleList(extraction_layers)


    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        pass


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        pass


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        # TODO

    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO

    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    pass


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    pass

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    pass


if __name__ == "__main__":
    net = FeatureNet()
    print(net)
    