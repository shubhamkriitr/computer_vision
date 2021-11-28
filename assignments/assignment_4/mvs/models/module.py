import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        
        extraction_layers = []

        configuration_ = [
            # filter, stride, in_channels, out_channels
            ((3, 3), 1, 3, 8, (1, 1)),
            ((3, 3), 1, 8, 8, (1, 1)),
            ((5, 5), 2, 8, 16, (2, 2)),
            ((3, 3), 1, 16, 16, (1, 1)),
            ((3, 3), 1, 16, 16, (1, 1)),
            ((5, 5), 2, 16, 32, (2, 2)),
            ((3, 3), 1, 32, 32, (1, 1)),
            ((3, 3), 1, 32, 32, (1, 1))
        ]

        for conf in configuration_:
            filter_, stride_, in_channnels_, out_channels_, padding_ = conf
            conv_layer =  nn.Conv2d(in_channels=in_channnels_, out_channels=out_channels_,
                kernel_size=filter_, stride=stride_, padding=padding_)
            bn_layer = nn.BatchNorm2d(num_features=out_channels_)
            relu_layer = nn.ReLU()
            for layer in [conv_layer, bn_layer, relu_layer]:
                extraction_layers.append(layer)

        final_conv_layer = nn.Conv2d(32, 32, (3, 3), 1, padding=(1,1))
        extraction_layers.append(final_conv_layer)

        self.module_list = nn.ModuleList(extraction_layers)


    def forward(self, x):
        # x: [B,3,H,W]
        # TODO:A
        out_ = x
        for layer in self.module_list:
            out_ = layer(out_)
        
        return out_


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        
        self.conv_0 = nn.Conv2d(in_channels=G, out_channels=8,
                kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv_1 = nn.Conv2d(in_channels=8, out_channels=16,
                kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32,
                kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.convt_0 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                kernel_size=(3,3), stride=2,
                padding=1, output_padding=1) 
        self.convt_1 = nn.ConvTranspose2d(in_channels=16, out_channels=8,
                kernel_size=(3,3), stride=2,
                padding=1, output_padding=1)
        self.final_conv2d = nn.Conv2d(in_channels=8, out_channels=1,
                kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.relu = nn.ReLU()



    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        x_in = torch.transpose(x, 1, 2) # [B,D,G,H,W]
        B,D,G,H,W = x_in.shape
        x_in = torch.reshape(x_in, (B*D, G, H, W))
        c_0 = self.conv_0(x_in)
        c_0 = self.relu(c_0)

        c_1 = self.conv_1(c_0)
        c_1 = self.relu(c_1)

        c_2 = self.conv_2(c_1)
        c_2 = self.relu(c_2)

        c_3 = self.convt_0(c_2)

        c_1_3 = c_1 + c_3

        c_4 = self.convt_1(c_1_3)

        c_0_4 = c_0 + c_4

        S_bar = self.final_conv2d(c_0_4) # B*D x 1 x H x W

        out = torch.reshape(S_bar, (B, D, H, W))

        return out



def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B,C,H,W = src_fea.size()
    D = depth_values.size(1)
    warped_src_fea = None
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
        
        det_M_sign = torch.sign(torch.det(rot).unsqueeze(dim=1)) # B x 1
        norm_m_3 = torch.norm(rot[:, 2, :], p=2, dim=1).unsqueeze(dim=1) # B x1
        w = det_M_sign*norm_m_3*depth_values # B x D

        # Create (x, y, 1)^T

        one_col = torch.ones_like(x)
        x_hom = torch.cat([x.unsqueeze(dim=1),  # H*W x 3
                        y.unsqueeze(dim=1),
                        one_col.unsqueeze(dim=1)], dim=1)
        Xw = w.view(B, D, 1, 1)*x_hom.unsqueeze(dim=0).unsqueeze(dim=0) # B x D x H*W x 3

        one_shape = list(Xw.shape)
        one_shape[-1] = 1
        ones_to_pad = torch.ones(one_shape)

        X_cam = torch.cat((Xw, ones_to_pad), dim=3) # B x D x H*W x 4

        X_cam = X_cam.transpose(1, 3) # B x 4 x H*W x D

        X_cam = torch.reshape(X_cam, shape=(B, 4, H*W*D)) # B x 4 x H*W*D

        X_cam_2 = torch.matmul(proj, X_cam)

        X_cam_2 = torch.reshape(X_cam_2, shape=(B, 4, H*W, D)).transpose(1, 3) # B x D x H*W x 4

        x_cam_2 = X_cam_2[:, :, :, 0:2]/X_cam_2[:, :, :, 2:3] # B x D x H*W x 2

        # swap x, y position as H->y W->x
        swap_index = torch.tensor([1, 0])
        x_cam_2[:, :, :, swap_index] = x_cam_2

        out_ = F.grid_sample(src_fea, x_cam_2) # B x C x D x H*W

        warped_src_fea = out_.reshape(B, C, D, H, W)



    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO  
    
    return warped_src_fea

def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B, C, D, H, W = warped_src_fea.shape
    k = int(C/G)

    T = torch.reshape(ref_fea, (B, k, G, 1, H, W))
    U = torch.reshape(warped_src_fea, (B, k, G, D, H, W))

    out = T * U

    out = torch.sum(out, dim=1)

    out = out/k

    return out






def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    d = depth_values.unsqueeze(dim=2).unsqueeze(dim=3)
    prob_wtd_depth = d*p
    depth_map = torch.sum(prob_wtd_depth, dim=1)
    return depth_map

def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    pass


if __name__ == "__main__":
    net = FeatureNet()
    print(net)
    net2 = SimlarityRegNet(8)
    print(net2)
