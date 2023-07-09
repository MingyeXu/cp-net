import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pytorch_utils as pt_utils
from pointnet2_modules import PointnetSAModule, PointnetSAModuleMSGRSCNN, PointnetFPModule
import numpy as np

import torch.nn.functional as F

from GDANet_util import local_operator, GDM

def ShapeMask(feats, mask):
    '''
    feats B，C，N
    mask  B，16，N
    '''
    B,C,N = feats.size()
    C2 = int(C//16)
    mask = mask.unsqueeze(1).expand(-1, C2,-1, -1) # B, 32,16,N
    feats_ = feats[:,:,:].view(B,C2,16,N)
    # feats = feats[:,:,:] + (feats_ * mask).view(B,-1,N)
    feats = (feats_ * mask).view(B,-1,N)

    return feats


class MetricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss().cuda()

    def get_metric_loss(self, x, ref):
        '''
        :param x: (bs, n_rkhs)
        :param ref: (bs, n_rkhs, n_loc)
        :return: loss
        '''

        bs, n_rkhs, n_loc = ref.size()
        ref = ref.transpose(0, 1).reshape(n_rkhs, -1)
        score = torch.matmul(x, ref) * 64.  # (bs * n_loc, bs)
        score = score.view(bs, bs, n_loc).transpose(1, 2).reshape(bs * n_loc, bs)
        gt_label = torch.arange(bs, dtype=torch.long, device=x.device).view(bs, 1).expand(bs, n_loc).reshape(-1)
        return self.ce(score, gt_label)

    def forward(self, x, refs):
        loss = 0.
        for ref in refs:
            loss += self.get_metric_loss(x, ref)
        return loss

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    print(src.type())
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        '''
        :param x: (bs, np, 3)
        :param y: (bs, np, 3)
        :return: loss
        '''

        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        dist = torch.sqrt(1e-6 + torch.sum(torch.pow(x - y, 2), 3)) # bs, ny, nx
        min1, _ = torch.min(dist, 1)
        min2, _ = torch.min(dist, 2)

        return min1.mean() + min2.mean()

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # print('xyz1',xyz1.size())
        # print('xyz2',xyz2.size())
        # print('points1',points1.size())
        # print('points2',points2.size())
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        return new_points

class Normalize(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=self.dim, keepdim=True)
        return x / norm

class RSCNN_SSN(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, n_rkhs, input_channels=0, relation_prior=1, use_xyz=True, point_wise_out=False, multi=1.0):
        super().__init__()

        self.SA_modules = nn.ModuleList()

        print('Using', multi, 'times larger RSCNN model')

        self.point_wise_out = point_wise_out

        self.SA_modules.append(
            PointnetSAModuleMSGRSCNN(
                npoint=1024,
                radii=[0.075, 0.1, 0.125],
                nsamples=[16, 32, 48],
                mlps=[[input_channels, int(multi * 64)],[input_channels, int(multi * 64)],[input_channels, int(multi * 64)]],
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_1 = 64*3
        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSGRSCNN(
                npoint=256,
                radii=[0.1, 0.15, 0.2],
                nsamples=[16, 48, 64],
                mlps=[[c_in, 128], [c_in, 128], [c_in, 128]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_2 = 128*3
        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSGRSCNN(
                npoint=64,
                radii=[0.2, 0.3, 0.4],
                nsamples=[16, 32, 48],
                mlps=[[c_in, 256], [c_in, 256], [c_in, 256]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_3 = 256*3
        c_in = c_out_3
        self.SA_modules.append(    # 3
            PointnetSAModuleMSGRSCNN(
                npoint=16,
                radii=[0.4, 0.6, 0.8],
                nsamples=[16, 24, 32],
                mlps=[[c_in, 512], [c_in, 512], [c_in, 512]],
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_4 = 512*3
        self.SA_modules.append(   # 4   global pooling
            PointnetSAModule(
                nsample = 16,
                mlp=[c_out_4, 128], use_xyz=use_xyz
            )
        )
        global_out = 128
        
        self.SA_modules.append(   # 5   global pooling
            PointnetSAModule(
                nsample = 64,
                mlp=[c_out_3, 128], use_xyz=use_xyz
            )
        )
        global_out2 = 128




        # self.prediction_modules = nn.ModuleList()

        # mid_channel = min(int(multi * 128), n_rkhs)
        # self.prediction_modules.append(
        #     nn.Sequential(
        #         nn.Conv1d(int(multi * 128), mid_channel, 1),
        #         nn.BatchNorm1d(mid_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Conv1d(mid_channel, n_rkhs, 1),
        #         Normalize(dim=1)
        #     )
        # )

        # mid_channel = min(int(multi * 512), n_rkhs)
        # self.prediction_modules.append(
        #     nn.Sequential(
        #         nn.Conv1d(int(multi * 512), mid_channel, 1),
        #         nn.BatchNorm1d(mid_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Conv1d(mid_channel, n_rkhs, 1),
        #         Normalize(dim=1)
        #     )
        # )

        # mid_channel = min(int(multi * 1024), n_rkhs)
        # self.prediction_modules.append(
        #     nn.Sequential(
        #         nn.Conv1d(int(multi * 1024), mid_channel, 1),
        #         nn.BatchNorm1d(mid_channel),
        #         nn.ReLU(inplace=True),
        #         nn.Conv1d(mid_channel, n_rkhs, 1),
        #         Normalize(dim=1)
        #     )
        # )
        # self.fp2 = PointNetFeaturePropagation(390, [512, 512])
        # self.fp1 = PointNetFeaturePropagation(198, [128, 128])
        # self.fp2 = PointnetFPModule(mlp= [1667, 512])
        # self.fp1 = PointnetFPModule(mlp= [646, 128])
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[256, 128,128], in_channels=input_channels)
        )
        self.FP_modules.append(PointnetFPModule(mlp=[512, 256,256],in_channels=c_out_1))
        self.FP_modules.append(PointnetFPModule(mlp=[512, 512,512],in_channels=c_out_2))
        self.FP_modules.append(
            PointnetFPModule(mlp=[c_out_4, 512,512],in_channels = c_out_3)
        )

        self.FP_modules2 = nn.ModuleList()
        self.FP_modules2.append(PointnetFPModule(mlp=[256, 128],in_channels=128))
        self.FP_modules2.append(PointnetFPModule(mlp=[512, 128],in_channels=128))
        self.FP_modules2.append(PointnetFPModule(mlp=[512, 64],in_channels=128))
        self.FP_modules2.append(PointnetFPModule(mlp=[1664-128,512, 64],in_channels=128))


        self.adaptive_maxpool = nn.AdaptiveMaxPool1d(1)

        if point_wise_out:
            self.sharemlp = nn.Sequential(
                nn.Conv1d(515, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            )
            self.upsample = nn.Sequential(
                nn.Conv1d(128, 3, 1),
                Normalize(dim=1)
            )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud, get_feature=False):
        """
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """

        # mask = mask.unsqueeze(-1).expand(-1, -1, 2048)
        # cate = cate.unsqueeze(-1).expand(-1, -1,-1, 2048) # B, 32,16,N

        xyz, features = self._break_up_pc(pointcloud)
        
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            if i < 5:
                # print('LAYER:',i)
                li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
                # print('li_features:',li_features.size())
                if li_xyz is not None:
                    random_index = np.arange(li_xyz.size()[1])
                    np.random.shuffle(random_index)
                    li_xyz = li_xyz[:, random_index, :]
                    li_features = li_features[:, :, random_index]
                l_xyz.append(li_xyz)
                l_features.append(li_features)


        # g_encoder = torch.cat([self.adaptive_maxpool(now_out) for now_out in l_features[1:-1]], dim=1)


        _, global_out2_feat = self.SA_modules[5](l_xyz[3], l_features[3])
        # print('GLOBAL',global_out2_feat.size())
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            # print(i)
            # l_features[i - 1] = ShapeMask(l_features[i - 1],mask)
            l_features[i - 1 - 1] = self.FP_modules[i](
                l_xyz[i - 1 - 1], l_xyz[i - 1], l_features[i - 1 - 1], l_features[i - 1]
            )

        for i in range(-1, -(len(self.FP_modules)+1), -1):
            # print(i-1)
            # print('In:',l_features[i - 1].size())
            l_features[i - 1] = self.FP_modules2[i](
                l_xyz[0], l_xyz[i - 1], l_features[0], l_features[i - 1]
            )
            # print('Out:',l_features[i - 1].size())
            # print(l_features[i - 1].size())

        # exit(0)
        # for i in range(len(l_features[1:-1])):
        #     print(l_features[i+1].size())


            # print('FP',l_features[i - 1 - 1].size())
        
        # out = []
        # feats = [features]
        # xyz_bank = [xyz]
        # for module, prediction_modules in zip(self.SA_modules, self.prediction_modules):
        #     xyz, features = module(xyz, features) # xyz[B,N,3] feature [B,C,N]
        #     # print('XYZ:',xyz.size())
        #     # print('features:',features.size())
        #     out.append(prediction_modules(features))
        #     xyz_bank.append(xyz)
        #     feats.append(features)

        # # exit(0)
        # x3 = torch.cat([xyz_bank[2].permute(0,2,1).contiguous(), feats[2], feats[3].expand(-1, -1, 128)], dim=1)
        # x2 = self.fp2(xyz_bank[1],xyz_bank[2],feats[1],x3)
        # x2 = torch.cat([xyz_bank[1].permute(0,2,1).contiguous(), feats[1], x2], dim=1)
        # # print('XYZ',xyz_bank[0].size())
        # # print('FEATS',feats[0].size())
        # if feats[0] == None:
        #     x1 = xyz_bank[0].permute(0,2,1).contiguous()
        # else:
        #     x1 = torch.cat([xyz_bank[0].permute(0,2,1).contiguous(), feats[0]], dim=1)
        # x1 = self.fp1(xyz_bank[0],xyz_bank[1],x1,x2)





        if not get_feature:
            if not self.point_wise_out:
                return out, torch.cat([self.adaptive_maxpool(now_out).squeeze(2) for now_out in out], dim=1)
            else:
                # global_feature = torch.cat([self.adaptive_maxpool(now_out).squeeze(2) for now_out in out], dim=1)
                global_feature = global_out2_feat
                interpolated_feats = global_feature.expand(-1, -1, 2048)
                pt_feat=torch.cat(l_features[1:-1], dim=1)
                # print("DFGHJK",pt_feat.size())
                final_feature = torch.cat(
                    [ interpolated_feats, pt_feat, xyz.permute(0,2,1).contiguous()],
                    dim=1)
                save_feature = torch.cat(
                    [ pt_feat, xyz.permute(0,2,1).contiguous()],
                    dim=1)
                # print("ewfewff",final_feature.size())
                point_wise_feats = self.sharemlp(final_feature)
                point_wise_pred = self.upsample(point_wise_feats)
                g_decoder = torch.cat([self.adaptive_maxpool(now_out) for now_out in l_features[1:-1]], dim=1)
                # print('1',g_decoder.size())
                # print('2',global_feature.size())
                global_feature = torch.cat([self.adaptive_maxpool(point_wise_feats),global_feature],dim=1)
                # global_feature = torch.cat([g_encoder, g_decoder,self.adaptive_maxpool(point_wise_feats),global_feature],dim=1)

                # print(global_feature.size())
                return global_feature, l_features[1:-1], point_wise_pred, point_wise_feats, l_features[-2]
        else:
            global_feature = torch.cat([self.adaptive_maxpool(now_out).squeeze(2) for now_out in out], dim=1)
            return global_feature, out, xyz_bank

            
                # # global_feature = torch.cat([self.adaptive_maxpool(now_out).squeeze(2) for now_out in out], dim=1)
                # global_feature = global_out2_feat
                # interpolated_feats = global_feature.expand(-1, -1, 2048)
                # final_feature = torch.cat(
                #     [ interpolated_feats, l_features[0], xyz.permute(0,2,1).contiguous()],
                #     dim=1)
                # save_feature = torch.cat(
                #     [ l_features[0], xyz.permute(0,2,1).contiguous()],
                #     dim=1)                    
                # point_wise_feats = self.sharemlp(final_feature)
                # point_wise_pred = self.upsample(point_wise_feats)
                # global_feature = torch.cat([self.adaptive_maxpool(point_wise_feats),global_feature],dim=1)
                # # print(global_feature.size())
                # # IIC_feats = F.log_softmax(self.toIIC(point_wise_feats),dim=1)
                # return global_feature, l_features, point_wise_pred, save_feature



class MainModule(nn.Module):
    # def __init__(self, feature_size, out_dim, num_points,  final_layer=False):
    #   super(DiffPool, self).__init__()
    def __init__(self, n_rkhs, args):
        super(MainModule, self).__init__()
        self.encoder = RSCNN_SSN(n_rkhs=n_rkhs, input_channels=args.input_channels, relation_prior=args.relation_prior, use_xyz=True, point_wise_out=True, multi=args.multiplier)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn11)
    def forward(self,x, points, points_con):
        # print()
        # x = points.data.clone()

        # x1 = local_operator(x.permute(0,2,1).contiguous(), k=50)

        # x1 = F.relu(self.conv1(x1))
        # x1 = F.relu(self.conv11(x1))
        # x1 = x1.max(dim=-1, keepdim=False)[0]

        x1_s, x1_g,xss,xgg = GDM(x.permute(0,2,1).contiguous(),points.permute(0,2,1), M=1024)

        # print(x1_s.size())

        global_feature1, infoNCE_feats1, point_wise_pred1, point_wise_feats1,saveFeats = self.encoder(points)
        global_feature2, infoNCE_feats2, point_wise_pred2, point_wise_feats2,_ = self.encoder(x1_g)
        # print('ghjsdfk')
        # global_feature = torch.cat([global_feature1,global_feature2],dim=1)
        # global_feature = global_feature1

        return infoNCE_feats1, infoNCE_feats2, point_wise_pred1, point_wise_feats1, point_wise_feats2,global_feature1, global_feature2,saveFeats,x1_g,_,xss,xgg