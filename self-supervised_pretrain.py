import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import os
from torchvision import transforms
from models.rscnn_ssn_ss_segmentation import RSCNN_SSN, ChamferLoss, MainModule
from models.pointnet2_ss import PointNet2, NormalLoss

from models.foldingnet import FoldingNet, NormalNet
from data import ModelNetCls, ScanObjectNNCls, ScanNetCls, shapeNet_dataset
from data.ShapeNetPart_Loader import PartNormalDataset


import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml
from sklearn.svm import LinearSVC
import scipy.io as sio

import h5py


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from visdom import Visdom
viz = Visdom(env = 'DataAUG_rotate15_AE_DropPart')


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False



class NCESoftmaxLoss(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, label):
        loss = self.criterion(x, label)
        return loss
def infoNCE(pt_wisefeats, pt_wisefeats_con, n_sample):
    pt_wisefeats = pt_wisefeats.permute(0,2,1).contiguous()
    pt_wisefeats_con = pt_wisefeats_con.permute(0,2,1).contiguous()

    f_v1 = pt_wisefeats
    f_v2 = pt_wisefeats_con


    bs = f_v1.size(0)
    inds = np.random.choice(f_v1.size(1), n_sample, replace=False)

    f_v1 = f_v1[:,inds,:]
    f_v2 = f_v2[:,inds,:]

    logits = torch.matmul(f_v1, f_v2.transpose(2, 1))
    logits = logits.view(-1, n_sample)
    
    # print(logits.size())
    labels = torch.arange(n_sample).view(1, n_sample).expand(bs, n_sample).reshape(-1).cuda().long()

    criterion = NCESoftmaxLoss().cuda()
    loss_infoNCE = criterion(logits, labels)
    return loss_infoNCE





parser = argparse.ArgumentParser(description='Global-Local Reasoning Training')
parser.add_argument('--config', default='cfgs/config.yaml', type=str)
parser.add_argument('--name', default='default', type=str)
parser.add_argument('--arch', default='pointnet2', type=str)
parser.add_argument('--dataset', default='modelnet', type=str)
parser.add_argument('--dataroot', default='/home/xumingye/AAAI/data/hdf5_data', type=str)
parser.add_argument('--jitter', default=False, help="randomly jitter point cloud")
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def main():
    global svm_best_acc40
    svm_best_acc40 = 0
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")

    os.makedirs('./ckpts/', exist_ok=True)

    # dataset
    train_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])


    ss_dataset = PartNormalDataset(npoints=2048, split='trainval',normalize=True, jitter=args.jitter)
    ss_dataloader = torch.utils.data.DataLoader(ss_dataset, batch_size=args.batch_size,shuffle=True, num_workers=torch.cuda.device_count())



    if args.dataset == 'modelnet':
        train_dataset = ModelNetCls(transforms=train_transforms, self_supervision=False, train=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            pin_memory=True, worker_init_fn=worker_init_fn
        )

        test_dataset = ModelNetCls(transforms=test_transforms, self_supervision=False, train=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=True
        )
    elif args.dataset == 'scannet':
        train_dataset = ScanNetCls(transforms=train_transforms, self_supervision=False, train=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            pin_memory=True, worker_init_fn=worker_init_fn
        )

        test_dataset = ScanNetCls(transforms=test_transforms, self_supervision=False, train=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=True
        )
    elif args.dataset == 'scanobjectnn':
        train_dataset = ScanObjectNNCls(transforms=train_transforms, self_supervision=False, train=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            pin_memory=True, worker_init_fn=worker_init_fn
        )
        test_dataset = ScanObjectNNCls(transforms=test_transforms, self_supervision=False, train=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=int(args.workers),
            pin_memory=True
        )
    elif args.dataset == 'ShapeNetPart':

        train_dataset = PartNormalDataset(npoints=2048, split='trainval',normalize=True, jitter=args.jitter)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True, num_workers=torch.cuda.device_count())
        test_dataset = PartNormalDataset(npoints=2048, split='test',normalize=True,jitter=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False, num_workers=torch.cuda.device_count())
    else:
        raise NotImplementedError

    # models
    n_rkhs = 512

    if args.arch == 'pointnet2':
        encoder = PointNet2(n_rkhs=n_rkhs, input_channels=args.input_channels, use_xyz=True, point_wise_out=True, multi=args.multiplier)
        print('Using PointNet++ backbone')
    elif args.arch == 'rscnn':
        # encoder = RSCNN_SSN(n_rkhs=n_rkhs, input_channels=args.input_channels, relation_prior=args.relation_prior, use_xyz=True, point_wise_out=True, multi=args.multiplier)
        encoder = MainModule(n_rkhs, args)
        print('Using RSCNN backbone')
    else:
        raise NotImplementedError

    encoder = nn.DataParallel(encoder).cuda()
    decoer = FoldingNet(in_channel=256)
    decoer = nn.DataParallel(decoer).cuda()

    # optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoer.parameters()), lr=args.base_lr, weight_decay=args.weight_decay)

    # resume
    begin_epoch = -1
    checkpoint_name = './ckpts/' + args.name + '_best.pth'
    print(checkpoint_name)
    # exit(0)
    if os.path.isfile(checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoer.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # svm_best_acc40 = checkpoint['svm_best_acc40']
        begin_epoch = checkpoint['epoch'] - 1
        print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_name, begin_epoch))
        print(svm_best_acc40)
        # exit(0)

    lr_lbmd = lambda e: max(args.lr_decay**(e // args.decay_step), args.lr_clip / args.base_lr)
    bnm_lmbd = lambda e: max(args.bn_momentum * args.bn_decay**(e // args.decay_step), args.bnm_clip)
    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=begin_epoch)
    bnm_scheduler = pt_utils.BNMomentumScheduler(encoder, bnm_lmbd, last_epoch=begin_epoch)

    num_batch = len(ss_dataset)/args.batch_size

    args.val_freq_epoch = 1.0
    
    # training & evaluation
    train(ss_dataloader, train_dataloader, test_dataloader, encoder, decoer, optimizer, lr_scheduler, bnm_scheduler, args, num_batch, begin_epoch)
    test(ss_dataloader, train_dataloader, test_dataloader, encoder, decoer, optimizer, lr_scheduler, bnm_scheduler, args, num_batch, begin_epoch)
# 
def train(ss_dataloader, train_dataloader, test_dataloader, encoder, decoer, optimizer, lr_scheduler, bnm_scheduler, args, num_batch, begin_epoch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    PointcloudRotate = d_utils.PointcloudRotate()
    PointcloudJitter = d_utils.PointcloudJitter()
    PointcloudPartDrop = d_utils.PointcloudRandomPartDropout()
    # metric_criterion = MetricLoss()
    chamfer_criterion = ChamferLoss()
    global svm_best_acc40
    batch_count = 0
    encoder.train()
    decoer.train()

    # viz.line([0.],[0.],win='train_loss_metric', opts = dict(title='train_loss_metric'))
    viz.line([0.],[0.],win='train_loss_norm', opts = dict(title='train_loss_norm'))
    viz.line([0.],[0.],win='train_loss_info', opts = dict(title='train_loss_info'))
    global_step = 0

    for epoch in range(begin_epoch, args.epochs):
        np.random.seed()
        for batch_idx, data in enumerate(ss_dataloader,0):
            points, label, target, norm_plt = data
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)
            if bnm_scheduler is not None:
                bnm_scheduler.step(epoch-1)
            points = Variable(points.float()).cuda()
            norm_plt = Variable(norm_plt.float()).cuda()
            points = torch.cat([points,norm_plt],dim = 2)


            # data augmentation
            # sampled_points = 1200
            has_normal = (points.size(2) > 3)

            if has_normal:
                normals = points[:, :, 3:6].contiguous()
            points = points[:, :, 0:3].contiguous()
            if has_normal:
                # normals = pointnet2_utils.gather_operation(normals.transpose(1, 2).contiguous(), fps_idx)
                normals = normals.transpose(1, 2).contiguous()
            points_gt = points
            points = PointcloudScaleAndTranslate(points.data)

            # optimize
            optimizer.zero_grad()
            points_contrast =(PointcloudScaleAndTranslate(PointcloudJitter(PointcloudRotate(points.data.clone()))))
            # points_contrast = PointcloudPartDrop(points_contrast, target)


            # points = points
            x_points = points.data.clone()
            # out, global_feature, saveFeats,_ = encoder(points)
            NCE_feats1, NCE_feats2, normals_pred, ptwiseF1, ptwiseF2,fuse_global1,fuse_global2,_ = encoder(x_points, points, points_contrast)
            # fuse_global, _, normals_pred = encoder(points)
            # global_feature1 = features1[2].squeeze(2)
            # refs1 = features1[0:2]
            recon1 = decoer(fuse_global1).transpose(1, 2)  # bs, np, 3
            recon2 = decoer(fuse_global2).transpose(1, 2)  # bs, np, 3

            # loss_metric = metric_criterion(global_feature1, refs1)
            loss_recon = chamfer_criterion(recon1, points_gt) + chamfer_criterion(recon2, points_gt)

            nceloss1 = infoNCE(NCE_feats1[0], NCE_feats2[0], 512)
            nceloss2 = infoNCE(NCE_feats1[1], NCE_feats2[1], 256)
            nceloss3 = infoNCE(NCE_feats1[2], NCE_feats2[2], 128)
            nceloss4 = infoNCE(NCE_feats1[3], NCE_feats2[3], 128)
            nceloss_p = infoNCE(ptwiseF1, ptwiseF2, 512)
            loss_infoNCE = nceloss1 + nceloss2 +nceloss3 +nceloss_p + nceloss4


            if has_normal:
                loss_normals = NormalLoss(normals_pred, normals)
            else:
                loss_normals = normals_pred.new(1).fill_(0)
            loss = loss_infoNCE + loss_normals + loss_recon 
            loss.backward()
            global_step += 1
            loss_metric = normals_pred.new(1).fill_(0)
            # loss_recon = normals_pred.new(1).fill_(0)
            # viz.line([loss_metric.item()],[global_step],win = 'train_loss_metric', update = 'append') 
            viz.line([loss_normals.item()],[global_step],win = 'train_loss_norm', update = 'append') 
            viz.line([loss_infoNCE.item()],[global_step],win = 'train_loss_info', update = 'append') 


            optimizer.step()
            if batch_idx % args.print_freq_iter == 0:
                print('[epoch %3d: %3d/%3d] \t INFO/chamfer/normal loss: %0.6f/%0.6f/%0.6f \t lr: %0.5f' % (epoch+1, batch_idx, num_batch, loss_infoNCE.item(), loss_recon.item(), loss_normals.item(), lr_scheduler.get_lr()[0]))
            batch_count += 1
            
            # validation
            # if args.evaluate and batch_count % int(args.val_freq_epoch * num_batch) == 0:
            #     svm_acc40 = validate(train_dataloader, test_dataloader, encoder, args)
            if  epoch % 2 == 0:
                save_dict = {'epoch': epoch + 1,  # after training one epoch, the start_epoch should be epoch+1
                             'optimizer_state_dict': optimizer.state_dict(),
                             'encoder_state_dict': encoder.state_dict(),
                             'decoder_state_dict': decoer.state_dict(),
                            #  'svm_best_acc40': svm_best_acc40,
                             }
                checkpoint_name = './ckpts/' + args.name + '.pth'
                torch.save(save_dict, checkpoint_name)
                # if svm_acc40 == svm_best_acc40:
                checkpoint_name = './ckpts/' + args.name + '_best.pth'
                torch.save(save_dict, checkpoint_name)

def test(ss_dataloader, train_dataloader, test_dataloader, encoder, decoer, optimizer, lr_scheduler, bnm_scheduler, args, num_batch, begin_epoch):
    PointcloudScaleAndTranslate = d_utils.PointcloudScaleAndTranslate()   # initialize augmentation
    PointcloudRotate = d_utils.PointcloudRotate()
    PointcloudJitter = d_utils.PointcloudJitter()
    # metric_criterion = MetricLoss()
    chamfer_criterion = ChamferLoss()

    batch_count = 0

    encoder.eval()
    decoer.eval()

    test_features = []
    test_label = []
    test_cate = []

    train_features = []
    train_label = []
    train_cate = []

    train_reco = []
    train_xyz0 = []
    train_xyz1 = []
    train_xyz2 = []
    # train_hm1 = []
    # train_hm2 = []
    train_out1 = []
    train_out2 = []


    test_reco = []
    test_xyz0 = []
    test_xyz1 = []
    test_xyz2 = []
    # test_hm1 = []
    # test_hm2 = []
    test_out1 = []
    test_out2 = []

    PointcloudRotate = d_utils.PointcloudRotate()

    # feature extraction
    with torch.no_grad():

        ii=0
        for j, data in enumerate(train_dataloader, 0):
            points, label, target, norm_plt = data
            points, label, target = points.cuda(), label.cuda(), target.cuda()

            points_contrast =(PointcloudScaleAndTranslate(PointcloudJitter(PointcloudRotate(points.data.clone()))))

            x_points = points.data.clone()
            _, _, normals_pred, _,_,_,_,saveFeats,_,_,_,_ = encoder(x_points,points,points_contrast)


            train_features.append(saveFeats.data.cpu().numpy())
            train_label.append(target.data)
            train_cate.append(label.data)
        print('Train data ok!')

        train_label = torch.cat(train_label[0:int(len(train_features))], dim=0)
        train_cate = torch.cat(train_cate[0:int(len(train_features))], dim=0)
        train_features = np.concatenate(train_features[0:int(len(train_features))], axis=0)
        


        print('T_F',train_features.shape)
        print('T_L',train_label.size())

        train_file=h5py.File('train_features_saved.h5','w')
        train_file.create_dataset('train_features', data = train_features)
        train_file.create_dataset('train_label', data = train_label.cpu().numpy())
        train_file.create_dataset('train_category', data = train_cate.cpu().numpy())
        train_file.close()


        print('Train_data Saved')
        ii = 0
        points_arr = []
        points_con_arr =[]
        Feats_arr = []
        Feats_con_arr = []
        points_ss_arr = []
        points_gg_arr = []
        target_arr = []
        for j, data in enumerate(test_dataloader, 0):
            points, label, target,norm_plt = data
            points, label, target = points.cuda(), label.cuda(), target.cuda()

            points_contrast =(PointcloudScaleAndTranslate(PointcloudJitter(PointcloudRotate(points.data.clone()))))

            # points = points
            # out, global_feature, saveFeats,_ = encoder(points)
            x_points = points.data.clone()
            _, _, normals_pred, _,_,_,_,saveFeats,points_con,saveFeats_Con,point_ss,point_gg = encoder(x_points,points, points_contrast)
            
            test_label.append(target.data)
            test_cate.append(label.data)
            test_features.append(saveFeats.data.cpu().numpy())



        test_label = torch.cat(test_label, dim=0)


        # train_features = np.concatenate(train_features, dim=0)
        test_features = np.concatenate(test_features, axis=0)
        test_cate = torch.cat(test_cate, dim=0)

        test_file=h5py.File('test_features_saved.h5','w')
        test_file.create_dataset('test_features', data = test_features)
        test_file.create_dataset('test_label', data = test_label.cpu().numpy())
        test_file.create_dataset('test_category', data = test_cate.cpu().numpy())
        test_file.close()
        print('test data saved')





def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def adjust_learning_rate(optimizer, epoch, args):
    step = int(epoch // 20)
    lr = args.base_lr * (0.7 ** step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def validate(train_dataloader, test_dataloader, encoder, args):
    global svm_best_acc40
    encoder.eval()

    test_features = []
    test_label = []

    train_features = []
    train_label = []

    PointcloudRotate = d_utils.PointcloudRotate()

    # feature extraction
    with torch.no_grad():
        for j, data in enumerate(train_dataloader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()

            num_points = 1024

            fps_idx = pointnet2_utils.furthest_point_sample(points, num_points)  # (B, npoint)
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

            

            feature,_,_ = encoder(points, get_feature=True)
            target = target.view(-1)

            train_features.append(feature.data)
            train_label.append(target.data)

        for j, data in enumerate(test_dataloader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()

            fps_idx = pointnet2_utils.furthest_point_sample(points, args.num_points)  # (B, npoint)
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()

            feature,_,_ = encoder(points, get_feature=True)
            target = target.view(-1)
            test_label.append(target.data)
            test_features.append(feature.data)

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

    # train svm
    svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

    if svm_acc > svm_best_acc40:
        svm_best_acc40 = svm_acc

    encoder.train()
    print('Results: svm acc=', svm_acc, 'best svm acc=', svm_best_acc40)
    print(args.name, args.arch)

    return svm_acc


if __name__ == "__main__":
    main()