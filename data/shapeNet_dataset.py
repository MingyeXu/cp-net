
import os
import sys
import h5py
import torch
import numpy as np
import torch.utils.data as data


cls_shapenet = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife',
                'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 
                'Table']


class ShapeNet(data.Dataset):
    def __init__(self,transforms, root, self_supervision=False, train=True):
        super(ShapeNet, self).__init__()
        self.root = root
        self.is_training = train
        self.self_supervision = self_supervision
        self.transforms = transforms
        if self.is_training:
            self.files = self.get_files(os.path.join(root, 'train_hdf5_file_list.txt'))
        else:
            self.files = self.get_files(os.path.join(root, 'test_hdf5_file_list.txt'))
        self.point, self.label, self.segm = \
                    self.load_all_data_seg(self.files)

    def load_all_data_cls(self, files):
        print('---> loading all data in to memory')
        data = np.array([self.load_h5(f) for f in files])
        point = np.concatenate(data[:, 0], axis=0)
        label = np.concatenate(data[:, 1], axis=0)
        return point, label

    def load_all_data_seg(self, files):
        print('---> loading all data in to memory')
        data = np.array([self.load_h5_seg(f) for f in files])
        point = np.concatenate(data[:, 0], axis=0)
        label = np.concatenate(data[:, 1], axis=0)
        segm  = np.concatenate(data[:, 2], axis=0)
        return point, label, segm

    def __getitem__(self, index):
        point = self.point[index]
        label = self.label[index]
        segm  = self.segm[index]
        if self.transforms is not None:
            point = self.transforms(self.point[index])
            # point = point.permute(1, 0).contiguous()
            label = self.transforms(label)
            segm  = self.transforms(segm)
        # label = torch.LongTensor({self.label[index]})

        # print('point',point.size())
        # print('segm_now',segm[0])
        # print('segm_org',self.segm[index,0])
        # print('segmType',self.segm[index].shape)
        # print('&&',index)
        if self.self_supervision == True:
            return point



        return point, label, segm

    def __len__(self):
        return self.label.shape[0]

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return data, label

    def load_h5_seg(self, h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        seg = f['pid'][:]
        return data, label, seg

    def get_files(self, fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
            files = [os.path.join(self.root, line.strip()) \
                        for line in open(fn)]
        return files


def get_dataloader(transforms,root, network, batch_size, num_workers, train,self_supervision):
    """
    Get dataloader accroding to network (cls or seg)
    """
    if network == 'cls':
        pass
    elif network == 'seg':
        dataset = ShapeNet(transforms = transforms,root=root, train=train, self_supervision=self_supervision)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, \
                         shuffle=True, num_workers=num_workers)
    return dataloader, dataset


if __name__ == "__main__":
    root = '/home/zzp/pytorch/GCN/pytorch_workspace/sgm/data/hdf5_data'
    dataloader = get_dataloader(root=root, network='seg', batch_size=32, num_workers=2,train=True)

    print(len(dataloader))

    for idx, data in enumerate(dataloader):
        point, label, segm = data
        print(point.size())
        print(label.size())
        print(segm.size())
        break