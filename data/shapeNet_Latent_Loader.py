
import os
import sys
import h5py
import torch
import numpy as np
import torch.utils.data as data


cls_shapenet = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife',
                'Lamp', 'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 
                'Table']


def sample_by_category(point, label, cate, sampling_rate):
    cate_num = np.zeros(16)
    for i in range(label.shape[0]):
        cate_num[int(cate[i])] += 1
    choose_id = []
    # print(cate_num)
    arr = np.arange(label.shape[0])
    np.random.shuffle(arr)
    cate_count = np.zeros(16)
    for i in range(label.shape[0]):
        item_id = arr[i]
        item_cate = int(cate[item_id])
        if cate_count[item_cate] < cate_num[item_cate] * sampling_rate or cate_count[item_cate] == 0:
            choose_id.append(item_id)
            cate_count[item_cate] += 1

    # print(cate_count)
    # exit(0)
    return point[choose_id],label[choose_id],cate[choose_id] 

class shapeNet_Latent_Loader(data.Dataset):
    def __init__(self,transforms, root, self_supervision=False, train=True, sampling_rate=0.01):
        super(shapeNet_Latent_Loader, self).__init__()
        self.root = root
        self.is_training = train
        self.self_supervision = self_supervision
        self.transforms = transforms
        if self.is_training:
            self.files =  'train_features_saved.h5'
        else:
            self.files = 'test_features_saved.h5'
        self.point, self.label, self.cate = \
                    self.load_all_data_seg(self.files)
        if self.is_training:
            self.point, self.label, self.cate = sample_by_category(self.point, self.label, self.cate, sampling_rate)

        print('Load Point:',self.point.shape)
        print('Load Label:',self.label.shape)
        print('Load Cate:',self.cate.shape)

    def load_all_data_cls(self, files):
        print('---> loading all data in to memory')
        data = np.array([self.load_h5(f) for f in files])
        point = np.concatenate(data[:, 0], axis=0)
        label = np.concatenate(data[:, 1], axis=0)
        return point, label

    def load_all_data_seg(self, files):
        print('---> loading all data in to memory')
        data = (self.load_h5_seg(files))
        point, label, cate = data
        point = np.array(point)
        label = np.array(label)
        cate  = np.array(cate)
        return point, label, cate

    def __getitem__(self, index):
        point = self.point[index]
        label = self.label[index]
        cate  = self.cate[index]
        if self.transforms is not None:
            point = self.transforms(self.point[index])
            # point = point.permute(1, 0).contiguous()
            label = self.transforms(label)
            cate  = self.transforms(cate)
        # label = torch.LongTensor({self.label[index]})

        # print('point',point.size())
        # print('segm_now',segm[0])
        # print('segm_org',self.segm[index,0])
        # print('segmType',self.segm[index].shape)
        # print('&&',index)
        # if self.self_supervision == True:
        #     return point

        return point, label, cate

    def __len__(self):
        return self.label.shape[0]

    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        label = f['label'][:]
        return data, label

    def load_h5_seg(self, h5_filename):
        f = h5py.File(h5_filename)
        if self.is_training == True:
            data = f['train_features'][:]
            label = f['train_label'][:]
            cate = f['train_category'][:]
        else:
            data = f['test_features'][:]
            label = f['test_label'][:]
            cate = f['test_category'][:]
        return data, label, cate

    def get_files(self, fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
            files = [os.path.join(self.root, line.strip()) \
                        for line in open(fn)]
        return files


def get_dataloader(transforms,root, network, batch_size, num_workers, train,self_supervision, sampling_rate=0.01):
    """
    Get dataloader accroding to network (cls or seg)
    """
    if network == 'cls':
        pass
    elif network == 'seg':
        dataset = shapeNet_Latent_Loader(transforms = transforms,root=root, train=train, self_supervision=self_supervision, sampling_rate=sampling_rate)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, \
                         shuffle=False, num_workers=num_workers)
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