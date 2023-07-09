import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

USE_CUDA = True

class SegNet(nn.Module):    
    def __init__(self, latent_caps_size, latent_vec_size ,num_classes):
        super(SegNet, self).__init__()
        self.num_classes=num_classes
        self.latent_caps_size=latent_caps_size
        # self.seg_convs= nn.Conv1d(latent_vec_size+16, num_classes, 1)   
        self.seg_convs1 = nn.Sequential(
                nn.Conv1d(latent_vec_size-3+16, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True) )
        self.seg_convs2 = nn.Sequential(
                nn.Conv1d(1024+16, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True))
        self.seg_convs3 = nn.Sequential(
                nn.Conv1d(512+16, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),  
                nn.Conv1d(256, num_classes, 1),
            )
        # self.seg_convs= nn.Conv1d(latent_vec_size, num_classes, 1)   
        self.linear1 = nn.Linear(2048, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 16)



    def forward(self, data, cate):
        batchsize= data.size(0)
        num_points= data.size(2)
        cate_ = cate.unsqueeze(-1).expand(-1, -1, 2048)

        cate = cate.view(batchsize, 4, 4)
        cate = cate.sum(-1)
        # print(cate.size())

        cate = cate.unsqueeze(-2).expand(-1, 128, -1)
        cate = cate.unsqueeze(-1).expand(-1, -1,-1, 2048) # B, 32,16,N
        
        # 
        data_ = data[:,:-3,:].view(batchsize,128,4,num_points)
        data = data[:,:-3,:] + (data_*cate).view(batchsize,-1,num_points)

        data = torch.cat([data[:,:,:],cate_],dim = 1)
        data = self.seg_convs1(data)

        # x1 = F.adaptive_max_pool1d(data, 1).view(batchsize, -1)
        # x2 = F.adaptive_avg_pool1d(data, 1).view(batchsize, -1)
        # x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)

        data = torch.cat([data,cate_],dim = 1)
        data = self.seg_convs2(data)
        data = torch.cat([data,cate_],dim = 1)
        output = self.seg_convs3(data)

        output = output.transpose(2,1).contiguous()
        output = F.log_softmax(output.view(-1,self.num_classes), dim=-1)
        output = output.view(batchsize, self.latent_caps_size, self.num_classes)
        return output, None
    
if __name__ == '__main__':
    USE_CUDA = True
   