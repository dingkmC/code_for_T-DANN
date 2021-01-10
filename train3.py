from __future__ import absolute_import, division, print_function, unicode_literals

import torchvision
import torch.optim as optim

import numpy as np
from sklearn.manifold import TSNE

import argparse, sys, os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import time
from collections import Counter
import matplotlib.pyplot as plt 
from utils import *

SEED=334
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

# Training settings
parser = argparse.ArgumentParser(description='PyTorch T-DANN Implementation')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--source', default='2_20140404.npy', type=str,metavar='N',
                    help='training data1')
parser.add_argument('--target', type=str,default='1_20131027.npy', metavar='N',
                    help='training data2')
parser.add_argument('--label1', type=str, default='label_sub.npy',metavar='N',
                    help='training data1')
parser.add_argument('--label2', type=str, default='label_sub.npy', metavar='N',
                    help='training data2')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epoch', type=int, default=200, metavar='N',
                    help='epoch')

# args = parser.parse_args(args=[])
args = parser.parse_args()
print(args)

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Extractor(nn.Module):

    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv1d(5, 64, 5, 1) 
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 5, 1) 
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 5, 1) 
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AvgPool1d(5, stride=5)
#         self.pool = nn.MaxPool1d(5, stride=5)
      

    def forward(self, input):
        x = input.permute(0,2,1)
#         print('after permute ',x.shape)  # 64,5,62
        x = F.relu(self.bn1(self.conv1(x))) 
#         print('after CNN1 ',x.shape)    # 64, 64, 58
        x = F.relu(self.bn2(self.conv2(x)))  
#         print('after CNN2 ',x.shape)     #64, 64, 54
#         x = F.avg_pool1d(x,kernel_size=5) 
        x = F.relu(self.bn3(self.conv3(x))) 
        x = self.pool(x)
        
#         print('before fc ',x.shape)    #64, 64, 10
        x = x.view(-1, 64 * 10) 
    
        return x


class Class_classifier(nn.Module):

    def __init__(self):
        super(Class_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 100)
        # self.bn2 = nn.BatchNorm1d(100)
        # self.fc3 = nn.Linear(100, 10)
        self.fc1 = nn.Linear(64 * 10, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, input):
        # logits = F.relu(self.bn1(self.fc1(input)))
        # logits = self.fc2(F.dropout(logits))
        # logits = F.relu(self.bn2(logits))
        # logits = self.fc3(logits)
        logits = F.relu(self.fc1(input))
        logits = self.fc2(F.dropout(logits))
        logits = F.relu(logits)
        logits = self.fc3(logits)

        return F.log_softmax(logits, 1)

class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        # self.fc1 = nn.Linear(50 * 4 * 4, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, 2)
        self.fc1 = nn.Linear(64 * 10, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input, constant):
        input = GradReverse.grad_reverse(input, constant)
        # logits = F.relu(self.bn1(self.fc1(input)))
        # logits = F.log_softmax(self.fc2(logits), 1)
        logits = F.relu(self.fc1(input))
        logits = F.log_softmax(self.fc2(logits), 1)

        return logits



def train(feature_extractor, class_classifier,class_criterion,
          source_dataloader, optimizer, epoch):
    """
    Execute target domain adaptation
    :param training_mode:
    :param feature_extractor:
    :param class_classifier:
    :param domain_classifier:
    :param class_criterion:
    :param domain_criterion:
    :param source_dataloader:
    :param target_dataloader:
    :param optimizer:
    :return:
    """

    # setup models
    feature_extractor.train()
    class_classifier.train()

    # steps
    start_steps = epoch * len(source_dataloader)
    total_steps = 10 * len(source_dataloader)

    for batch_idx, sdata in enumerate(source_dataloader):
        # setup hyperparameters
        p = float(batch_idx + start_steps) / total_steps
        constant = 2. / (1. + np.exp(-gamma * p)) - 1

        # prepare the data
        input1, label1 = sdata
        input1, label1 = Variable(input1.cuda()), Variable(label1.cuda().long())

        # setup optimizer
        optimizer = optimizer_scheduler(optimizer, p)
        optimizer.zero_grad()

        # compute the output of source domain and target domain
        src_feature = feature_extractor(input1)
        # compute the class loss of src_feature
        class_preds = class_classifier(src_feature)
        class_loss = class_criterion(class_preds, label1)
        class_loss.backward()
        optimizer.step()

        # print loss
        if (batch_idx + 1) % 100 == 0:
            print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                batch_idx * len(input1), len(source_dataloader.dataset),
                batch_idx * len(input1)/len(source_dataloader.dataset),
                class_loss.item()
            ))
            c_loss.append(class_loss.item())
                
def test(feature_extractor, class_classifier, source_dataloader, target_dataloader):
    """
    Test the performance of the model
    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :return: None
    """
    # setup the network
    feature_extractor.eval()
    class_classifier.eval()
    source_correct = 0.0
    target_correct = 0.0

    for batch_idx, sdata in enumerate(source_dataloader):
        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1.

        input1, label1 = sdata
        input1, label1 = Variable(input1.cuda()), Variable(label1.cuda().long())
        src_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())

        output1 = class_classifier(feature_extractor(input1))
        pred1 = output1.data.max(1, keepdim = True)[1]
        source_correct += pred1.eq(label1.data.view_as(pred1)).cpu().sum()

    for batch_idx, tdata in enumerate(target_dataloader):
        # setup hyperparameters
        p = float(batch_idx) / len(source_dataloader)
        constant = 2. / (1. + np.exp(-10 * p)) - 1

        input2, label2 = tdata
        input2, label2 = Variable(input2.cuda()), Variable(label2.cuda().long())
        tgt_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())

        output2 = class_classifier(feature_extractor(input2))
        pred2 = output2.data.max(1, keepdim=True)[1]
        target_correct += pred2.eq(label2.data.view_as(pred2)).cpu().sum()

#     print('\nSource Accuracy: {}/{} ({:.4f}%)\nTarget Accuracy: {}/{} ({:.4f}%)\n'.
#         format(
#         source_correct, len(source_dataloader.dataset), 100. * float(source_correct) / len(source_dataloader.dataset),
#         target_correct, len(target_dataloader.dataset), 100. * float(target_correct) / len(target_dataloader.dataset)
#     ))
    acc_list1.append(100. * float(source_correct) / len(source_dataloader.dataset))
    acc_list2.append(100. * float(target_correct) / len(target_dataloader.dataset))
                
def main():

        # prepare the source data and target data
    data1 = np.load(r'/vm_volum8/3session012/subs/' + args.source)
    data2 = np.load(r'/vm_volum8/3session012/subs/' + args.target)
    label1 = np.load(r'/vm_volum8/3session012/subs/' + args.label1)
    label2 = np.load(r'/vm_volum8/3session012/subs/' + args.label2)
    
#     btt1, btt2 = three2two(label1), three2two(label2)
#     data1, label1 = data1[btt1,:,:], label1[btt1]
#     data2, label2 = data2[btt2,:,:], label2[btt2]

#     for i in range(len(label1)):
#         if label1[i] == 2:
#             label1[i]=1
#     for i in range(len(label2)):
#         if label2[i] == 2:
#             label2[i]=1  
    
    src_train_dataloader = get_train_loader(data1,label1,batch_size=batch_size,shuffle=True)
    src_test_dataloader = get_test_loader(data1,label1,batch_size=batch_size,shuffle=True)
    tgt_train_dataloader = get_train_loader(data2,label2,batch_size=batch_size,shuffle=True)
    tgt_test_dataloader = get_test_loader(data2,label2,batch_size=batch_size,shuffle=True)

    # init models
    feature_extractor = Extractor()
    class_classifier = Class_classifier()

    feature_extractor.cuda()
    class_classifier.cuda()

    # init criterions
    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()

    # init optimizer
    optimizer = optim.SGD([
            {'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()}
    ], lr= 0.01, momentum= 0.9)
    
    optimizer2 = optim.Adam([
            {'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()}
    ], lr= 0.001)

    for epoch in range(200):
        if epoch % 50 == 0:
            print("finish (%) ", 100*epoch/args.epoch)
        train(feature_extractor, class_classifier, class_criterion, src_train_dataloader, optimizer2, epoch)
        test(feature_extractor, class_classifier, src_test_dataloader, tgt_test_dataloader)

        
total_loss, d_loss, c_loss = [],[],[]
acc_list1, acc_list2, acc_list3 = [],[],[]
if __name__ == '__main__':
    gamma = 10
    theta = 1
    batch_size = 128
    time_start=time.time()
    main()
    time_end=time.time()
    print('total run time: (min)',(time_end-time_start)/60.)
    print('max target accuracy: ',max(acc_list2))
