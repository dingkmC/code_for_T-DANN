from __future__ import print_function
import argparse

import torch.utils.data as data
from PIL import Image
import numpy as np
import torch.utils.data
# import torchnet as tnt
from builtins import object
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--F', type=int, default=3, metavar='N',
                    help='repeatation of discriminator')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=300, metavar='N',
                    help='how many epochs')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--save_model', action='store_true', default=True,
                    help='save_model or not')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
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

# class GradReverse(torch.autograd.Function):
#     def __init__(self, lambd):
#         self.lambd = lambd

#     def forward(self, x):
#         return x.view_as(x)

#     def backward(self, grad_output):
#         return (grad_output * -self.lambd)


# def grad_reverse(x, lambd=1.0):
#     return GradReverse(lambd)(x)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
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


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
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

        return logits
    
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
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
    
    
# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64, learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10):
        self.batch_size = 128
        self.num_k = 1
        self.F = args.F

        print(args.source)
        data1= np.load(r'/vm_volum8/3session012/subs/' + args.source)
        data2= np.load(r'/vm_volum8/3session012/subs/' + args.target)
        label1 = np.load(r'/vm_volum8/3session012/subs/' + args.label1)
        label2 = np.load(r'/vm_volum8/3session012/subs/' + args.label2)
        
#         btt1, btt2 = three2two(label1), three2two(label2)
#         data1, label1 = data1[btt1,:,:], label1[btt1]
#         data2, label2 = data2[btt2,:,:], label2[btt2]

#         for i in range(len(label1)):
#             if label1[i] == 2:
#                 label1[i]=1
#         for i in range(len(label2)):
#             if label2[i] == 2:
#                 label2[i]=1  
                
        
        self.data1 = data1
        self.data2 = data2
        self.label1 = label1
        self.label2 = label2

        print('dataset loading')
        self.datasets, self.dataset_test = get_train_loader(self.data1,self.label1,self.batch_size),get_train_loader(self.data2,self.label2,self.batch_size)
        print('load finished!')
        self.G = Generator()
        self.D = Discriminator()
        self.C1 = Classifier()
        self.C2 = Classifier()
        
        self.G.cuda()
        self.D.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam([
                {'params':self.C1.parameters()},
                {'params':self.D.parameters()}],
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam([
                {'params':self.C2.parameters()},
                {'params':self.D.parameters()}],
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train(self, epoch):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.D.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, (sdata,tdata) in enumerate(zip(self.datasets, self.dataset_test)):
            img_t,_ = tdata
            img_s,label_s = sdata

            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            imgs = Variable(torch.cat((img_s, \
                                       img_t), 0))
            label_s = Variable(label_s.long().cuda())

            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            feat_t = self.G(img_t)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_c = loss_s1 + loss_s2
            loss_c.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()
            
            for p in range(self.F):
                feat_s = self.G(img_s)
                feat_t = self.G(img_t)
                source_labels = Variable(torch.zeros((img_s.size()[0])).type(torch.LongTensor).cuda())
                target_labels = Variable(torch.ones((img_t.size()[0])).type(torch.LongTensor).cuda())
                tgt_preds = self.D(feat_s, 1.0)
                src_preds = self.D(feat_t, 1.0)
                tgt_loss = nn.NLLLoss()(tgt_preds, target_labels)
                src_loss = nn.NLLLoss()(src_preds, source_labels)
                domain_loss = tgt_loss + src_loss
                domain_loss.backward()
                self.opt_g.step()
                self.opt_c1.step()
                self.opt_c2.step()
                self.reset_grad()
        

            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            feat_t = self.G(img_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_dis = self.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            for i in range(self.num_k):
                #
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
            if batch_idx > 500:
                return batch_idx

#             if batch_idx % self.interval == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
#                     epoch, batch_idx, 100,
#                     100. * batch_idx / 70000, loss_s1.item(), loss_s2.item(), loss_dis.item()))
                
            c1_loss.append(loss_s1.item())
            c2_loss.append(loss_s2.item()) 
            d_loss.append(loss_dis.item())
            
        return batch_idx

    def test(self, epoch, save_model=False):
        self.G.eval()
        self.D.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0
    
        tgt_correct = 0.0
        src_correct = 0.0
        domain_correct = 0.0
        
        for batch_idx, (sdata,tdata) in enumerate(zip(self.datasets, self.dataset_test)):
            img,label = tdata
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img), Variable(label)
            feat = self.G(img)
            output1 = self.C1(feat)
            output2 = self.C2(feat)
            test_loss += F.nll_loss(output1, label).item()
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred2.eq(label.data).cpu().sum()
            correct3 += pred_ensemble.eq(label.data).cpu().sum()
            size += k
        
            img_s,label_s = sdata
            img_s, label_s = img_s.cuda(), label_s.long().cuda()
            img_s, label_s = Variable(img_s), Variable(label_s)
            feat_s = self.G(img_s)
            
            source_labels = Variable(torch.zeros((img_s.size()[0])).type(torch.LongTensor).cuda())
            target_labels = Variable(torch.ones((img.size()[0])).type(torch.LongTensor).cuda())
            tgt_preds = self.D(feat, 1.0)
            src_preds = self.D(feat_s, 1.0)
            src_preds = src_preds.data.max(1, keepdim= True)[1]
            src_correct += src_preds.eq(source_labels.data.view_as(src_preds)).cpu().sum()
            tgt_preds = tgt_preds.data.max(1, keepdim= True)[1]
            tgt_correct += tgt_preds.eq(target_labels.data.view_as(tgt_preds)).cpu().sum()
            domain_correct = tgt_correct + src_correct
#         print("domain accuracy: ",100. * float(domain_correct) / (len(self.datasets.dataset) + len(self.dataset_test.dataset)) )
#         acc_list4.append(100. * float(domain_correct) / (len(self.datasets.dataset) + len(self.dataset_test.dataset)) )
#         test_loss = test_loss / size   
            
#         print(
#             '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
#                 test_loss, correct1, size,
#                 100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size))
        
        acc_list1.append(100. * correct1 / size)
        acc_list2.append(100. * correct2 / size)
        acc_list3.append(100. * correct3 / size)
        


def main():
    # if not args.one_step:

    solver = Solver(args, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, num_k=args.num_k, all_use=args.all_use,
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch)

    if args.eval_only:
        solver.test(0)
    else:
        count = 0
        for t in range(args.max_epoch):
            if not args.one_step:
                num = solver.train(t)
            else:
                num = solver.train_onestep(t)
            count += num
            if t % 1 == 0:
                solver.test(t, save_model=args.save_model)


c1_loss, c2_loss, d_loss = [],[],[]
acc_list1, acc_list2, acc_list3, acc_list4 = [],[],[],[]
if __name__ == '__main__':
    main()
    print("max accuracy is ",max(acc_list3))
#     plt.plot(range(len(acc_list1)),acc_list1,c='r',label='C1_acc')
#     plt.plot(acc_list2,c='b',label='C2_acc')
#     plt.plot(acc_list3,c='y',label='emsemble_acc')
#     plt.plot(acc_list4,c='g',label='domain_acc')
#     plt.axhline(max(acc_list3),c='b',linestyle='--')
#     plt.title('target domain: session1')
#     plt.legend(loc='best')
#     plt.show()
