# Framework modified from: https://github.com/jfzhang95/pytorch-video-recognition
import os
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader.dataset import VideoDataset
from models.C3D import C3D
from models.SlowFast import SlowFastResnet50, SlowFastResnet101, SlowFastResnet152, SlowFastResnet200
from models.Resnet3D import Resnet3D_10, Resnet3D_18, Resnet3D_34, Resnet3D_50, Resnet3D_101, Resnet3D_152, Resnet3D_200
from models.Densenet3D import Densenet3D_121, Densenet3D_169, Densenet3D_201, Densenet3D_264
from models.Mobilenet3D_v1 import MobileNet3D_v1
from models.Mobilenet3D_v2 import MobileNet3D_v2
from models.Shufflenet3D_v1 import ShuffleNet3D_v1
from models.Shufflenet3D_v2 import ShuffleNet3D_v2
from models.Squeezenet3D import SqueezeNet3D
from models.Resnext3D import Resnext3D_50, Resnext3D_101, Resnext3D_152
from models.Lenet3D import Lenet3D
from models.WideResnet3D import WideResnet3D_50, WideResnet3D_101, WideResnet3D_152, WideResnet3D_200
from models.R3D import R3D_18, R3D_34
from models.R2Plus1D import R2Plus1D_18, R2Plus1D_34
from models.VGG3D import VGG3D_11, VGG3D_13, VGG3D_16, VGG3D_19
from models.Resnet_I3D import I3D_Resnet50, I3D_Resnet50_NL
from models.Baseline_Nonlocal import Baseline_Nonlocal
from models.ARTNet import ARTNet
from models.LTC import LTC
from models.P3D import P3D
from models.Fast_S3D import Fast_S3D
from models.S3D_G import S3D_G
from models.I3D import I3D
from models.FstCN import FstCN
from models.LRCN import LRCN
from models.FFC_C3D import FFC_C3D
from models.MnasNet import MnasNet
from models.Baseline_Spectral_Norm import Baseline_Spectral_Norm
from models.I3D_pretrained import InceptionI3d
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Video action recognition training")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--test_interval', type=int, default=20)
parser.add_argument('--snapshot', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='ucf101')
parser.add_argument('--model_name', type=str, default='C3D', choices=['C3D', 'SlowFast', 'Resnet3D', 'Densenet3D', 'Mobilenet3D_v1', 'Mobilenet3D_v2',
                                                                                    'Shufflenet3D_v1', 'Shufflenet3D_v2', 'Squeezenet3D', 'Resnext3D', 'Lenet3D',
                                                                                    'WideResnet3D', 'R3D', 'R2Plus1D', 'VGG3D', 'Resnet_I3D', 'Baseline_Nonlocal',
                                                                                    'ARTNet', 'LTC', 'P3D', 'S3D', 'I3D', 'FstCN', 'LRCN', 'FFC_C3D', 'Mnasnet',
                                                                                    'Baseline_Spectral_Norm', 'I3D_pretrained'])
parser.add_argument('--slowfast_version', type=str, default='resnet50', choices=['resnet50', 'resnet101', 'resnet152', 'resnet200'])
parser.add_argument('--resnet3d_version', type=str, default='resnet50', choices=['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200'])
parser.add_argument('--densenet3d_version', type=str, default='dense121', choices=['dense121', 'dense169', 'dense201', 'dense264'])
parser.add_argument('--resnext3d_version', type=str, default='resnext50', choices=['resnext50', 'resnext101', 'resnext152'])
parser.add_argument('--wide_resnet3d_version', type=str, default='wide50', choices=['wide50', 'wide101', 'wide152', 'wide200'])
parser.add_argument('--r3d_version', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
parser.add_argument('--r2plus1d_version', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
parser.add_argument('--vgg3d_version', type=str, default='vgg19', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])
parser.add_argument('--resnet_i3d_version', type=str, default='resnet50_nl', choices=['resnet50', 'resnet50_nl'])
parser.add_argument('--s3d_version', type=str, default='fast_s3d', choices=['fast_s3d', 's3d_g'])
parser.add_argument('--width_shufflenetv1', type=float, default='1.0', choices=[0.25, 0.5, 1.0, 1.5, 2.0])
parser.add_argument('--width_shufflenetv2', type=float, default='1.0', choices=[0.25, 0.5, 1.0, 1.5, 2.0])
parser.add_argument('--width_mobilenetv1', type=float, default='1.0', choices=[0.5, 1.0, 1.5, 2.0])
parser.add_argument('--width_mobilenetv2', type=float, default='1.0', choices=[0.2, 0.7, 0.45, 1.0])
parser.add_argument('--type_pretrained_i3d', type=str, default='imagenet', choices=['imagenet', 'charades'])
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--step_size', type=int, default=10, help='step size for scheduler')
args = parser.parse_args()

### calculate top 1 acc (https://github.com/pytorch/examples/blob/main/imagenet/main.py) ###
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res
### calculate top 1 acc ###

if args.dataset == 'ucf101':
    num_classes = 101
else:
    print('This dataset does not exist.')
    raise NotImplementedError

saveName = args.model_name + '-' + args.dataset
if not os.path.exists('save_model'):
    os.makedirs('save_model')

def main():
    if args.model_name == 'C3D':
        model = C3D(num_classes=num_classes, pretrained='pretrained/pretrained_c3d/c3d-pretrained.pth')
    elif args.model_name == 'SlowFast':
        if args.slowfast_version == 'resnet50':
            model = SlowFastResnet50(class_num=num_classes)
        elif args.slowfast_version == 'resnet101':
            model = SlowFastResnet101(class_num=num_classes)
        elif args.slowfast_version == 'resnet152':
            model = SlowFastResnet152(class_num=num_classes)
        elif args.slowfast_version == 'resnet200':
            model = SlowFastResnet200(class_num=num_classes)
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Resnet3D':
        if args.resnet3d_version == 'resnet10':
            model = Resnet3D_10(n_classes=num_classes, pretrained=None, type_resnet='resnet10')
        elif args.resnet3d_version == 'resnet18':
            model = Resnet3D_18(n_classes=num_classes, pretrained='pretrained/pretrained_resnet3D/resnet-18-kinetics.pth', type_resnet='resnet18')
        elif args.resnet3d_version == 'resnet34':
            model = Resnet3D_34(n_classes=num_classes, pretrained='pretrained/pretrained_resnet3D/resnet-34-kinetics.pth', type_resnet='resnet34')
        elif args.resnet3d_version == 'resnet50':
            model = Resnet3D_50(n_classes=num_classes, pretrained='pretrained/pretrained_resnet3D/resnet-50-kinetics.pth', type_resnet='resnet50')
        elif args.resnet3d_version == 'resnet101':
            model = Resnet3D_101(n_classes=num_classes, pretrained='pretrained/pretrained_resnet3D/resnet-101-kinetics.pth', type_resnet='resnet101')
        elif args.resnet3d_version == 'resnet152':
            model = Resnet3D_152(n_classes=num_classes, pretrained=None, type_resnet='resnet152')
        elif args.resnet3d_version == 'resnet200':
            model = Resnet3D_200(n_classes=num_classes, pretrained=None, type_resnet='resnet200')
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Densenet3D':
        if args.densenet3d_version == 'dense121':
            model = Densenet3D_121(num_classes=num_classes)
        elif args.densenet3d_version == 'dense169':
            model = Densenet3D_169(num_classes=num_classes)
        elif args.densenet3d_version == 'dense201':
            model = Densenet3D_201(num_classes=num_classes)
        elif args.densenet3d_version == 'dense264':
            model = Densenet3D_264(num_classes=num_classes)
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Mobilenet3D_v1':
        if args.width_mobilenetv1 == 0.5:
            model = MobileNet3D_v1(num_classes=num_classes, width_mult=0.5, pretrained='pretrained/pretrained_mobilenet3D_v1/kinetics_mobilenet_0.5x_RGB_16_best.pth')
        elif args.width_mobilenetv1 == 1.0:
            model = MobileNet3D_v1(num_classes=num_classes, width_mult=1.0, pretrained='pretrained/pretrained_mobilenet3D_v1/kinetics_mobilenet_1.0x_RGB_16_best.pth')
        elif args.width_mobilenetv1 == 1.5:
            model = MobileNet3D_v1(num_classes=num_classes, width_mult=1.5, pretrained='pretrained/pretrained_mobilenet3D_v1/kinetics_mobilenet_1.5x_RGB_16_best.pth')
        elif args.width_mobilenetv1 == 2.0:
            model = MobileNet3D_v1(num_classes=num_classes, width_mult=2.0, pretrained='pretrained/pretrained_mobilenet3D_v1/kinetics_mobilenet_2.0x_RGB_16_best.pth')
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Mobilenet3D_v2':
        if args.width_mobilenetv2 == 0.2:
            model = MobileNet3D_v2(num_classes=num_classes, width_mult=0.2, pretrained='pretrained/pretrained_mobilenet3D_v2/kinetics_mobilenetv2_0.2x_RGB_16_best.pth')
        elif args.width_mobilenetv2 == 0.7:
            model = MobileNet3D_v2(num_classes=num_classes, width_mult=0.7, pretrained='pretrained/pretrained_mobilenet3D_v2/kinetics_mobilenetv2_0.7x_RGB_16_best.pth')
        elif args.width_mobilenetv2 == 0.45:
            model = MobileNet3D_v2(num_classes=num_classes, width_mult=0.45, pretrained='pretrained/pretrained_mobilenet3D_v2/kinetics_mobilenetv2_0.45x_RGB_16_best.pth')
        elif args.width_mobilenetv2 == 1.0:
            model = MobileNet3D_v2(num_classes=num_classes, width_mult=1.0, pretrained='pretrained/pretrained_mobilenet3D_v2/kinetics_mobilenetv2_1.0x_RGB_16_best.pth')
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Shufflenet3D_v1':
        if args.width_shufflenetv1 == 0.25:
            model = ShuffleNet3D_v1(groups=3, num_classes=num_classes, width_mult=0.25, pretrained=None)
        elif args.width_shufflenetv1 == 0.5:
            model = ShuffleNet3D_v1(groups=3, num_classes=num_classes, width_mult=0.5, pretrained='pretrained/pretrained_shufflenet3D_v1/kinetics_shufflenet_0.5x_G3_RGB_16_best.pth')
        elif args.width_shufflenetv1 == 1.0:
            model = ShuffleNet3D_v1(groups=3, num_classes=num_classes, width_mult=1.0, pretrained='pretrained/pretrained_shufflenet3D_v1/kinetics_shufflenet_1.0x_G3_RGB_16_best.pth')
        elif args.width_shufflenetv1 == 1.5:
            model = ShuffleNet3D_v1(groups=3, num_classes=num_classes, width_mult=1.5, pretrained='pretrained/pretrained_shufflenet3D_v1/kinetics_shufflenet_1.5x_G3_RGB_16_best.pth')
        elif args.width_shufflenetv1 == 2.0:
            model = ShuffleNet3D_v1(groups=3, num_classes=num_classes, width_mult=2.0, pretrained='pretrained/pretrained_shufflenet3D_v1/kinetics_shufflenet_2.0x_G3_RGB_16_best.pth')
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Shufflenet3D_v2':
        if args.width_shufflenetv2 == 0.25:
            model = ShuffleNet3D_v2(num_classes=num_classes, width_mult=0.25, pretrained='pretrained/pretrained_shufflenet3D_v2/kinetics_shufflenetv2_0.25x_RGB_16_best.pth')
        elif args.width_shufflenetv2 == 0.5:
            model = ShuffleNet3D_v2(num_classes=num_classes, width_mult=0.5, pretrained=None)
        elif args.width_shufflenetv2 == 1.0:
            model = ShuffleNet3D_v2(num_classes=num_classes, width_mult=1.0, pretrained='pretrained/pretrained_shufflenet3D_v2/kinetics_shufflenetv2_1.0x_RGB_16_best.pth')
        elif args.width_shufflenetv2 == 1.5:
            model = ShuffleNet3D_v2(num_classes=num_classes, width_mult=1.5, pretrained='pretrained/pretrained_shufflenet3D_v2/kinetics_shufflenetv2_1.5x_RGB_16_best.pth')
        elif args.width_shufflenetv2 == 2.0:
            model = ShuffleNet3D_v2(num_classes=num_classes, width_mult=2.0, pretrained='pretrained/pretrained_shufflenet3D_v2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth')
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Squeezenet3D':
        model = SqueezeNet3D(version=1.1, sample_size=112, sample_duration=16, num_classes=num_classes, pretrained='pretrained/pretrained_squeezenet3D/kinetics_squeezenet_RGB_16_best.pth')
    elif args.model_name == 'Resnext3D':
        if args.resnext3d_version == 'resnext50':
            model = Resnext3D_50(sample_size=112, sample_duration=16, num_classes=num_classes, pretrained=None, type_resnext='resnext50')
        elif args.resnext3d_version == 'resnext101':
            model = Resnext3D_101(sample_size=112, sample_duration=16, num_classes=num_classes, pretrained='pretrained/pretrained_resnext3D/kinetics_resnext_101_RGB_16_best.pth', type_resnext='resnext101')
        elif args.resnext3d_version == 'resnext152':
            model = Resnext3D_152(sample_size=112, sample_duration=16, num_classes=num_classes, pretrained=None, type_resnext='resnext152')
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Lenet3D':
        model = Lenet3D(num_classes=num_classes)
    elif args.model_name == 'WideResnet3D':
        if args.wide_resnet3d_version == 'wide50':
            model = WideResnet3D_50(sample_size=112, sample_duration=16, num_classes=num_classes)
        elif args.wide_resnet3d_version == 'wide101':
            model = WideResnet3D_101(sample_size=112, sample_duration=16, num_classes=num_classes)
        elif args.wide_resnet3d_version == 'wide152':
            model = WideResnet3D_152(sample_size=112, sample_duration=16, num_classes=num_classes)
        elif args.wide_resnet3d_version == 'wide200':
            model = WideResnet3D_200(sample_size=112, sample_duration=16, num_classes=num_classes)
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'R3D':
        if args.r3d_version == 'resnet18':
            model = R3D_18(num_classes=num_classes)
        elif args.r3d_version == 'resnet34':
            model = R3D_34(num_classes=num_classes)
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'R2Plus1D':
        if args.r2plus1d_version == 'resnet18':
            model = R2Plus1D_18(num_classes=num_classes)
        elif args.r2plus1d_version == 'resnet34':
            model = R2Plus1D_34(num_classes=num_classes)
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'VGG3D':
        if args.vgg3d_version == 'vgg11':
            model = VGG3D_11(num_classes=num_classes)
        elif args.vgg3d_version == 'vgg13':
            model = VGG3D_13(num_classes=num_classes)
        elif args.vgg3d_version == 'vgg16':
            model = VGG3D_16(num_classes=num_classes)
        elif args.vgg3d_version == 'vgg19':
            model = VGG3D_19(num_classes=num_classes)
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Resnet_I3D':
        if args.resnet_i3d_version == 'resnet50':
            model = I3D_Resnet50(num_classes=num_classes)
        elif args.resnet_i3d_version == 'resnet50_nl':
            model = I3D_Resnet50_NL(num_classes=num_classes)
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'Baseline_Nonlocal':
        model = Baseline_Nonlocal(num_classes=num_classes, pretrained='pretrained/pretrained_c3d/c3d-pretrained.pth')
    elif args.model_name == 'ARTNet':
        model = ARTNet(num_classes=num_classes)
    elif args.model_name == 'LTC':
        model = LTC(num_classes=num_classes)
    elif args.model_name == 'P3D':
        model = P3D(num_classes=num_classes)
    elif args.model_name == 'S3D':
        if args.s3d_version == 'fast_s3d':
            model = Fast_S3D(num_classes=num_classes)
        elif args.s3d_version == 's3d_g':
            model = S3D_G(num_classes=num_classes)
        else:
            print('This version does not exist.')
            raise NotImplementedError
    elif args.model_name == 'I3D':
        model = I3D(num_classes=num_classes)
    elif args.model_name == 'FstCN':
        model = FstCN(num_classes=num_classes)
    elif args.model_name == 'LRCN':
        model = LRCN(num_classes=num_classes)
    elif args.model_name == 'FFC_C3D':
        model = FFC_C3D(num_classes=num_classes)
    elif args.model_name == 'Mnasnet':
        model = MnasNet(num_classes=num_classes)
    elif args.model_name == 'Baseline_Spectral_Norm':
        model = Baseline_Spectral_Norm(num_classes=num_classes)
    elif args.model_name == 'I3D_pretrained':
        if args.type_pretrained_i3d == 'imagenet':
            print('Training with imagenet pretrained weights.')
            model = InceptionI3d(num_classes=400)
            model.load_state_dict(torch.load('pretrained/pretrained_i3d/rgb_imagenet.pt'))
        elif args.type_pretrained_i3d == 'charades':
            print('Training with charades pretrained weights.')
            model = InceptionI3d(num_classes=157)
            model.load_state_dict(torch.load('pretrained/pretrained_i3d/rgb_charades.pt'))
        else:
            print('This pretrained weight does not exist.')
            raise NotImplementedError
        model.replace_logits(num_classes=num_classes)
    else:
        print('This model does not exist.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    model.cuda() # avoid cpu and cuda conflict when loading checkpoint for optimizer
    if args.resume_epoch == 0:
        print("Training {} from scratch...".format(args.model_name))
    else:
        checkpoint = torch.load(os.path.join('save_model', saveName + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar'), map_location=lambda storage, loc: storage)
        print('Initializing weights from {}...'.format(os.path.join('save_model', saveName + '_epoch-' + str(args.resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
    print('Total parameters: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # model.cuda()
    criterion.cuda()
    print('Training model on {} dataset...'.format(args.dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=args.dataset, split='train', clip_len=16), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(VideoDataset(dataset=args.dataset, split='val', clip_len=16), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = DataLoader(VideoDataset(dataset=args.dataset, split='test', clip_len=16), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    ### eval top 1 acc ###
    top1_train = AverageMeter()
    top5_train = AverageMeter()
    top1_val = AverageMeter()
    top5_val = AverageMeter()
    top1_test = AverageMeter()
    top5_test = AverageMeter()
    ### eval top 1 acc ###

    for epoch in range(args.resume_epoch, args.num_epochs):
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0.0
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs = Variable(inputs, requires_grad=True).cuda()
                labels = Variable(labels).cuda()
                optimizer.zero_grad()
                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)
                ### eval top 1 acc ###
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                if phase == 'train':
                    top1_train.update(prec1.item(), inputs.size(0))
                    top5_train.update(prec5.item(), inputs.size(0))
                else:
                    top1_val.update(prec1.item(), inputs.size(0))
                    top5_val.update(prec5.item(), inputs.size(0))
                ### eval top 1 acc ###
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]
            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, args.num_epochs, epoch_loss, epoch_acc))
            ### eval top 1 acc ###
            if phase == 'train':
                print('[train] Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(top1_acc=top1_train.avg, top5_acc=top5_train.avg))
            else:
                print('[val] Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(top1_acc=top1_val.avg, top5_acc=top5_val.avg))
            ### eval top 1 acc ###

        # Save model every 10 epochs
        if epoch % args.snapshot == args.snapshot - 1:
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()},
                       os.path.join('save_model', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print('Saved model at {}\n'.format(os.path.join('save_model', saveName + '_epoch-' + str(epoch) + '.pth.tar')))
        # Eval on test dataset every 20 epochs
        if epoch % args.test_interval == args.test_interval - 1:
            model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.cuda()
                labels = labels.cuda()
                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)
                ### eval top 1 acc ###
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                top1_test.update(prec1.item(), inputs.size(0))
                top5_test.update(prec5.item(), inputs.size(0))
                ### eval top 1 acc ###
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size
            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, args.num_epochs, epoch_loss, epoch_acc))
            ### eval top 1 acc ###
            print('[test] Top-1 accuracy: {top1_acc:.2f}%, Top-5 accuracy: {top5_acc:.2f}%'.format(top1_acc=top1_test.avg, top5_acc=top5_test.avg))
            ### eval top 1 acc ###

if __name__ == "__main__":
    main()
