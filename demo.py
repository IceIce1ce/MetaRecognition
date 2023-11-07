import torch
import numpy as np
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
import cv2
import argparse
import time
torch.backends.cudnn.benchmark = True # just for inference

parser = argparse.ArgumentParser(description="Video action recognition demo")
parser.add_argument('--dataset', type=str, default='ucf101')
parser.add_argument('--infer_file', type=str, default='save_model/C3D-ucf101_epoch-9.pth.tar')
parser.add_argument('--video_path', type=str, default='demo.mp4')
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
args = parser.parse_args()

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def main():
    if args.dataset == 'ucf101':
        num_classes = 101
        with open('dataloader/ucf_labels.txt', 'r') as f:
            class_names = f.readlines()
            f.close()
        if args.model_name == 'C3D':
            model = C3D(num_classes=num_classes, pretrained='pretrained/pretrained_c3d/c3d-pretrained.pth')
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
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
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
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
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
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
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
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
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'Mobilenet3D_v2':
            if args.width_mobilenetv2 == 0.2:
                model = MobileNet3D_v2(num_classes=101, width_mult=0.2, pretrained='pretrained/pretrained_mobilenet3D_v2/kinetics_mobilenetv2_0.2x_RGB_16_best.pth')
            elif args.width_mobilenetv2 == 0.7:
                model = MobileNet3D_v2(num_classes=101, width_mult=0.7, pretrained='pretrained/pretrained_mobilenet3D_v2/kinetics_mobilenetv2_0.7x_RGB_16_best.pth')
            elif args.width_mobilenetv2 == 0.45:
                model = MobileNet3D_v2(num_classes=101, width_mult=0.45, pretrained='pretrained/pretrained_mobilenet3D_v2/kinetics_mobilenetv2_0.45x_RGB_16_best.pth')
            elif args.width_mobilenetv2 == 1.0:
                model = MobileNet3D_v2(num_classes=101, width_mult=1.0, pretrained='pretrained/pretrained_mobilenet3D_v2/kinetics_mobilenetv2_1.0x_RGB_16_best.pth')
            else:
                print('This version does not exist.')
                raise NotImplementedError
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
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
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
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
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'Squeezenet3D':
            model = SqueezeNet3D(version=1.1, sample_size=112, sample_duration=16, num_classes=num_classes, pretrained='pretrained/pretrained_squeezenet3D/kinetics_squeezenet_RGB_16_best.pth')
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
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
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'Lenet3D':
            model = Lenet3D(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
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
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'R3D':
            if args.r3d_version == 'resnet18':
                model = R3D_18(num_classes=num_classes)
            elif args.r3d_version == 'resnet34':
                model = R3D_34(num_classes=num_classes)
            else:
                print('This version does not exist.')
                raise NotImplementedError
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'R2Plus1D':
            if args.r2plus1d_version == 'resnet18':
                model = R2Plus1D_18(num_classes=num_classes)
            elif args.r2plus1d_version == 'resnet34':
                model = R2Plus1D_34(num_classes=num_classes)
            else:
                print('This version does not exist.')
                raise NotImplementedError
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
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
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'Resnet_I3D':
            if args.resnet_i3d_version == 'resnet50':
                model = I3D_Resnet50(num_classes=num_classes)
            elif args.resnet_i3d_version == 'resnet50_nl':
                model = I3D_Resnet50_NL(num_classes=num_classes)
            else:
                print('This version does not exist.')
                raise NotImplementedError
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'Baseline_Nonlocal':
            model = Baseline_Nonlocal(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'ARTNet':
            model = ARTNet(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'LTC':
            model = LTC(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'P3D':
            model = P3D(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'S3D':
            if args.s3d_version == 'fast_s3d':
                model = Fast_S3D(num_classes=num_classes)
            elif args.s3d_version == 's3d_g':
                model = S3D_G(num_classes=num_classes)
            else:
                print('This version does not exist.')
                raise NotImplementedError
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'I3D':
            model = I3D(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'FstCN':
            model = FstCN(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'LRCN':
            model = LRCN(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'FFC_C3D':
            model = FFC_C3D(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'Mnasnet':
            model = MnasNet(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'Baseline_Spectral_Norm':
            model = Baseline_Spectral_Norm(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        elif args.model_name == 'I3D_pretrained':
            model = InceptionI3d(num_classes=num_classes)
            checkpoint = torch.load(args.infer_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
        else:
            print('This model does not exist.')
            raise NotImplementedError
    else:
        print('This dataset does not exist.')
        raise NotImplementedError
    cap = cv2.VideoCapture(args.video_path)
    prev_frame_time = 0.0
    clip = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        tmp_ = center_crop(cv2.resize(frame, (171, 128))) # [128, 171, 3] -> [112, 112, 3]
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs) # [1, 3, 16, 112, 112]
            inputs = torch.autograd.Variable(inputs, requires_grad=False).cuda()
            with torch.no_grad():
                outputs = model.forward(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame, "Class: " + class_names[label].split(' ')[-1].strip(), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
            cv2.putText(frame, "Score: %.3f" % probs[0][label], (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
            cv2.putText(frame, "FPS: " + str(int(fps)), (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
            clip.pop(0)
        else:
            print('Error clip length')
            exit()
        cv2.imshow(args.model_name + ' result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()