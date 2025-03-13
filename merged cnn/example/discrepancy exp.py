import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
import numpy as np
import time
import imageio
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx

import sys
sys.path.append("..//")

from github.engine.netload import compute_quant_net_diff, compute_net_diff, compute_prune_net_diff


class CNet1(nn.Module):
    def __init__(self):
        super(CNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.r2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.r3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.r1(self.conv1(x))
        x = F.max_pool2d(self.r2(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = self.r3(self.fc1(x))
        x = self.fc2(x)
        return x #F.log_softmax(x)


class CNet2(nn.Module):
    def __init__(self):
        super(CNet2, self).__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.r2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.r3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = x.to('cpu')
        x = self.quant(x)
        x = self.r1(self.conv1(x))
        x = F.max_pool2d(self.r2(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = self.r3(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x


class VGG1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.re1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.re1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.re2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.re2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.re3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.re4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.re5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.refc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.refc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.re1_1(self.conv1_1(x))
        x = self.pool1(self.re1_2(self.conv1_2(x)))
        x = self.re2_1(self.conv2_1(x))
        x = self.pool2(self.re2_2(self.conv2_2(x)))
        x = self.re3_1(self.conv3_1(x))
        x = self.re3_2(self.conv3_2(x))
        x = self.pool3(self.re3_3(self.conv3_3(x)))
        x = self.re4_1(self.conv4_1(x))
        x = self.re4_2(self.conv4_2(x))
        x = self.pool4(self.re4_3(self.conv4_3(x)))
        x = self.re5_1(self.conv5_1(x))
        x = self.re5_2(self.conv5_2(x))
        x = self.pool5(self.re5_3(self.conv5_3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG2(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.re1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.re1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.re2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.re2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.re3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.re3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.re4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.re4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.re5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.re5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 128)
        self.refc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.refc2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.re1_1(self.conv1_1(x))
        x = self.pool1(self.re1_2(self.conv1_2(x)))
        x = self.re2_1(self.conv2_1(x))
        x = self.pool2(self.re2_2(self.conv2_2(x)))
        x = self.re3_1(self.conv3_1(x))
        x = self.re3_2(self.conv3_2(x))
        x = self.pool3(self.re3_3(self.conv3_3(x)))
        x = self.re4_1(self.conv4_1(x))
        x = self.re4_2(self.conv4_2(x))
        x = self.pool4(self.re4_3(self.conv4_3(x)))
        x = self.re5_1(self.conv5_1(x))
        x = self.re5_2(self.conv5_2(x))
        x = self.pool5(self.re5_3(self.conv5_3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


if __name__ == '__main__':
    small_path = '../data/mnist_net_s.pth'
    large_path = '../data/mnist_net_l.pth'
    large_path2 = '../data/mnist_fx_net_l.pth'
    small_path2 = '../data/mnist_fx_net_s.pth'
    large_path3 = '../data/mnist_ptsq_net_l.pth'
    small_path3 = '../data/mnist_ptsq_net_s.pth'
    image_path = '../data/13.jpg'
    IM = imageio.v2.imread(image_path)
    IM = IM[:, :, np.newaxis]
    IM = IM / 255
    IM = (IM - 0.5) / 0.5
    lb = -0.05
    ub = 0.05

    # MNIST Quant
    print("MNIST Dataset with Quantization Model")
    print("Eager Mode QAT")
    net1 = torch.load(large_path, map_location=torch.device('cpu'))
    net2 = torch.load(small_path, map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_quant_net_diff(CNet2(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("FX Mode QAT")
    net2 = torch.load(small_path2, map_location=torch.device('cpu'))
    temp1 = net2['conv1_input_scale_0']
    temp2 = net2['conv1_input_zero_point_0']
    net3 = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_input_scale_0'
                                    else (k, v) for k, v in net2.items()])
    net3 = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_input_zero_point_0'
                                    else (k, v) for k, v in net3.items()])
    start = time.time()
    max_range = compute_quant_net_diff(CNet2(), net1, net3, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("Eager Mode Static")
    net1 = CNet1()
    net2 = CNet2()
    fused_list = [['conv1', 'r1'], ['conv2', 'r2'], ['fc1', 'r3']]
    net1.load_state_dict(torch.load(large_path3, map_location=torch.device('cpu')))
    net2.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    net2 = torch.quantization.fuse_modules(net2, fused_list, inplace=True)
    net2 = torch.quantization.prepare_qat(net2.train(), inplace=True)
    net2 = torch.quantization.convert(net2)
    net2.load_state_dict(torch.load(small_path3, map_location=torch.device('cpu')))
    test_img = imageio.v2.imread('../data/13.jpg')
    test_img = test_img[:, :, np.newaxis]
    test_img = (test_img / 255 - 0.5) / 0.5
    input = torch.from_numpy(test_img.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
    with torch.no_grad():
        output1 = net1(input)
        output2 = net2(input)
    print(output1)
    print(output2)
    print(np.linalg.norm(abs(np.array(output1 - output2))))

    print("FX Mode Static")
    net2 = torch.load('../data/mnist_fxsta_net_s.pth', map_location=torch.device('cpu'))
    temp1 = net2['conv1_input_scale_0']
    temp2 = net2['conv1_input_zero_point_0']
    net3 = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_input_scale_0'
                                    else (k, v) for k, v in net2.items()])
    net3 = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_input_zero_point_0'
                                    else (k, v) for k, v in net3.items()])
    start = time.time()
    max_range = compute_quant_net_diff(CNet2(), net1, net3, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    # MNIST Pruning
    print("MNIST Dataset with Pruning Model")
    print("Local Unstru")
    net1 = torch.load(large_path, map_location=torch.device('cpu'))
    net2 = torch.load('../data/mnist_lopru_net_s.pth', map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_prune_net_diff(CNet1(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("Global Unstru")
    net2 = torch.load('../data/mnist_glpru_net_s.pth', map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_prune_net_diff(CNet1(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("Local Stru")
    net2 = torch.load('../data/mnist_strupru_net_s.pth', map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_prune_net_diff(CNet1(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("Random Stru")
    net2 = torch.load('../data/mnist_randstrupru_net_s.pth', map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_prune_net_diff(CNet1(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    image_path = '../data/0022.jpg'
    IM = imageio.v2.imread(image_path)
    # IM = IM[:, :, np.newaxis]
    IM = IM / 255
    IM = (IM - 0.5) / 0.5
    lb = -0.05
    ub = 0.05
    net1 = torch.load('../data/cifar_vgg_l.pth', map_location=torch.device('cpu'))

    # CIFAR10 Quant
    print("CIFAR10 Dataset with Quant Model")
    print("Eager Mode QAT")
    net2 = torch.load('../data/cifar_vgg_s.pth', map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_quant_net_diff(VGG2(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("FX Mode QAT")
    net2 = torch.load('../data/cifar_vgg_fx_s.pth', map_location=torch.device('cpu'))
    temp1 = net2['conv1_1_input_scale_0']
    temp2 = net2['conv1_1_input_zero_point_0']
    net3 = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_1_input_scale_0'
                                    else (k, v) for k, v in net2.items()])
    net3 = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_1_input_zero_point_0'
                                    else (k, v) for k, v in net3.items()])
    start = time.time()
    max_range = compute_quant_net_diff(VGG2(), net1, net3, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("Eager Mode Static")
    net1 = VGG1()
    net2 = VGG2()
    fused_list = [['conv1_1', 're1_1'], ['conv1_2', 're1_2'], ['conv2_1', 're2_1'], ['conv2_2', 're2_2'],
                  ['conv3_1', 're3_1'], ['conv3_2', 're3_2'], ['conv3_3', 're3_3'], ['conv4_1', 're4_1'],
                  ['conv4_2', 're4_2'], ['conv4_3', 're4_3'], ['conv5_1', 're5_1'], ['conv5_2', 're5_2'],
                  ['conv5_3', 're5_3']]
    net1.load_state_dict(torch.load('../data/cifar_vgg_l.pth', map_location=torch.device('cpu')))
    net2.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    net2 = torch.quantization.fuse_modules(net2, fused_list, inplace=True)
    net2 = torch.quantization.prepare_qat(net2.train(), inplace=True)
    net2 = torch.quantization.convert(net2)
    net2.load_state_dict(torch.load('../data/cifar_vgg_ptsq_s.pth', map_location=torch.device('cpu')))
    test_img = imageio.v2.imread('../data/0022.jpg')
    test_img = (test_img / 255 - 0.5) / 0.5
    input = torch.from_numpy(test_img.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
    with torch.no_grad():
        output1 = net1(input)
        output2 = net2(input)
    print(output1)
    print(output2)
    print(np.linalg.norm(abs(np.array(output1 - output2))))

    print("FX Mode Static")
    net1 = VGG1()
    net2 = VGG2()
    net1.load_state_dict(torch.load('../data/cifar_vgg_l.pth', map_location=torch.device('cpu')))
    qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
    example_input = (torch.randn(1, 3, 32, 32))
    net2 = quantize_fx.prepare_qat_fx(net2.train(), qconfig_mapping, example_input)
    net2 = quantize_fx.convert_fx(net2)
    net2.load_state_dict(torch.load('../data/cifar_vgg_fxsta_s.pth', map_location=torch.device('cpu')))
    test_img = imageio.v2.imread('../data/0022.jpg')
    test_img = (test_img / 255 - 0.5) / 0.5
    input = torch.from_numpy(test_img.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
    with torch.no_grad():
        output1 = net1(input)
        output2 = net2(input)
    print(output1)
    print(output2)
    print(np.linalg.norm(abs(np.array(output1 - output2))))

    # CIFAR10 Pruning
    print("Local Unstru")
    net2 = torch.load('../data/cifar_vgg_lopru_s.pth', map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_prune_net_diff(VGG1(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("Global Unstru")
    net2 = torch.load('../data/cifar_vgg_glpru_s.pth', map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_prune_net_diff(VGG1(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("Local Stru")
    net2 = torch.load('../data/cifar_vgg_strupru_s.pth', map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_prune_net_diff(VGG1(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))

    print("Random Stru")
    net2 = torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu'))
    start = time.time()
    max_range = compute_prune_net_diff(VGG1(), net1, net2, IM, lb, ub)
    end = time.time()
    print("Time: " + str(end - start))
    print(max_range)
    print(np.linalg.norm(np.mean(abs(np.array(max_range)), 1)))
