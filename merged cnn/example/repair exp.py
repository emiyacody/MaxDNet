import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub
import collections
import imageio
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn.utils.prune as prune
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    QConfigMapping,
)

import sys
sys.path.append("..//")

from github.engine.netload import compute_quant_net_diff, compute_prune_net_diff


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


def quant_repair(model, model_s, pth_l, pth_s, IM, lb, ub, new_pth, alpha, fx):
    max_range = compute_quant_net_diff(model, pth_l, pth_s, IM, lb, ub)
    max_range = np.array(max_range)
    optimizer = optim.SGD(model_s.parameters(), lr=0.0001, momentum=0.9)
    loss_func = torch.nn.MSELoss()
    dis_list = []

    dis = np.linalg.norm(np.mean(abs(max_range), 1))
    dis_list.append(dis)
    temp_range = max_range

    for k in range(5):
        print("%d iteration" % (k + 1))
        with torch.no_grad():
            inputs = torch.from_numpy(IM.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
            outputs2 = model_s(inputs)
            labels = outputs2.squeeze() + (torch.from_numpy(np.mean(temp_range, 1)[np.newaxis, :]).to(torch.float32)) * alpha

        print("Start re-train")

        for i in range(5):
            print("%d epoch re-train" % (i + 1))
            outputs = model_s(inputs.to(torch.float32))
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # torch.save(model_s.state_dict(), new_pth)
        if fx:
            net_temp = copy.deepcopy(model_s)
            net_temp = quantize_fx.convert_fx(net_temp)
            torch.save(net_temp.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            temp1 = pth_s_new['conv1_1_input_scale_0']
            temp2 = pth_s_new['conv1_1_input_zero_point_0']
            pth_s_new = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_1_input_scale_0'
                                            else (k, v) for k, v in pth_s_new.items()])
            pth_s_new = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_1_input_zero_point_0'
                                            else (k, v) for k, v in pth_s_new.items()])
        else:
            net_temp = copy.deepcopy(model_s)
            net_temp = torch.quantization.convert(net_temp)
            torch.save(net_temp.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
        new_range = compute_quant_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
        new_range = np.array(new_range)
        dis = np.linalg.norm(np.mean(abs(new_range), 1))
        dis_list.append(dis)
        temp_range = new_range
        print(dis)

    return dis_list


def pruning_repair(model, model_s, pth_l, pth_s, IM, lb, ub, new_pth, alpha, p_method):
    max_range = compute_prune_net_diff(model, pth_l, pth_s, IM, lb, ub)
    max_range = np.array(max_range)
    optimizer = optim.SGD(model_s.parameters(), lr=0.0001, momentum=0.9)
    loss_func = torch.nn.MSELoss()
    dis_list = []

    dis = np.linalg.norm(np.mean(abs(max_range), 1))
    dis_list.append(dis)
    temp_range = max_range

    for k in range(5):
        print("%d iteration" % (k + 1))
        with torch.no_grad():
            inputs = torch.from_numpy(IM.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
            outputs2 = model_s(inputs)
            labels = outputs2.squeeze() + (
                torch.from_numpy(np.mean(temp_range, 1)[np.newaxis, :]).to(torch.float32)) * alpha

        print("Start re-train")

        for i in range(5):
            print("%d epoch re-train" % (i + 1))
            outputs = model_s(inputs.to(torch.float32))
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if p_method == 'local_unstru':
            for name, module in model_s.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
            torch.save(model_s.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_prune_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range
        elif p_method == 'global_unstru':
            parameters_to_prune = ((model_s.conv1_1, 'weight'), (model_s.conv1_2, 'weight'), (model_s.conv2_1, 'weight'),
                                   (model_s.conv2_2, 'weight'), (model_s.conv3_1, 'weight'), (model_s.conv3_2, 'weight'),
                                   (model_s.conv3_3, 'weight'), (model_s.conv4_1, 'weight'), (model_s.conv4_2, 'weight'),
                                   (model_s.conv4_3, 'weight'), (model_s.conv5_1, 'weight'), (model_s.conv5_2, 'weight'),
                                   (model_s.conv5_3, 'weight'), (model_s.fc1, 'weight'), (model_s.fc2, 'weight'),
                                   (model_s.fc3, 'weight'))
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
            for name, module in model_s.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.remove(module, 'weight')
            torch.save(model_s.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_prune_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range
        elif p_method == 'l1_stru':
            for name, module in model_s.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=0.2, n=1, dim=1)
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.ln_structured(module, name='weight', amount=0.2, n=1, dim=1)
                    prune.remove(module, 'weight')
            torch.save(model_s.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_prune_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range
        elif p_method == 'rand_stru':
            for name, module in model_s.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.random_structured(module, name='weight', amount=0.2, dim=1)
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.random_structured(module, name='weight', amount=0.2, dim=1)
                    prune.remove(module, 'weight')
            torch.save(model_s.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_prune_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range

    return dis_list


def dis_quant_repair(model, teacher, student, pth_l, pth_s, IM, lb, ub, new_pth, fx, T, learning_rate,
                     alpha, dis_add):
    max_range = compute_quant_net_diff(model, pth_l, pth_s, IM, lb, ub)
    max_range = np.array(max_range)
    dis_list = []
    acc_list = []
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode

    dis = np.linalg.norm(np.mean(abs(max_range), 1))
    dis_list.append(dis)
    temp_range = max_range

    criterion2 = nn.KLDivLoss()
    mse_loss = nn.MSELoss()

    for k in range(5):
        print("%d iteration" % (k + 1))
        with torch.no_grad():
            inputs = torch.from_numpy(IM.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
            if dis_add:
                teacher_logits = teacher(inputs) - 0.5 * torch.from_numpy(np.mean(temp_range, 1)[np.newaxis, :]).to(torch.float32)
            else:
                teacher_logits = teacher(inputs)

        print("Start re-train")

        for i in range(5):
            print("%d epoch re-train" % (i + 1))
            # Forward pass with the student model
            student_logits = student(inputs)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (
                    T ** 2)

            # # Calculate the true label loss
            # label_loss = ce_loss(student_logits, labels)
            label_loss = mse_loss(student_logits, teacher_logits)

            # Weighted sum of the two losses
            loss = alpha * soft_targets_loss + (1 - alpha) * label_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # torch.save(model_s.state_dict(), new_pth)
        if fx:
            # CIFAR VGG
            net_temp = copy.deepcopy(student)
            net_temp = quantize_fx.convert_fx(net_temp)
            torch.save(net_temp.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            temp1 = pth_s_new['conv1_1_input_scale_0']
            temp2 = pth_s_new['conv1_1_input_zero_point_0']
            pth_s_new = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_1_input_scale_0'
                                                 else (k, v) for k, v in pth_s_new.items()])
            pth_s_new = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_1_input_zero_point_0'
                                                 else (k, v) for k, v in pth_s_new.items()])
        else:
            net_temp = copy.deepcopy(student)
            net_temp = torch.quantization.convert(net_temp)
            torch.save(net_temp.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
        new_range = compute_quant_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
        new_range = np.array(new_range)
        dis = np.linalg.norm(np.mean(abs(new_range), 1))
        dis_list.append(dis)
        temp_range = new_range
        print(dis)

    return dis_list


def dis_pruning_repair(model, teacher, student, pth_l, pth_s, IM, lb, ub, new_pth, T, learning_rate,
                     alpha, dis_add, p_method):
    max_range = compute_prune_net_diff(model, pth_l, pth_s, IM, lb, ub)
    max_range = np.array(max_range)
    dis_list = []
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode

    dis = np.linalg.norm(np.mean(abs(max_range), 1))
    dis_list.append(dis)
    temp_range = max_range
    mse_loss = nn.MSELoss()

    for k in range(5):
        print("%d iteration" % (k + 1))
        with torch.no_grad():
            inputs = torch.from_numpy(IM.copy().transpose(2, 0, 1)).unsqueeze_(0).to(torch.float32)
            if dis_add:
                teacher_logits = teacher(inputs) - 0.5 * torch.from_numpy(np.mean(temp_range, 1)[np.newaxis, :]).to(
                    torch.float32)
            else:
                teacher_logits = teacher(inputs)

        print("Start re-train")

        for i in range(5):
            print("%d epoch re-train" % (i + 1))
            # Forward pass with the student model
            student_logits = student(inputs)

            # Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper
            # "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (
                    T ** 2)

            # # Calculate the true label loss
            # label_loss = ce_loss(student_logits, labels)
            label_loss = mse_loss(student_logits, teacher_logits)

            # Weighted sum of the two losses
            loss = alpha * soft_targets_loss + (1 - alpha) * label_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if p_method == 'local_unstru':
            for name, module in student.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')
            torch.save(student.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_prune_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range
        elif p_method == 'global_unstru':
            parameters_to_prune = ((student.conv1_1, 'weight'), (student.conv1_2, 'weight'), (student.conv2_1, 'weight'),
                                   (student.conv2_2, 'weight'), (student.conv3_1, 'weight'), (student.conv3_2, 'weight'),
                                   (student.conv3_3, 'weight'), (student.conv4_1, 'weight'), (student.conv4_2, 'weight'),
                                   (student.conv4_3, 'weight'), (student.conv5_1, 'weight'), (student.conv5_2, 'weight'),
                                   (student.conv5_3, 'weight'), (student.fc1, 'weight'), (student.fc2, 'weight'),
                                   (student.fc3, 'weight'))
            prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)
            for name, module in student.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.remove(module, 'weight')
            torch.save(student.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_prune_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range
        elif p_method == 'l1_stru':
            for name, module in student.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=0.2, n=1, dim=1)
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.ln_structured(module, name='weight', amount=0.2, n=1, dim=1)
                    prune.remove(module, 'weight')
            torch.save(student.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_prune_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range
        elif p_method == 'rand_stru':
            for name, module in student.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.random_structured(module, name='weight', amount=0.2, dim=1)
                    prune.remove(module, 'weight')
                elif isinstance(module, torch.nn.Linear):
                    prune.random_structured(module, name='weight', amount=0.2, dim=1)
                    prune.remove(module, 'weight')
            torch.save(student.state_dict(), new_pth)
            pth_s_new = torch.load(new_pth, map_location=torch.device('cpu'))
            new_range = compute_prune_net_diff(model, pth_l, pth_s_new, IM, lb, ub)
            new_range = np.array(new_range)
            dis = np.linalg.norm(np.mean(abs(new_range), 1))
            dis_list.append(dis)
            temp_range = new_range

    return dis_list


if __name__ == '__main__':
    image_path = '../data/0022.jpg'
    IM = imageio.v2.imread(image_path)
    IM = IM / 255
    IM = (IM - 0.5) / 0.5
    lb = -0.05
    ub = 0.05

    net1 = torch.load('../data/cifar_vgg_l.pth', map_location=torch.device('cpu'))
    print("Rebuild Dataset Retraining")
    for al in (0.5, 0.2, 0.7, 0.9):
        print("alpha = %.1f" % al)

        print("FX Mode QAT")
        net2 = torch.load('../data/cifar_vgg_fx_s.pth', map_location=torch.device('cpu'))
        temp1 = net2['conv1_1_input_scale_0']
        temp2 = net2['conv1_1_input_zero_point_0']
        net3 = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_1_input_scale_0'
                                        else (k, v) for k, v in net2.items()])
        net3 = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_1_input_zero_point_0'
                                        else (k, v) for k, v in net3.items()])
        net2 = VGG2()
        example_input = (torch.randn(1, 3, 32, 32))
        qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
        net2 = quantize_fx.prepare_qat_fx(net2.train(), qconfig_mapping, example_input)
        net2.load_state_dict(torch.load('../data/cifar_vgg_fx_s_be.pth', map_location=torch.device('cpu')))

        dis_list = quant_repair(VGG2(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_fx_temp.pth', al, True)
        print(dis_list)

        print('Eager Mode QAT')
        net3 = torch.load('../data/cifar_vgg_s.pth', map_location=torch.device('cpu'))
        net2 = VGG2()
        net2.eval()
        net2.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        fused_list = [['conv1_1', 're1_1'], ['conv1_2', 're1_2'], ['conv2_1', 're2_1'], ['conv2_2', 're2_2'],
                      ['conv3_1', 're3_1'], ['conv3_2', 're3_2'], ['conv3_3', 're3_3'], ['conv4_1', 're4_1'],
                      ['conv4_2', 're4_2'], ['conv4_3', 're4_3'], ['conv5_1', 're5_1'], ['conv5_2', 're5_2'],
                      ['conv5_3', 're5_3']]
        net2 = torch.quantization.fuse_modules(net2, fused_list, inplace=True)
        net2 = torch.quantization.prepare_qat(net2.train(), inplace=True)
        net2.load_state_dict(torch.load('../data/cifar_vgg_s_be.pth', map_location=torch.device('cpu')))
        dis_list = quant_repair(VGG2(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_s_temp.pth', al, False)
        print(dis_list)

        print('Local Unstructured Pruning')
        net3 = torch.load('../data/cifar_vgg_lopru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_lopru_s.pth', map_location=torch.device('cpu')))
        dis_list = pruning_repair(VGG1(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth', al, 'local_unstru')
        print(dis_list)

        print('Global Unstructured Pruning')
        net3 = torch.load('../data/cifar_vgg_glpru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_glpru_s.pth', map_location=torch.device('cpu')))
        dis_list = pruning_repair(VGG1(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth', al, 'global_unstru')
        print(dis_list)

        print('Local Structured Pruning')
        net3 = torch.load('../data/cifar_vgg_strupru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_strupru_s.pth', map_location=torch.device('cpu')))
        dis_list = pruning_repair(VGG1(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_strupru_s_temp.pth', al, 'l1_stru')
        print(dis_list)

        print('Random Structured Pruning')
        net3 = torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu')))
        dis_list = pruning_repair(VGG1(), net2, net1, net3, IM, lb, ub, '../data/cifar_vgg_randstrupru_s_temp.pth', al, 'rand_stru')
        print(dis_list)

    net1 = VGG1()
    net1.load_state_dict(torch.load('../data/cifar_vgg_l.pth', map_location=torch.device('cpu')))
    pth_l = torch.load('../data/cifar_vgg_l.pth', map_location=torch.device('cpu'))
    temper = 2
    print("Knowledge Distillation with Discrepancy")
    for al in (0.2, 0.5, 0.7, 0.9):
        print("alpha = %.1f" % al)

        print("FX Mode QAT")
        net2 = torch.load('../data/cifar_vgg_fx_s.pth', map_location=torch.device('cpu'))
        temp1 = net2['conv1_1_input_scale_0']
        temp2 = net2['conv1_1_input_zero_point_0']
        pth_s = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_1_input_scale_0'
                                        else (k, v) for k, v in net2.items()])
        pth_s = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_1_input_zero_point_0'
                                        else (k, v) for k, v in pth_s.items()])
        net2 = VGG2()
        example_input = (torch.randn(1, 3, 32, 32))
        qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
        net2 = quantize_fx.prepare_qat_fx(net2.train(), qconfig_mapping, example_input)
        net2.load_state_dict(torch.load('../data/cifar_vgg_fx_s_be.pth', map_location=torch.device('cpu')))

        dis_list = dis_quant_repair(VGG2(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_fx_temp.pth', True,
                                    temper, 0.001, al, True)
        print(dis_list)

        print('Eager Mode QAT')
        pth_s = torch.load('../data/cifar_vgg_s.pth', map_location=torch.device('cpu'))
        net2 = VGG2()
        net2.eval()
        net2.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        fused_list = [['conv1_1', 're1_1'], ['conv1_2', 're1_2'], ['conv2_1', 're2_1'], ['conv2_2', 're2_2'],
                      ['conv3_1', 're3_1'], ['conv3_2', 're3_2'], ['conv3_3', 're3_3'], ['conv4_1', 're4_1'],
                      ['conv4_2', 're4_2'], ['conv4_3', 're4_3'], ['conv5_1', 're5_1'], ['conv5_2', 're5_2'],
                      ['conv5_3', 're5_3']]
        net2 = torch.quantization.fuse_modules(net2, fused_list, inplace=True)
        net2 = torch.quantization.prepare_qat(net2.train(), inplace=True)
        net2.load_state_dict(torch.load('../data/cifar_vgg_s_be.pth', map_location=torch.device('cpu')))
        dis_list = dis_quant_repair(VGG2(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_fx_temp.pth', False,
                                    temper, 0.001, al, True)
        print(dis_list)

        print('Local Unstructured Pruning')
        pth_s = torch.load('../data/cifar_vgg_lopru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_lopru_s.pth', map_location=torch.device('cpu')))
        dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
                                      temper, 0.001, al, True, 'local_unstru')
        print(dis_list)

        print('Global Unstructured Pruning')
        pth_s = torch.load('../data/cifar_vgg_glpru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_glpru_s.pth', map_location=torch.device('cpu')))
        dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
                                      temper, 0.001, al, True, 'global_unstru')
        print(dis_list)

        print('Local Structured Pruning')
        pth_s = torch.load('../data/cifar_vgg_strupru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_strupru_s.pth', map_location=torch.device('cpu')))
        dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
                                      temper, 0.001, al, True, 'l1_stru')
        print(dis_list)

        print('Random Structured Pruning')
        pth_s = torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu')))
        dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
                                      temper, 0.001, al, True, 'rand_stru')
        print(dis_list)

    al = 0.5
    for temper in (2, 0.8, 5, 10):
        print("Temperature = %.1f" % temper)

        print("FX Mode QAT")
        net2 = torch.load('../data/cifar_vgg_fx_s.pth', map_location=torch.device('cpu'))
        temp1 = net2['conv1_1_input_scale_0']
        temp2 = net2['conv1_1_input_zero_point_0']
        pth_s = collections.OrderedDict([('quant.scale', temp1) if k == 'conv1_1_input_scale_0'
                                        else (k, v) for k, v in net2.items()])
        pth_s = collections.OrderedDict([('quant.zero_point', temp2) if k == 'conv1_1_input_zero_point_0'
                                        else (k, v) for k, v in pth_s.items()])
        net2 = VGG2()
        example_input = (torch.randn(1, 3, 32, 32))
        qconfig_mapping = get_default_qat_qconfig_mapping("qnnpack")
        net2 = quantize_fx.prepare_qat_fx(net2.train(), qconfig_mapping, example_input)
        net2.load_state_dict(torch.load('../data/cifar_vgg_fx_s_be.pth', map_location=torch.device('cpu')))

        dis_list = dis_quant_repair(VGG2(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_fx_temp.pth', True,
                                    temper, 0.001, al, True)
        print(dis_list)

        print('Eager Mode QAT')
        pth_s = torch.load('../data/cifar_vgg_s.pth', map_location=torch.device('cpu'))
        net2 = VGG2()
        net2.eval()
        net2.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        fused_list = [['conv1_1', 're1_1'], ['conv1_2', 're1_2'], ['conv2_1', 're2_1'], ['conv2_2', 're2_2'],
                      ['conv3_1', 're3_1'], ['conv3_2', 're3_2'], ['conv3_3', 're3_3'], ['conv4_1', 're4_1'],
                      ['conv4_2', 're4_2'], ['conv4_3', 're4_3'], ['conv5_1', 're5_1'], ['conv5_2', 're5_2'],
                      ['conv5_3', 're5_3']]
        net2 = torch.quantization.fuse_modules(net2, fused_list, inplace=True)
        net2 = torch.quantization.prepare_qat(net2.train(), inplace=True)
        net2.load_state_dict(torch.load('../data/cifar_vgg_s_be.pth', map_location=torch.device('cpu')))
        dis_list = dis_quant_repair(VGG2(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_fx_temp.pth', False,
                                    temper, 0.001, al, True)
        print(dis_list)

        print('Local Unstructured Pruning')
        pth_s = torch.load('../data/cifar_vgg_lopru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_lopru_s.pth', map_location=torch.device('cpu')))
        dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
                                      temper, 0.001, al, True, 'local_unstru')
        print(dis_list)

        print('Global Unstructured Pruning')
        pth_s = torch.load('../data/cifar_vgg_glpru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_glpru_s.pth', map_location=torch.device('cpu')))
        dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
                                      temper, 0.001, al, True, 'global_unstru')
        print(dis_list)

        print('Local Structured Pruning')
        pth_s = torch.load('../data/cifar_vgg_strupru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_strupru_s.pth', map_location=torch.device('cpu')))
        dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
                                      temper, 0.001, al, True, 'l1_stru')
        print(dis_list)

        print('Random Structured Pruning')
        pth_s = torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu'))
        net2 = VGG1()
        net2.load_state_dict(torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu')))
        dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
                                      temper, 0.001, al, True, 'rand_stru')
        print(dis_list)


    # temper = 2
    # for al in (0.2, 0.5, 0.7, 0.9):
    #     print("alpha = %.1f" % al)
    #     print('Random Structured Pruning')
    #     pth_s = torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu'))
    #     net2 = VGG1()
    #     net2.load_state_dict(torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu')))
    #     dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
    #                                   temper, 0.001, al, True, 'rand_stru')
    #     print(dis_list)
    #
    # al = 0.5
    # for temper in (2, 0.8, 5, 10):
    #     print("Temperature = %.1f" % temper)
    #     print('Random Structured Pruning')
    #     pth_s = torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu'))
    #     net2 = VGG1()
    #     net2.load_state_dict(torch.load('../data/cifar_vgg_randstrupru_s.pth', map_location=torch.device('cpu')))
    #     dis_list = dis_pruning_repair(VGG1(), net1, net2, pth_l, pth_s, IM, lb, ub, '../data/cifar_vgg_lopru_s_temp.pth',
    #                                   temper, 0.001, al, True, 'rand_stru')
    #     print(dis_list)

    # Rebuild Dataset Retraining
    # alpha = 0.5
    # al_repair_list1 = [[17.0159, 18.1480, 11.3299, 6.9041, 4.2543, 2.2669],
    #                    [16.3547, 28.9640, 23.2230, 14.8200, 8.4599, 9.0541],
    #                    [1.9602, 2.1744, 2.9405, 2.5069, 1.1136, 0.5915],
    #                    [0.9487, 2.1489, 0.376, 0.2090, 0.1326, 0.1633],
    #                    [32.5631, 12.7388, 15.6981, 8.4703, 2.7437, 0.9242],
    #                    [34.4265, 37.0091, 36.8323, 36.8606, 36.5052, 37.2667]]
    # model_list = ['FX QAT', 'Eager QAT', 'L-Unstru', 'G-Unstru', 'L-Stru', 'R-Stru']
    # x = [0, 1, 2, 3, 4, 5]
    # plt.figure(1)
    # plt.subplot(221)
    # plt.subplots_adjust(wspace=0.15, hspace=0.28)
    # plt.plot(x, al_repair_list1[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # # plt.legend()
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # # plt.xlabel('Epoch')
    # plt.ylabel('Discrepancy', fontweight='bold')
    # plt.title("Restoration \u03B2=0.5", fontweight='bold')
    #
    # # alpha = 0.2
    # al_repair_list2 = [[17.0159, 49.7563, 36.5522, 17.7726, 16.4339, 17.1840],
    #                    [16.3547, 23.7964, 26.0538, 18.8446, 12.0234, 8.7409],
    #                    [1.9602, 1.7875, 3.1237, 9.6764, 10.1975, 6.4987],
    #                    [0.9487, 1.3190, 2.3212, 7.3007, 10.3264, 7.9780],
    #                    [32.5631, 26.9553, 19.6763, 13.3569, 12.5066, 8.5676],
    #                    [34.4265, 38.4006, 39.7576, 36.6022, 36.8444, 37.0809]]
    # plt.subplot(222)
    # plt.plot(x, al_repair_list2[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # # plt.legend()
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.title("Restoration \u03B2=0.2", fontweight='bold')
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Discrepancy')
    #
    # # alpha = 0.7
    # al_repair_list3 = [[17.0159, 37.0152, 31.4372, 30.3657, 27.5570, 20.0554],
    #                [16.3547, 27.9664, 17.9052, 12.1141, 9.26309, 6.6716],
    #                [1.9602, 2.4650, 2.2777, 0.6267, 0.2518, 0.1254],
    #                [0.9487, 2.4183, 1.2086, 0.6787, 0.2912, 0.1648],
    #                [32.5631, 2.9510, 2.1989, 3.5418, 4.0629, 3.4599],
    #                [34.4265, 40.0203, 37.7566, 37.5839, 37.3043, 37.0177]]
    # plt.subplot(223)
    # plt.plot(x, al_repair_list3[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # # plt.legend()
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.xlabel('Epoch', fontweight='bold')
    # plt.ylabel('Discrepancy', fontweight='bold')
    # plt.title("Restoration \u03B2=0.7", fontweight='bold')
    #
    # # alpha = 0.9
    # al_repair_list4 = [[17.0159, 28.2078, 9.3894, 6.3262, 4.2187, 4.6673],
    #                [16.3546, 17.0808, 9.0188, 8.7126, 4.3832, 2.7320],
    #                [1.9602, 2.5039, 1.2626, 0.3771, 0.5612, 0.2525],
    #                [0.9487, 1.7146, 1.6020, 0.3410, 0.2605, 0.1836],
    #                [32.2631, 2.8909, 2.6792, 2.9946, 2.9394, 2.2905],
    #                [34.4265, 37.7057, 36.3620, 34.8492, 32.3743, 35.4302]]
    # plt.subplot(224)
    # plt.plot(x, al_repair_list4[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # # plt.legend()
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.xlabel('Epoch', fontweight='bold')
    # plt.title("Restoration \u03B2=0.9", fontweight='bold')
    # # plt.ylabel('Discrepancy')
    # plt.subplot(221)
    # font = matplotlib.font_manager.FontProperties(weight='bold')
    # plt.legend(ncol=4, bbox_to_anchor=(-0.03, 1.5), loc=2, prop=font)
    # plt.show()


    # Knowledge Distillation with Discrepancy
    # alpha = 0.5 \u03B1
    # al_repair_list1 = [[17.0159, 16.0726, 13.0464, 12.5303, 8.2822, 3.8791],
    #                    [16.3547, 19.8056, 13.3258, 8.9022, 4.6200, 3.6152],
    #                    [1.9602, 32.4089, 23.9370, 20.7356, 17.1848, 13.7829],
    #                    [0.9487, 32.9091, 20.2831, 15.1047, 12.4833, 10.5672],
    #                    [32.5631, 24.4849, 18.6494, 8.2906, 3.9545, 2.0776],
    #                    [34.4265, 35.8907, 36.1300, 35.9849, 36.0434, 36.9128]]
    # model_list = ['FX QAT', 'Eager QAT', 'L-Unstru', 'G-Unstru', 'L-Stru', 'R-Stru']
    # x = [0, 1, 2, 3, 4, 5]
    # plt.figure(2)
    # plt.subplot(221)
    # plt.subplots_adjust(wspace=0.15, hspace=0.28)
    # plt.plot(x, al_repair_list1[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, al_repair_list1[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.ylabel('Discrepancy', fontweight='bold')
    # plt.title("Restoration \u03B1=0.5", fontweight='bold')
    #
    # # alpha = 0.2
    # al_repair_list2 = [[17.0159, 21.1650, 16.6515, 9.5380, 4.9203, 3.2217],
    #                    [16.3549, 21.2221, 14.1368, 10.1070, 6.9807, 4.9157],
    #                    [1.9602, 32.4205, 24.0735, 21.0011, 17.5275, 15.1636],
    #                    [0.9487, 32.9491, 20.9534, 16.1252, 13.0874, 10.4162],
    #                    [32.5631, 26.0019, 18.7422, 10.6446, 5.1566, 3.1321],
    #                    [34.4265,39.2665, 38.2108, 35.3195, 34.8050, 35.6922]]
    # plt.subplot(222)
    # plt.plot(x, al_repair_list2[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, al_repair_list2[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.title("Restoration \u03B1=0.2", fontweight='bold')
    #
    # # alpha = 0.7
    # al_repair_list3 = [[17.0159, 17.8230, 11.2896, 8.6758, 4.6883, 3.2668],
    #                    [16.3547, 26.7524, 13.5409, 10.0336, 7.5729, 5.9510],
    #                    [1.9602, 32.3572, 23.5277, 20.5929, 17.1037, 14.5191],
    #                    [0.9487, 32.6525, 18.1259, 13.8412, 10.1815, 6.9580],
    #                    [32.5631, 23.1873, 17.8805, 9.9654, 2.7813, 1.1076],
    #                    [34.4265, 36.0511, 36.7356, 36.6940, 36.0341, 37.7178]]
    # plt.subplot(223)
    # plt.plot(x, al_repair_list3[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, al_repair_list3[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.xlabel('Epoch', fontweight='bold')
    # plt.ylabel('Discrepancy', fontweight='bold')
    # plt.title("Restoration \u03B1=0.7", fontweight='bold')
    #
    # # alpha = 0.9
    # al_repair_list4 = [[17.0159, 27.5133, 11.1682, 6.7256, 4.5233, 4.6501],
    #                    [16.3547, 25.2687, 19.5323, 13.9775, 12.1875, 9.9261],
    #                    [1.9602, 32.4518, 20.5700, 17.3608, 11.6369, 8.7380],
    #                    [0.9487, 32.4490, 17.8514, 14.7314, 6.6739, 5.8600],
    #                    [32.5631, 17.6335, 14.6738, 10.1525, 4.0932, 1.6181],
    #                    [34.4265, 37.6407, 35.3240, 36.6840, 36.4877, 35.1513]]
    # plt.subplot(224)
    # plt.plot(x, al_repair_list4[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, al_repair_list4[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.xlabel('Epoch', fontweight='bold')
    # plt.title("Restoration \u03B1=0.9", fontweight='bold')
    # # plt.ylabel('Discrepancy')
    # plt.subplot(221)
    # font = matplotlib.font_manager.FontProperties(weight='bold')
    # plt.legend(ncol=4, bbox_to_anchor=(-0.03, 1.5), loc=2, prop=font)
    # plt.show()
    #
    #
    # # Knowledge Distillation with Discrepancy
    # # T = 2
    # t_repair_list1 = [[17.0159, 16.0726, 13.0464, 12.5303, 8.2822, 3.8791],
    #                    [16.3547, 19.8056, 13.3258, 8.9022, 4.6200, 3.6152],
    #                    [1.9602, 32.4089, 23.9370, 20.7356, 17.1848, 13.7829],
    #                    [0.9487, 32.9091, 20.2831, 15.1047, 12.4833, 10.5672],
    #                    [32.5631, 24.4849, 18.6494, 8.2906, 3.9545, 2.0776],
    #                    [34.4265, 38.3773, 38.7000, 35.9948, 36.3187, 37.5785]]
    # model_list = ['FX QAT', 'Eager QAT', 'L-Unstru', 'G-Unstru', 'L-Stru', 'R-Stru']
    # x = [0, 1, 2, 3, 4, 5]
    # plt.figure(3)
    # plt.subplot(221)
    # plt.subplots_adjust(wspace=0.15, hspace=0.28)
    # plt.plot(x, t_repair_list1[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, t_repair_list1[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, t_repair_list1[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, t_repair_list1[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, t_repair_list1[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, t_repair_list1[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # # plt.legend()
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # # plt.xlabel('Epoch')
    # plt.ylabel('Discrepancy', fontweight='bold')
    # plt.title("Restoration T=2", fontweight='bold')
    #
    # # T = 0.8
    # t_repair_list2 = [[17.0159, 20.1269, 13.8727, 11.6018, 6.5191, 3.3584],
    #                    [16.3547, 22.9886, 13.6304, 9.4138, 6.3932, 3.8716],
    #                    [1.9602, 32.4268, 24.1465, 20.9578, 17.5498, 15.2013],
    #                    [0.9487, 32.9535, 20.9900, 16.1201, 13.0593, 10.3763],
    #                    [32.5631, 26.3698, 18.5130, 11.1060, 5.1678, 2.8418],
    #                    [34.4265, 38.9840, 38.0522, 35.5213, 37.1898, 38.4897]]
    # plt.subplot(222)
    # plt.plot(x, t_repair_list2[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, t_repair_list2[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, t_repair_list2[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, t_repair_list2[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, t_repair_list2[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, t_repair_list2[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # # plt.legend()
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.title("Restoration T=0.8", fontweight='bold')
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Discrepancy')
    #
    # # T = 5
    # t_repair_list3 = [[17.0159, 17.0460, 10.5331, 4.9179, 4.2617, 2.5741],
    #                [16.3547, 25.2393, 12.6097, 10.9419, 4.2369, 3.8855],
    #                [1.9602, 32.4657, 22.5573, 19.2451, 13.7647, 10.4465],
    #                [0.9487, 32.4185, 17.6263, 12.6506, 10.1718, 6.8230],
    #                [32.5631, 23.5907, 17.9559, 6.6282, 2.8104, 1.3392],
    #                [34.4265, 37.4281, 36.4475, 34.7257, 34.4698, 36.4166]]
    # plt.subplot(223)
    # plt.plot(x, t_repair_list3[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, t_repair_list3[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, t_repair_list3[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, t_repair_list3[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, t_repair_list3[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, t_repair_list3[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # # plt.legend()
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.xlabel('Epoch', fontweight='bold')
    # plt.ylabel('Discrepancy', fontweight='bold')
    # plt.title("Restoration T=5", fontweight='bold')
    #
    # # T = 10
    # t_repair_list4 = [[17.0159, 18.7740, 12.3453, 9.8254, 6.5849, 4.1901],
    #                [16.3546, 21.8156, 14.6317, 8.7723, 4.6072, 3.3420],
    #                [1.9602, 32.3769, 22.0557, 18.0023, 13.9534, 10.6369],
    #                [0.9487, 32.8585, 18.0836, 13.7133, 9.4766, 6.0498],
    #                [32.2631, 26.2142, 18.2777, 5.3345, 3.6205, 2.2946],
    #                [34.4265, 39.1932, 38.5602, 37.3130, 36.7417, 37.0974]]
    # plt.subplot(224)
    # plt.plot(x, t_repair_list4[0], color='blue', label=model_list[0], marker='.', markersize='7')
    # plt.plot(x, t_repair_list4[1], color='black', label=model_list[1], marker='.', markersize='7')
    # plt.plot(x, t_repair_list4[2], color='red', label=model_list[2], marker='.', markersize='7')
    # plt.plot(x, t_repair_list4[3], color='green', label=model_list[3], marker='.', markersize='7')
    # plt.plot(x, t_repair_list4[4], color='purple', label=model_list[4], marker='.', markersize='7')
    # plt.plot(x, t_repair_list4[5], color='orange', label=model_list[5], marker='.', markersize='7')
    # # plt.legend()
    # plt.xticks(np.array(x), x, fontweight='bold')
    # plt.yticks(fontweight='bold')
    # plt.xlabel('Epoch', fontweight='bold')
    # plt.title("Restoration T=10", fontweight='bold')
    # # plt.ylabel('Discrepancy')
    # plt.subplot(221)
    # font = matplotlib.font_manager.FontProperties(weight='bold')
    # plt.legend(ncol=4, bbox_to_anchor=(-0.03, 1.5), loc=2, prop=font)
    # plt.show()
