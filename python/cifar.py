#!/usr/bin/env python
# coding: utf-8

import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as op
import os

trans = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                        torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.CIFAR10(root="./datasets", train=True, download=True, transform=trans)
test_data = torchvision.datasets.CIFAR10(root="./datasets", train=False, download=True, transform=trans)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, residual_path=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = nn.Conv2d(in_channels, out_channels, (3, 3), stride=strides, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.a1 = nn.ReLU()

        self.c2 = nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_channels)

        if self.residual_path:
            self.down_c1 = nn.Conv2d(in_channels, out_channels, (1, 1), stride=strides, padding=0, bias=False)
            self.down_b1 = nn.BatchNorm2d(out_channels)

        self.a2 = nn.ReLU()

    def forward(self, inputs):
        residual = inputs
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        y = self.b2(x)
        if self.residual_path:
            residual = self.down_c1(residual)
            residual = self.down_b1(residual)
        outputs = self.a2(y + residual)
        return outputs


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        self.a1 = nn.ReLU()
        self.r1 = ResnetBlock(64, 64, residual_path=False)
        self.r2 = ResnetBlock(64, 64, residual_path=False)
        self.r3 = ResnetBlock(64, 128, strides=2, residual_path=True)
        self.r4 = ResnetBlock(128, 128, residual_path=False)
        self.r5 = ResnetBlock(128, 256, strides=2, residual_path=True)
        self.r6 = ResnetBlock(256, 256, residual_path=False)
        self.r7 = ResnetBlock(256, 512, strides=2, residual_path=True)
        self.r8 = ResnetBlock(512, 512, residual_path=False)
        self.p1 = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Flatten()
        self.l1 = nn.Linear(512, 10)

    def forward(self, inputs):
        outputs = self.c1(inputs)
        outputs = self.b1(outputs)
        outputs = self.a1(outputs)
        outputs = self.r1(outputs)
        outputs = self.r2(outputs)
        outputs = self.r3(outputs)
        outputs = self.r4(outputs)
        outputs = self.r5(outputs)
        outputs = self.r6(outputs)
        outputs = self.r7(outputs)
        outputs = self.r8(outputs)
        outputs = self.p1(outputs)
        outputs = self.f1(outputs)
        outputs = self.l1(outputs)
        return outputs


save_path = "./save.pth"
if os.path.exists(save_path):
    model = torch.load(save_path)
    print("[*]Load Model...")
else:
    model = ResNet18()
device = torch.device("cuda")
model = model.to(device)
loss = nn.CrossEntropyLoss().to(device)
optimizer = op.Adam(model.parameters(), lr=1e-3)

train_eposchs = 0
print("[*]Start Training...")
for epochs in range(train_eposchs):
    model.train()
    for datas in train_loader:
        imgs, labels = datas
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        losses = loss(outputs, labels)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print("[*]Eposch:", epochs + 1, "\nTrain Loss:", losses.item())

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0

        for datas in test_loader:
            imgs, labels = datas
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            pred = outputs.argmax(dim=1)

            correct = torch.eq(pred, labels).float().sum().item()
            total_correct += correct
            total_num += imgs.size(0)
    acc = total_correct / total_num
    print("Test Acc:", acc)

print("[*]Over Training...")
torch.save(model, save_path)
inputs = torch.randn(1, 3, 32, 32).to(device)
torch.onnx.export(model, inputs, "save.onnx", export_params=True)
print("[*]Save Model...")
print(model)
