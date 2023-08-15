#!/usr/bin/env python
''' MENet Training code
Training code for Brain diseases classification
Written by Yilin Leng
https://doi.org/10.1016/j.compbiomed.2023.106788
'''
import os
import xlwt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from loaddata.dataset import *
from loaddata.metric_compute import getMCA
from pytorch_metric_learning.regularizers import LpRegularizer

from setting import parse_opts_dual
from model.generate_model import generate_model

keys_ = ['MRI_image']


def save_result_in_xls():
    file = xlwt.Workbook(encoding='utf-8')
    sheet = file.add_sheet('sheet1', cell_overwrite_ok=True)
    sheet.write_merge(0, 0, 2, 7, 'train')
    sheet.write_merge(0, 0, 8, 14, 'test')
    sheet.write(1, 0, 'Epoch')
    sheet.write(1, 1, 'lr')
    # sheet.write(1, 1, 'loss')
    col = ('loss', 'ACC', 'AUC', 'F1', 'SEN', 'SPE')
    for i in range(0, 6):
        sheet.write(1, i + 2, col[i])
        sheet.write(1, i + 8, col[i])
    file_name = '{}/{} {}.xls'.format(sets.save_path, sets.classes, time.strftime('%m-%d %H-%M', time.localtime(time.time())))
    file.save(file_name)
    return file, file_name, sheet


def train(train_loader):

    correct, predicted, predicted1 = [], [], []
    train_loss = 0
    net.cuda().train()

    for i, sample in enumerate(train_loader):
        MRI_img, PET_img, label = sample['MRI_image'], sample['PET_image'], sample['label']
        MRI_img, PET_img, label = MRI_img.cuda(), PET_img.cuda(), label.cuda()

        embeddings, outputs = net(MRI_img, PET_img)

        _, pred = torch.max(outputs.data, 1)
        score = F.softmax(outputs.data, dim=1)
        outputs = outputs.float().cuda()
        loss_cross = lossfunction_cross(outputs, label)

        loss = loss_cross

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        correct.extend(label.cpu().numpy())
        predicted.extend(pred.cpu().numpy())
        predicted1.extend(score[:, 1].cpu().numpy())

    scheduler.step()
    del MRI_img, PET_img, label, outputs, sample
    torch.cuda.empty_cache()

    acc, auc, F1_score, sen, spe = getMCA(correct, predicted, predicted1)
    return train_loss/len(train_loader), acc, auc, F1_score, sen, spe
    exit()


def test(test_loader):

    net.cuda().eval()
    test_loss = 0
    correct, predicted, predicted1 = [], [], []

    for batch_idx, sample in enumerate(test_loader):
        MRI_img, PET_img, label = sample['MRI_image'], sample['PET_image'], sample['label']
        MRI_img, PET_img, label = MRI_img.cuda(), PET_img.cuda(), label.cuda()

        with torch.no_grad():
            embeddings, outputs = net(MRI_img, PET_img)

            _, pred = torch.max(outputs.data, 1)
            score = F.softmax(outputs.data, dim=1)
            outputs = outputs.float().cuda()

            test_cross = lossfunction_cross(outputs, label)
            testloss = test_cross
            test_loss += testloss.item()

            correct.extend(label.cpu().numpy())
            predicted.extend(pred.cpu().numpy())
            predicted1.extend(score[:, 1].cpu().numpy())
    torch.cuda.empty_cache()

    acc, auc, F1_score, sen, spe = getMCA(correct, predicted, predicted1)
    return test_loss / len(test_loader), acc, auc, F1_score, sen, spe
    exit()





def main():

    # load loaddata
    train_dataset = Dataset_dual(sets, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=sets.batch_size, num_workers=sets.num_workers, shuffle=True,pin_memory=True)
    test_dataset = Dataset_dual(sets, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=sets.batch_size, num_workers=sets.num_workers, shuffle=False, pin_memory=False)


    # settings
    print("Current setting is:")
    print(sets)
    print("\n\n")

    saveauc, saveacc = 0.5, 55
    for epoch in range(sets.n_epochs):
        print(time.strftime('[ %m-%d %H:%M:%S', time.localtime(time.time())), end=' ')
        print("epoch={} lr={}]".format(epoch, scheduler.get_last_lr()[0]))

        # training
        train_loss, train_acc, train_auc, train_F1_socre, sen1, spe1 = train(train_loader)
        print("Train: loss = {:.4f}, acc = {:.2f}%, auc = {:.4f}, F1_score = {:.4f}, sen = {:.4f}, spe = {:.4f}"
              .format(train_loss, train_acc, train_auc, train_F1_socre, sen1, spe1))

        # testing
        if train_acc > 60:
            test_loss, test_acc, test_auc, test_F1_socre, sen2, spe2 = test(test_loader)
            print("Test : loss = {:.4f}, acc = {:.2f}%, auc = {:.4f}, F1_score = {:.4f}, sen = {:.4f}, spe = {:.4f}"
                  .format(test_loss, test_acc, test_auc, test_F1_socre,sen2, spe2))

            # save model
            if test_auc > saveauc or test_acc > saveacc:
                saveauc, saveacc, savesen, savespe = int(test_auc * 100) / 100, int(test_acc * 100) / 100, int(
                    sen2 * 100) / 100, int(spe2 * 100) / 100
                model_save_path = '{}/epoch{}_acc{}_auc{}_sen{}_spe{}_{}.pth'\
                    .format(sets.save_path, epoch, saveauc, saveacc, savesen, savespe, sets.classes)
                torch.save(net.state_dict(), model_save_path)
        else:
            test_loss, test_acc, test_auc, test_F1_socre, sen2, spe2 = 0, 0, 0, 0, 0, 0
            print("train_acc < 0.6")


        # save results to excel
        datalist = [epoch, scheduler.get_last_lr()[0],
                    train_loss, train_acc, train_auc, train_F1_socre, sen1, spe1,
                    test_loss, test_acc, test_auc, test_F1_socre, sen2, spe2]
        for i in range(0, 14):
            sheet.write(epoch + 2, i, datalist[i])
        file.save(file_name)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    # settting
    sets = parse_opts_dual()

    # getting model
    torch.manual_seed(sets.manual_seed)
    net, parameters = generate_model(sets)
    print(net)

    # optimizer
    if sets.pretrain_path:
        params = [
            {'params': parameters['base_parameters'], 'lr': sets.learning_rate},
            {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}
        ]
    else:
        params = [{'params': parameters, 'lr': sets.learning_rate}]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200, 350, 500, 700], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    weight = torch.from_numpy(np.array(sets.w)).float().cuda()
    lossfunction_cross = nn.CrossEntropyLoss(weight=weight)


   # save
   #  model_save_path = sets.save_path
    if not os.path.exists(sets.save_path):
        os.makedirs(sets.save_path)
    file, file_name, sheet = save_result_in_xls()


    main()
