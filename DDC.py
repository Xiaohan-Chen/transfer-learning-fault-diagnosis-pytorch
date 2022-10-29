import argparse
import os
import numpy as np
from Utils.logger import setlogger

from turtle import forward
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from Backbone import ResNet1D, MLPNet, CNN1D
from loss import MKMMD, MMDLinear, CORAL
from PreparData.CWRU import CWRUloader
import Utils.utils as utils

from tqdm import *
import warnings
import logging

# ===== Define argments =====
def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of Deep Domain Confusion networks')

    # task setting
    parser.add_argument("--log_file", type=str, default="./logs/DDC.log", help="log file path")

    # dataset information
    parser.add_argument("--datadir", type=str, default="../datasets", help="data directory")
    parser.add_argument("--source_dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--target_dataname", type=str, default="CWRU", choices=["CWRU", "PU"], help="choice a dataset")
    parser.add_argument("--s_load", type=int, default=3, help="source domain working condition")
    parser.add_argument("--t_load", type=int, default=2, help="target domain working condition")
    parser.add_argument("--s_label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="source domain label set")
    parser.add_argument("--t_label_set", type=list, default=[0,1,2,3,4,5,6,7,8,9], help="target domain label set")
    parser.add_argument("--val_rat", type=float, default=0.3, help="training-validation rate")
    parser.add_argument("--test_rat", type=float, default=0.5, help="validation-test rate")
    parser.add_argument("--seed", type=int, default="29")

    # pre-processing
    parser.add_argument("--fft", type=bool, default=False, help="FFT preprocessing")
    parser.add_argument("--window", type=int, default=128, help="time window, if not augment data, window=1024")
    parser.add_argument("--normalization", type=str, default="0-1", choices=["None", "0-1", "mean-std"], help="normalization option")
    parser.add_argument("--savemodel", type=bool, default=False, help="whether save pre-trained model in the classification task")
    parser.add_argument("--pretrained", type=bool, default=False, help="whether use pre-trained model in transfer learning tasks")

    # backbone
    parser.add_argument("--backbone", type=str, default="ResNet1D", choices=["ResNet1D", "ResNet2D", "MLPNet", "CNN1D"])
    # if   backbone in ("ResNet1D", "CNN1D"),  data shape: (batch size, 1, 1024)
    # elif backbone == "ResNet2D",             data shape: (batch size, 3, 32, 32)
    # elif backbone == "MLPNet",               data shape: (batch size, 1024)


    # optimization & training
    parser.add_argument("--num_workers", type=int, default=0, help="the number of dataloader workers")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--lr_scheduler', type=str, default='stepLR', choices=['step', 'exp', 'stepLR', 'fix'], help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.8, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='30, 120', help='the learning rate decay for step and stepLR')
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])


    args = parser.parse_args()
    return args

# ===== Build Model =====
class FeatureNet(nn.Module):
    def __init__(self, args):
        super(FeatureNet, self).__init__()
        if args.backbone == "ResNet1D":
            self.feature_net = ResNet1D.resnet18()
        elif args.backbone == "ResNet2D":
            self.model_ft = models.resnet18(pretrained=True)
            self.bottleneck = nn.Sequential(nn.Linear(self.model_ft.fc.out_features, 512), nn.ReLU(), nn.Dropout(0.5))
            self.feature_net = nn.Sequential(self.model_ft, self.bottleneck)
        elif args.backbone == "MLPNet":
            if args.fft:
                self.feature_net = MLPNet.MLPNet(num_in=512)
            else:
                self.feature_net = MLPNet.MLPNet()
        elif args.backbone == "CNN1D":
            self.feature_net = CNN1D.CNN1D()
        else:
            raise Exception("model not implement")

    def forward(self, x):
        logits = self.feature_net(x)

        return logits

class Classifier(nn.Module):
    def __init__(self, args, num_out=10):
        super(Classifier, self).__init__()
        if args.backbone in ("ResNet1D", "ResNet2D"):
            self.classifier = nn.Sequential(nn.Linear(512,num_out, nn.Dropout(0.5)))
        if args.backbone in ("MLPNet", "CNN1D"):
            self.classifier = nn.Sequential(nn.Linear(64,num_out, nn.Dropout(0.5)))

    def forward(self, logits):
        outputs = self.classifier(logits)

        return outputs

# ===== Load Data =====
def loaddata(args):
    if args.source_dataname == "CWRU":
        source_data, source_label = CWRUloader(args, args.s_load, args.s_label_set)

    source_data, source_label = np.concatenate(source_data, axis=0), np.concatenate(source_label, axis=0)
    
    if args.target_dataname == "CWRU":
        target_data, target_label = CWRUloader(args, args.t_load, args.t_label_set)

    target_data, target_label = np.concatenate(target_data, axis=0), np.concatenate(target_label, axis=0)

    source_loader, _, _ = utils.DataSplite(args, source_data, source_label)
    target_trainloader, target_valloader, target_testloader = utils.DataSplite(args, target_data, target_label)
    
    return source_loader, target_trainloader, target_valloader, target_testloader

# ===== Test the Model =====
def tester(featurenet, classifier, dataloader):
    featurenet.eval()
    classifier.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    correct_num, total_num = 0, 0
    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # compute model cotput and loss
        logtis_batch = featurenet(x_batch)
        output_batch = classifier(logtis_batch)

        pre = torch.max(output_batch.cpu(), 1)[1].numpy()
        y = y_batch.cpu().numpy()
        correct_num += (pre == y).sum()
        total_num += len(y)
    accuracy = (correct_num / total_num) * 100.0
    return accuracy


# ===== Train the Model =====
def trainer(args):
    # Consider the gpu or cpu condition
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        logging.info('using {} gpus'.format(device_count))
        assert args.batch_size % device_count == 0, "batch size should be divided by device count"
    else:
        warnings.warn("gpu is not available")
        device = torch.device("cpu")
        device_count = 1
        logging.info('using {} cpu'.format(device_count))
    
    # load the dataset
    source_trainloader, target_trainloader, target_valloader, target_testloader = loaddata(args)

    # load the model
    featurenet = FeatureNet(args)
    classifier = Classifier(args, num_out=len(args.t_label_set))

    # load the checkpoint
    if args.pretrained:
        if args.backbone != "ResNet2D": # pretrained ResNet2D model is downloaded from torchvision module
            if not args.fft:
                path = "./checkpoints/{}_checkpoint.tar".format(args.backbone)
            else:
                path = "./checkpoints/{}FFT_checkpoint.tar".format(args.backbone)
            featurenet.load_state_dict(torch.load(path))

    parameter_list = [{"params": featurenet.parameters(), "lr": args.lr},
                       {"params": classifier.parameters(), "lr": args.lr}]

    # Define optimizer and learning rate decay
    optimizer, lr_scheduler = utils.optimizer(args, parameter_list)

    # define loss function
    loss_cls = nn.CrossEntropyLoss()
    loss_dis = MMDLinear.MMDLinear
    #loss_dis = CORAL.CORAL_loss

    featurenet.to(device)
    classifier.to(device)

    # train
    best_acc = 0.0
    meters = {"acc_source_train":[], "acc_target_train": [], "acc_target_val": []}

    for epoch in range(args.max_epoch):
        featurenet.train()
        classifier.train()
        with tqdm(total=len(target_trainloader), leave=False) as pbar:
            for i, ((x_s_batch, y_s_batch), (x_t_batch, y_t_batch)) in enumerate(zip(source_trainloader,target_trainloader)):

                if len(y_s_batch) != len(y_t_batch):
                    break
                batch_num = x_s_batch.size(0)

                inputs = torch.cat((x_s_batch, x_t_batch), dim=0)

                # move to GPU if available
                inputs = inputs.to(device)
                s_labels = y_s_batch.to(device)
                t_labels = y_t_batch.to(device)

                # compute model cotput and loss
                logits = featurenet(inputs)
                outputs = classifier(logits)

                classification_loss = loss_cls(outputs.narrow(0, 0, s_labels.size(0)), s_labels.long())
                distance_loss = loss_dis(outputs.view(outputs.size(0),-1).narrow(0, 0, s_labels.size(0)),\
                                            outputs.view(outputs.size(0),-1).narrow(0, s_labels.size(0), s_labels.size(0)))
                loss = classification_loss + distance_loss 

                # clear previous gradients, compute gradients
                optimizer.zero_grad()
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

                # evaluate
                # training accuracy
                acc_source_train = utils.accuracy(outputs.narrow(0, 0, batch_num), s_labels)
                acc_target_train = utils.accuracy(outputs.narrow(0, batch_num, batch_num), t_labels)              

                pbar.update()
        
        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        val_acc = tester(featurenet, classifier, target_valloader)
        if val_acc > best_acc:
            best_acc = val_acc
            if args.savemodel:
                utils.save_model(featurenet, args)
        
        logging.info("Epoch: {:>3}/{}, loss_cls: {:.4f}, loss: {:.4f}, source_train_acc: {:>6.2f}%, target_train_acc: {:>6.2f}%, target_val_acc: {:>6.2f}%".format(\
                epoch+1, args.max_epoch, classification_loss, loss, acc_source_train, acc_target_train, val_acc))
        meters["acc_source_train"].append(acc_source_train)
        meters["acc_target_train"].append(acc_target_train)
        meters["acc_target_val"].append(val_acc)

    logging.info("Best accuracy: {:.4f}".format(best_acc))
    utils.save_log(meters, "./logs/DDC_{}_{}_meters.pkl".format(args.backbone, args.max_epoch))

    logging.info("="*15+"Done!"+"="*15)

if __name__ == "__main__":

    args = parse_args()

    # set the logger
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    setlogger(args.log_file)

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
    
    trainer(args)