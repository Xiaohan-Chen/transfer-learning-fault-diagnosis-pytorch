import logging
import torch
import pickle
import torch.optim as optim
from sklearn.model_selection import train_test_split

def accuracy(outputs, labels):
    """
    Compute the accuracy
    outputs, labels: (tensor)
    return: (float) accuracy in [0, 100]
    """
    pre = torch.max(outputs.cpu(), 1)[1].numpy()
    y = labels.data.cpu().numpy()
    acc = ((pre == y).sum() / len(y)) * 100
    return acc

def save_log(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_model(model, args):
    if not args.fft:
        torch.save(model.state_dict(), "./checkpoints/{}_checkpoint.tar".format(args.backbone))
    else:
        torch.save(model.state_dict(), "./checkpoints/{}FFT_checkpoint.tar".format(args.backbone))

def DataSplite(args, data, label):
    """
    split the data and lebel and transform the narray type to tensor type
    """
    data_train, data_val, label_train, label_val = train_test_split(data, label, test_size = args.val_rat, random_state=args.seed)
    data_val, data_test, label_val, label_test = train_test_split(data_val, label_val, test_size= args.test_rat)

    # numpy to tensor
    data_train = torch.from_numpy(data_train).float()
    data_val = torch.from_numpy(data_val).float()
    data_test = torch.from_numpy(data_test).float()
    label_train = torch.from_numpy(label_train).float()
    label_val = torch.from_numpy(label_val).float()
    label_test = torch.from_numpy(label_test).float()

    # logging the data shape
    logging.info("training data/label shape: {},{}".format(data_train.size(), label_train.size()))
    logging.info("validation data/label shape: {},{}".format(data_val.size(), label_val.size()))
    logging.info("test data/label shape: {},{}".format(data_test.size(), label_test.size()))

    # build the dataloader
    train = torch.utils.data.TensorDataset(data_train, label_train)
    val = torch.utils.data.TensorDataset(data_val, label_val)
    test = torch.utils.data.TensorDataset(label_test, label_test)

    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size, \
        shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, \
        shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader

def optimizer(args, parameter_list):
    # define optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(parameter_list, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(parameter_list, lr=args.lr)
    else:
        raise Exception("optimizer not implement")

    # Define the learning rate decay
    if args.lr_scheduler == 'step':
        steps = [int(step) for step in args.steps.split(',')]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
    elif args.lr_scheduler == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    elif args.lr_scheduler == 'stepLR':
        steps = int(args.steps.split(",")[0])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
    elif args.lr_scheduler == 'cos':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, 0)
    elif args.lr_scheduler == 'fix':
        lr_scheduler = None
    else:
        raise Exception("lr schedule not implement")

    return optimizer, lr_scheduler
