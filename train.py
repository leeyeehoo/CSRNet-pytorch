import os
import time
import json
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from model import CSRNet
from utils import save_checkpoint
import dataset

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CSRNet')
    parser.add_argument('train_json', help='path to train json')
    parser.add_argument('test_json', help='path to test json')
    parser.add_argument('--pre', default=None, type=str, help='path to the pretrained model')
    parser.add_argument('gpu', help='GPU id to use.')
    parser.add_argument('task', help='task id to use.')
    return parser.parse_args()


class AverageMeter:
    """
    Computes and stores the average and current value
    """

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

def adjust_learning_rate(optimizer, epoch, steps, scales, original_lr):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = original_lr
    for step, scale in zip(steps, scales):
        if epoch >= step:
            lr *= scale
        else:
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main():
    args = parse_args()
    best_prec1 = 1e6
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size = 8
    args.momentum = 0.95
    args.decay = 5e-4
    args.start_epoch = 0
    args.epochs = 100
    args.steps = [-1, 1, 100, 150]
    args.scales = [1, 1, 1, 1]
    args.workers = 1
    args.seed = time.time()
    args.print_freq = 30

    with open(args.train_json, 'r') as outfile:
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    model = CSRNet().cuda()
    criterion = nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.pre and os.path.isfile(args.pre):
        print(f"=> loading checkpoint '{args.pre}'")
        checkpoint = torch.load(args.pre)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint '{args.pre}' (epoch {checkpoint['epoch']})")
    elif args.pre:
        print(f"=> no checkpoint found at '{args.pre}'")

    for epoch in range(args.start_epoch, args.epochs):
        args.lr = adjust_learning_rate(optimizer, epoch, args.steps, args.scales, args.original_lr)
        train(train_list, model, criterion, optimizer, epoch, args)
        prec1 = validate(val_list, model, criterion)

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(f' * best MAE {best_prec1:.3f}')
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task)

def train(train_list, model, criterion, optimizer, epoch, args):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    transform = transforms.Compose([transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list, shuffle=True, transform=transform, train=True, seen=model.seen, batch_size=args.batch_size, num_workers=args.workers),
        batch_size=args.batch_size)

    print(f'epoch {epoch}, processed {epoch * len(train_loader.dataset)} samples, lr {args.lr:.10f}')

    end = time.time()
    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img, target = img.cuda(), target.float().unsqueeze(0).cuda()

        output = model(img)
        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tData {data_time.val:.3f} ({data_time.avg:.3f})\tLoss {losses.val:.4f} ({losses.avg:.4f})')

def validate(val_list, model, criterion):
    print('begin test')
    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list, shuffle=False, transform=transform, train=False), batch_size=8)
    mae = sum(abs(output.data.sum() - target.sum().float().cuda()) for img, target in test_loader)
    mae /= len(test_loader)
    print(f' * MAE {mae:.3f}')
    return mae


if __name__ == '__main__':
    main()
    