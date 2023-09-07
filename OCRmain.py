import argparse
from torch.utils.data import DataLoader
from torch import nn
from loss import FocalLossManyClassification
from Licensedataset import OcrDataSet
from einops import rearrange
from tqdm import tqdm
import torch
import os
from OCRNet import OcrNet
import time,datetime,math
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('OCRNet training(and evaluation) script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,help='batch size when the model receive data')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--update_freq', default=10, type=int,help='gradient accumulation steps')

    # Model parameters
    #parser.add_argument('--model', default='SampleNet1', type=str, metavar='MODEL',help='Name of model to train')
    #parser.add_argument('--input_size', default=256, type=int,help='image input size')

    # Optimization parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='Optimizer',help='Optimizer (default: "sgd/adaw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='Epsilon',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='Beta',help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=0.4, metavar='Norm',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,help='weight decay (default: 0.05)')
    
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--use_amp', type=bool, default=True, 
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    path = '/content/drive/MyDrive/program2023/LicenseDetection/CCPD2020/ccpd_green/'
    # Dataset parameters
    parser.add_argument('--data_path', default=path, type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=41, type=int)

    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', type=bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    return parser


def main(args):
    print(args)
    device = torch.device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    model = OcrNet(70)
    #model.load_state_dict(torch.load('OCRNet_autodl.pth'))
    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    train_dataset = OcrDataSet()
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=args.pin_mem)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(train_dataset))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)
    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay)
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),**opt_args)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),**opt_args)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),momentum=args.momentum,**opt_args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        optimizer.zero_grad()
        for i, (images, targets, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            '''生成标签'''
            e = torch.tensor([])
            for i, j in enumerate(target_lengths):
                e = torch.cat((e, targets[i][:j]), dim=0)
            targets = e.long()#shape = torch.sum(target_lengths)
            # print(targets)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            '''预测'''
            predict = model(images)
            s, n, v = predict.shape
            input_lengths = torch.full(size=(n,), fill_value=s, dtype=torch.long)

            """计算损失，预测值需，log_softmax处理，网络部分不应该softmax"""
            loss = criterion(predict.log_softmax(2), targets, input_lengths, target_lengths)
            loss_value = loss.item()
            if not math.isfinite(loss_value): # this could trigger if using AMP
                print("Loss is {}, stopping training".format(loss_value))
                assert math.isfinite(loss_value)

            #loss /= args.update_freq
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(),args.clip_grad) 
            optimizer.step()
      
            if (i+1)%args.update_freq==0:
                #optimizer.step()
                #optimizer.zero_grad()
                print('epoch=',epoch,'[',i,'/',len(train_loader),']','total loss=',round(loss.item(),2))
            loss_sum += loss_value
        logs = f'epoch:{epoch},average loss:{loss_sum / len(train_loader)}'
        print(logs)
        scheduler.step()
        torch.save(model.state_dict(),'OCRNet_autodl.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OCRNet training(and evaluation) script', parents=[get_args_parser()])
    #parser = get_args_parser()
    arglist = ['--batch_size','256','--epochs','50','--update_freq','10',
              '--opt','adam','--opt_eps','1e-8','--opt_betas','0','--clip_grad','0.4',
              '--momentum','0.99','--weight_decay','1e-4','--lr','1e-5','--min_lr','1e-6',
              '--warmup_epochs','20','--warmup_steps','1','--use_amp','True']
    args = parser.parse_args(args=arglist)
    main(args)