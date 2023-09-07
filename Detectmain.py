import argparse
from torch.utils.data import DataLoader
from torch import nn
from loss import FocalLossManyClassification
from Licensedataset import DetectDataset
from einops import rearrange
from tqdm import tqdm
import torch
import os
from WpodNet import WpodNet
import time,datetime,math
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('WpodNet training(and evaluation) script', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,help='batch size when the model receive data')
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

    path = 'CCPD2020/ccpd_green/train'
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

    model = WpodNet()
    model.load_state_dict(torch.load('WpodNet_ccpd2020_87.pth'))
    model.to(device)
    criterion1 = nn.L1Loss()
    criterion2 = nn.CrossEntropyLoss()
    train_dataset = DetectDataset(args.data_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=args.pin_mem)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 100 ,eta_min = args.min_lr)
    print("Start training for %d epochs" % args.epochs)

    def count_loss(predict, target):
        condition_positive = target[:, :, :, 0] == 1
        condition_negative = target[:, :, :, 0] == 0

        predict_positive = predict[condition_positive]
        predict_negative = predict[condition_negative]

        target_positive = target[condition_positive]
        target_negative = target[condition_negative]
        #print(target_positive.shape)
        n, v = predict_positive.shape
        if n > 0:
            loss_c_positive = criterion2(predict_positive[:, 0:2], target_positive[:, 0].long())
        else:
            loss_c_positive = 0
        loss_c_nagative = criterion2(predict_negative[:, 0:2], target_negative[:, 0].long())
        loss_c = loss_c_nagative + loss_c_positive

        if n > 0:
            affine = torch.cat(
                (
                    predict_positive[:, 2:3],
                    predict_positive[:,3:4],
                    predict_positive[:,4:5],
                    predict_positive[:,5:6],
                    predict_positive[:,6:7],
                    predict_positive[:,7:8]
                ),
                dim=1
            )
            # print(affine.shape)
            # exit()
            trans_m = affine.reshape(-1, 2, 3)
            unit = torch.tensor([[-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]]).transpose(0, 1).to(
                    trans_m.device).float()
                # print(unit)
            point_pred = torch.einsum('n j k, k d -> n j d', trans_m, unit)
            point_pred = rearrange(point_pred, 'n j k -> n (j k)')
            loss_p = criterion1(point_pred, target_positive[:, 1:])
        else:
            loss_p = 0
        # exit()
        return loss_c, loss_p


    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        optimizer.zero_grad()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            predict = model(images)
            loss_c, loss_p = count_loss(predict, labels)
            loss = loss_c + loss_p
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
                print('epoch=',epoch,'[',i,'/',len(train_loader),']','total loss=',round(loss.item(),4),
                    'loss_c:',round(loss_c.item(),4),'loss_p:',round(loss_p.item(),4))
            #if i % 100 == 0:
            #    torch.save(self.net.state_dict(), config.weight)
            loss_sum += loss_value
        logs = f'epoch:{epoch},average loss:{loss_sum / len(train_loader)}'
        print(logs)
        scheduler.step()
        torch.save(model.state_dict(),'WpodNet_autodl.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Denoise training(and evaluation) script', parents=[get_args_parser()])
    #parser = get_args_parser()
    arglist = ['--batch_size','8','--epochs','10','--update_freq','10',
              '--opt','adamw','--opt_eps','1e-8','--opt_betas','0','--clip_grad','0.2','--opt_betas','0.9',
              '--momentum','0.9','--weight_decay','1e-4','--lr','1e-4','--min_lr','1e-7',
              '--warmup_epochs','20','--warmup_steps','1','--use_amp','True']
    args = parser.parse_args(args=arglist)
    main(args)