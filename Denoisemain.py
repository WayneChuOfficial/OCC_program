import argparse
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import  DataLoader
from utils import NativeScalerWithGradNormCount as NativeScaler
import torch.nn.functional as F
from Denoisedataset import MyDataset
from engine import train_one_epoch
import argparse
import os
import utils
import time,datetime
from NAFNet import NAFNet
from loss import MS_SSIM_L1_LOSS

def get_args_parser():
    parser = argparse.ArgumentParser('Denoise training(and evaluation) script', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int,help='batch size when the model receive data')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--update_freq', default=10, type=int,help='gradient accumulation steps')

    # Model parameters
    #parser.add_argument('--model', default='SampleNet1', type=str, metavar='MODEL',help='Name of model to train')

    # Optimization parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='Optimizer',help='Optimizer (default: "sgd/adaw"')
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


    path1 = 'CCPD2020/ccpd_green/'
    path2 = './occ_noise'
    # Dataset parameters
    parser.add_argument('--data_path', default=path1, type=str,
                        help='dataset path')
    parser.add_argument('--noise_path', default=path2, type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='/root/tf-logs/Denoise',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=41, type=int)

    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', type=bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    return parser

class PSNRLoss(torch.nn.Module):
  def __init__(self, loss_weight=1.0, reduction='mean'):
    super(PSNRLoss, self).__init__()
    assert reduction == 'mean'
    self.loss_weight = loss_weight
    self.scale = 10 / np.log(10)

  def forward(self, pred, target):
    return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self,):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-12
 
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target)**2 + self.eps))
class VaeLoss(torch.nn.Module):
    def __init__(self,beta = 5,kld_weight = 0.005,recon='mse'):
        super(VaeLoss, self).__init__()
        self.beta = beta
        self.kld_weight = kld_weight
        if recon =='mse':
            self.recon = F.mse_loss()
        else:
            self.recon = L1_Charbonnier_loss()
    def forward(self,pred,target):
        recons = pred[0]
        input = pred[1]
        mu = pred[2]
        log_var = pred[3]
        recons_loss =self.recon(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return recons_loss + self.beta*self.kld_weight*kld_loss

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


    transform = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.ToTensor(),
        ])
    
    train_dataset = MyDataset(args.data_path+'train', args.noise_path,transform=transform,makenoise = False,random_crop=True,crop_size=256,mixture=True,FFT=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=args.pin_mem)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    #model = MemNet(3,20,6,6)
    model = NAFNet(3,32,12,[2, 2, 4, 8],[2, 2, 2, 2])
    #model.load_state_dict(torch.load('NAFNet4denoise.pth'))
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    #total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(train_dataset))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    opt_args = dict(lr=args.lr, weight_decay=args.weight_decay,betas=[0.9,0.9])
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),**opt_args)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),**opt_args)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),momentum=args.momentum,**opt_args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = 100 ,eta_min = args.min_lr)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used
    #criterion = torch.nn.MSELoss()
    criterion = MS_SSIM_L1_LOSS()
    print("criterion = %s" % str(criterion))
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(0, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer,
            device, epoch, loss_scaler, args.clip_grad,
            log_writer=log_writer,  start_steps=epoch * num_training_steps_per_epoch,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp,schedule=scheduler)
        scheduler.step()
        torch.save(model.state_dict(),'NAFNet4denoise.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Denoise training(and evaluation) script', parents=[get_args_parser()])
    #parser = get_args_parser()
    arglist = ['--batch_size','2','--epochs','50','--update_freq','10',
              '--opt','adamw','--opt_eps','1e-8','--opt_betas','0','--clip_grad','0.4','--opt_betas','0.9',
              '--momentum','0.9','--weight_decay','1e-4','--lr','1e-4','--min_lr','1e-7',
              '--warmup_epochs','20','--warmup_steps','1','--use_amp','True']
    args = parser.parse_args(args=arglist)
    main(args)