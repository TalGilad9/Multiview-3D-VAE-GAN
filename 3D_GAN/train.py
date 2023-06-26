import torch
from torch import optim
from torch import  nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

from utils import ShapeNetDataset, var_or_cuda
from model import _G, _D
from lr_sh import  MultiStepLR


def save_train_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['train_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['train_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyparam_list = [("model", args.model_name),
                    ("cube", args.cube_len),
                    ("n_epochs", args.n_epochs),
                    ("bs", args.batch_size),
                    ("g_lr", args.g_lr),
                    ("d_lr", args.d_lr),
                    ("z", args.z_dis),
                    ("bias", args.bias),
                    ("z_size", args.z_size),
                    ("sl", args.soft_label),]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)

    # for using tensorboard
    if args.use_tensorboard:
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        import tensorflow as tf
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.output_dir +args.log_dir + log_param)

        image_saved_path = args.output_dir + '/' + args.model_name + '/' + args.log_dir + '/images'
        model_saved_path = args.output_dir + '/' + args.model_name + '/' + args.log_dir + '/models'

        if not os.path.exists(image_saved_path):
            os.makedirs(image_saved_path)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)

    image_path = args.output_dir + args.image_dir + log_param
    if not os.path.exists(image_path):
        os.makedirs(image_path)
        
    # datset define
    dsets_path = args.input_dir + args.data_dir + "train/"
    dsets = ShapeNetDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model define
    D = _D(args).to(device)
    G = _G(args).to(device)

    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)

    if args.lrsh:
        D_scheduler = MultiStepLR(D_solver, milestones=[500, 1000])

    criterion = nn.BCELoss()
    criterionMSE = nn.MSELoss()
    pickle_path = "." + args.pickle_dir + log_param
    #read_pickle(pickle_path, G, G_solver, D, D_solver)
        
    loss_G_vector = []
    loss_G_recon_vector = []
    loss_D_vector = []
    
    running_loss_G = 0.0
    running_loss_D = 0.0
    running_loss_adv_G = 0.0
    
    itr_train = 0
    
    for epoch in range(args.n_epochs):
        for i, X in enumerate(dset_loaders):
            itr_train += 1
            
            X = X.to(device)
            
            batchX = X.size(0)
            
            #if X.size()[0] != int(args.batch_size):
                #print("batch_size != {} drop last incompatible batch".format(int(args.batch_size)))
            #    continue
            
            
            
            real_labels = torch.ones(batchX).to(device)
            fake_labels = torch.zeros(batchX).to(device)
            
            if args.soft_label:
                real_labels = torch.Tensor(batchX).uniform_(0.7, 1.2).to(device)
                fake_labels = torch.Tensor(batchX).uniform_(0, 0.3).to(device)
            
            #real_labels = var_or_cuda(torch.ones(batch))
            #fake_labels = var_or_cuda(torch.zeros(batch))
           # 
            #if args.soft_label:
            #    real_labels = var_or_cuda(torch.Tensor(batch).uniform_(0.7, 1.2))
            #    fake_labels = var_or_cuda(torch.Tensor(batch).uniform_(0, 0.3))

            # ============= Train the discriminator =============#
          #  d_real = D(X)
          
            d_real = D(X).view(-1) 
            d_real_loss = criterion(d_real, real_labels)

            Z = generateZ(args)
            fake = G(Z)
            d_fake = D(fake).view(-1) 
            
            batchZ = fake.size(0)
            fake_labels = torch.zeros(batchZ).to(device)
            real_labels = torch.ones(batchZ).to(device)
            if args.soft_label:
                real_labels = torch.Tensor(batchZ).uniform_(0.7, 1.2).to(device)
                fake_labels = torch.Tensor(batchZ).uniform_(0, 0.3).to(device)
                
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss


            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))

            if d_total_acu <= args.d_thresh:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            # =============== Train the generator ===============#

            Z = generateZ(args)

            fake = G(Z)
            d_fake = D(fake).view(-1) 
            
            
            
            g_loss = criterion(d_fake, real_labels)
            
            f = fake.data[:batchX].squeeze().to(device)

            recon_g_loss = criterion(f, X)
              
            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()

        # =============== logging each iteration ===============#
            iteration = str(G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][0]]['step'])
            
            running_loss_G += recon_g_loss.item()# * X.size(0)
            running_loss_D += d_loss.item()# * X.size(0)
            running_loss_adv_G += g_loss.item() #* X.size(0)
            
            if args.use_tensorboard:
                loss_G = {
                    'adv_loss_G': g_loss,
                    'recon_loss_G': recon_g_loss,
                }
    
                loss_D = {
                    'adv_real_loss_D': d_real_loss,
                    'adv_fake_loss_D': d_fake_loss,
                }
    
                # if itr_val % 10 == 0 and phase == 'val':
                #     save_val_log(writer, loss_D, loss_G, itr_val)
    
                if itr_train % 10 == 0:
                    save_train_log(writer, loss_D, loss_G, itr_train)

        # =============== each epoch save model or save image ===============#
        #print(epoch)
        
        #print('Iter-{}; , D_loss : {:.4}, G_loss : {:.4}, D_acu : {:.4}, D_lr : {:.4}'.format(iteration, d_loss.item(), g_loss.item(), d_total_acu.item(), D_solver.state_dict()['param_groups'][0]["lr"]))
        
        epoch_loss_G = running_loss_G / itr_train
        epoch_loss_D = running_loss_D / itr_train
        epoch_loss_adv_G = running_loss_adv_G / itr_train
        
        #loss_G_vector.append(epoch_loss_adv_G)
        #loss_G_recon_vector.append(epoch_loss_G)
        #loss_D_vector.append(epoch_loss_D)
        
        loss_G_vector.append(g_loss.item())
        loss_G_recon_vector.append(recon_g_loss.item())
        loss_D_vector.append(d_loss.item())
        
        
        if (epoch + 1) % args.image_save_step == 0:
            #print("size:")
            #print(fake.cpu().squeeze().detach().numpy().shape)
            samples = fake.cpu().data[:8].squeeze().numpy()
        
            image_path = args.output_dir + args.image_dir + log_param
            if not os.path.exists(image_path):
                os.makedirs(image_path)
        
            SavePloat_Voxels(samples, image_path, iteration)
        
        if (epoch + 1) % args.pickle_step == 0:
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, iteration, G, G_solver, D, D_solver)
        
        if args.lrsh:
        
            try:
        
                D_scheduler.step()
        
        
            except Exception as e:
                a=1
                #print("fail lr scheduling", e)
                

              
                
                
    plt.plot(np.linspace(1, args.n_epochs, args.n_epochs), loss_G_vector, color='r',label='Generator')
    plt.plot(np.linspace(1, args.n_epochs, args.n_epochs), loss_G_recon_vector,color='b',label='Generator Real Vs Fake')
    plt.plot(np.linspace(1, args.n_epochs, args.n_epochs), loss_D_vector, color='g', label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Train')
    plt.legend()
    plt.savefig('LossTrain_' + log_param + '.png')
