import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from EasyLossUtil.global_utils import ParamsParent, checkDir, formatSeconds
from EasyLossUtil.saveTensor2Img import save_image
from EasyLossUtil.easyLossUtil import EasyLossUtil
from EasyLossUtil.quickAverageMethod import QuickAverageMethod
from get_dataloader import getDataloader
from get_fid_score import get_fid_score
from model.conv_model import ConvDiscriminator, ConvGenerator

from workbench_utils import denormalize, prepareEnv, get_rootdir_path
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from tqdm import tqdm
import numpy as np


def build_src_ckp_path(ckp_time_stamp, dataset_name):
    assert ckp_time_stamp is not None
    assert dataset_name is not None
    # 当前文件的路径
    current_path = os.path.abspath(__file__)
    # 当前文件所在的目录
    root_dir = os.path.dirname(current_path)
    # the path for saving model checkpoints
    ckp_root_path = os.path.join(root_dir, "ckp", ckp_time_stamp)
    checkDir(ckp_root_path)
    g_ckp_path = os.path.join(
        ckp_root_path,
        f"g_{dataset_name}_" + ckp_time_stamp
    )
    d_ckp_path = os.path.join(
        ckp_root_path,
        f"d_{dataset_name}_" + ckp_time_stamp
    )
    # the directory for samples during traing procedure
    sample_root_path = os.path.join(
        root_dir,
        "sample",
        ckp_time_stamp
    )
    checkDir(sample_root_path)
    # the path for saving loss data
    loss_root_path = os.path.join(root_dir, "loss", "loss_" + ckp_time_stamp)
    return g_ckp_path, d_ckp_path, sample_root_path, loss_root_path


class TrainConvGanParams(ParamsParent):
    dataset_name = "mnist"

    # img_size = 224
    img_size = 64
    in_c = 3
    norm_type="n2"

    batch_size = 128
    # batch_size = 64
    # batch_size = 48
    # batch_size = 32
    # batch_size = 24
    # batch_size = 16
    g_lr = 1e-6
    d_lr = 1e-6
    # g_lr = 1e-5
    # d_lr = 1e-5
    latent_dim = 256
    # total_epochs = 8000
    # total_epochs = 500
    # total_epochs = 800
    # total_epochs = 1500
    total_epochs = 2000
    delay = 1

    visualizing_sample_num = 4
    # fid_score_sampleNum = batch_size * 3
    fid_score_sampleNum = batch_size * 10

    # 是否使用进度条
    use_tqdm = False
    # use_tqdm = True
    # 是否快速调试
    quick_debug = False
    # quick_debug = True

    ckp_time_stamp = "test"   # 实验 test
    g_ckp_path, d_ckp_path, sample_root_path, loss_root_path = build_src_ckp_path(
        ckp_time_stamp,
        dataset_name
    )

    # nohup python -u main.py > ./log/test.txt 2>&1 &
    # 883296


# the function for initializing model
def init_model(net:nn.Module, ckp_path=None):
    if ckp_path is not None:
        print(f"The model has been loaded:{ckp_path}")
        net.load_state_dict(torch.load(ckp_path))
    # else:
    #     net.apply(init_weights)
    net.cuda()
    return net
    

def get_generator(latent_dim, dataset_name, ckp_time_stamp=None):
    g = ConvGenerator(latent_dim)
    if ckp_time_stamp is None:
        g = init_model(g)
    else:
        ckp_path = os.path.join(
            get_rootdir_path, "ckp",
            ckp_time_stamp,
            "g_" + dataset_name + "_" + ckp_time_stamp
        )
        g = init_model(g, ckp_path)
    return g


def get_discriminator(dataset_name, ckp_time_stamp=None):
    d = ConvDiscriminator()
    if ckp_time_stamp is None:
        d = init_model(d)
    else:
        ckp_path = os.path.join(
            get_rootdir_path, "ckp",
            ckp_time_stamp,
            "d_" + dataset_name + "_" + ckp_time_stamp
        )
        d = init_model(d, ckp_path)
    return d



@torch.no_grad()
def compute_fid(params:TrainConvGanParams, train_dataloader:DataLoader, generator:nn.Module, fixed_imgs:torch.Tensor):
    generator.eval()

    # 1. generate fake imgs
    fake_img = None
    for index in range(params.fid_score_sampleNum // params.batch_size):
        # generate noises
        noise = torch.randn((params.batch_size, params.latent_dim)).cuda()
        # inference
        current_fake_img = generator(noise)
        # save data
        if fake_img is None:
            fake_img = current_fake_img
        else:
            fake_img = torch.cat((fake_img, current_fake_img), dim=0)

    # 2. 调用函数计算fid值
    fid_score = get_fid_score(fixed_imgs, fake_img, device="cuda")

    del fake_img
    return float(fid_score)


def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1)).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = torch.ones((real_samples.shape[0], 1), requires_grad=False).cuda()
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return 10 * gradient_penalty


def train_procedure(
        params:TrainConvGanParams,
        generator:nn.Module, discriminator:nn.Module,
        train_dataloader:DataLoader,
        fixed_noises:torch.Tensor,
        fixed_fid_score_imgs:torch.Tensor,
        fixed_real_vis_imgs:torch.Tensor
    ):
    """
    train conv gan (wgan-gp)
    Args:
        params (TrainConvGanParams): all the training hyper params
        generator (nn.Module): 
        discriminator (nn.Module): 
        src_train_dataloader (DataLoader): training data
        fixed_noises (torch.Tensor): for visualizing training procedure
        fixed_imgs (torch.Tensor): for computing fid_score
        fixed_real_vis_imgs (torch.Tensor): for visulizing
    """
    # ------------------------------------------------------------
    # -- setup the optimizer, lr scheduler and criterion function
    # ------------------------------------------------------------
    # optim_g = optim.AdamW(
    #     generator.parameters(),
    #     lr=params.g_lr
    # )
    # optim_d = optim.AdamW(
    #     discriminator.parameters(),
    #     lr=params.d_lr,
    #     betas=(0.5, 0.999)
    # )
    optim_g = optim.Adam(
        generator.parameters(),
        lr=params.g_lr,
        betas=(0.5, 0.999)
    )
    optim_d = optim.Adam(
        discriminator.parameters(),
        lr=params.d_lr,
        betas=(0.5, 0.999)
    )
    # optim_g = optim.RMSprop(
    #     generator.parameters(),
    #     lr=params.g_lr
    # )
    # optim_d = optim.RMSprop(
    #     discriminator.parameters(),
    #     lr=params.d_lr
    # )
    lr_scheduler_g = optim.lr_scheduler.StepLR(optim_g, step_size=10, gamma=0.99)
    lr_scheduler_d = optim.lr_scheduler.StepLR(optim_d, step_size=10, gamma=0.99)
    bce_criterion = nn.BCELoss()
    # ------------------------------------------
    # -- setup loss util
    # ------------------------------------------
    loss_name_list = ["loss_g", "loss_d", "fid_score"]
    avgUtil = QuickAverageMethod(loss_name_list=loss_name_list[0:2])
    lossUtil = EasyLossUtil(
        loss_name_list,
        loss_root_dir=params.loss_root_path
    )
    # ---------------------
    # --- train model
    # ---------------------
    min_fid_score = None
    for epoch in range(params.total_epochs):
        start_time = time.time()
        # ---------------------
        # --- setup progress bar
        # ---------------------
        if params.use_tqdm:
            iter_object = tqdm(
                train_dataloader, 
                ncols=120
            )
        else:
            iter_object = train_dataloader
        # processing all data
        for iter_idx, (imgs, _) in enumerate(iter_object):
            # quickly debug
            if params.quick_debug:
                if iter_idx > 3:
                    break
            # send the data to gpu
            imgs = imgs.cuda()
            # generate noise from Gaussian distribution
            z = torch.randn((imgs.size(0), params.latent_dim)).cuda()

            # valid and fake flags for training
            valid_flags = torch.ones((imgs.size(0), 1), requires_grad=False).cuda()
            fake_flags = torch.zeros((imgs.size(0), 1), requires_grad=False).cuda()

            # ---------------------
            # --- train d
            # ---------------------
            # setup model
            generator.eval()
            discriminator.train()
            # generator inference
            fake_imgs = generator(z)
            # compute loss

            # -- dcgan loss
            # d should successfully output valid for the real imgs as much as it can
            loss_d_real = bce_criterion(discriminator(imgs), valid_flags)
            # d should successfully output fake for the fake imgs as much as it can
            loss_d_fake = bce_criterion(discriminator(fake_imgs), fake_flags)
            # sum
            loss_d = (loss_d_real + loss_d_fake) / 2

            # # -- wgan loss
            # loss_d = -torch.mean(discriminator(imgs)) + torch.mean(discriminator(fake_imgs))

            # # -- wgan gp loss
            # loss_gp = compute_gradient_penalty(discriminator, imgs, fake_imgs)
            # loss_d = -torch.mean(discriminator(imgs)) + torch.mean(discriminator(fake_imgs)) + loss_gp

            # optimize
            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()
            # add loss_d to loss util
            avgUtil.append(
                loss_name=loss_name_list[1],
                value=loss_d.item()
            )
            # end trian d

            # # Clip weights of discriminator
            # clip_value = 0.01
            # for p in discriminator.parameters():
            #     p.data.clamp_(-clip_value, clip_value)

            # ---------------------
            # --- train G
            # ---------------------
            train_g_flag = 0
            if iter_idx % params.delay == 0:
                train_g_flag = 1
                # setup model
                generator.train()
                discriminator.eval()
                # generator inference
                fake_imgs = generator(z)

                # compute loss

                # -- dcgan loss
                # g should try to get valid falgs from discriminator from as much as possible
                loss_g = bce_criterion(discriminator(fake_imgs), valid_flags)

                # # -- wgan loss
                # loss_g = -torch.mean(discriminator(fake_imgs))

                # optimize
                optim_g.zero_grad()
                loss_g.backward()
                optim_g.step()
                # add loss g to loss util
                avgUtil.append(
                    loss_name=loss_name_list[0],
                    value=loss_g.item()
                )
                # end train g
            # ---------------------
            # --- log on progress bar
            # ---------------------
            if train_g_flag == 1 and params.use_tqdm:
                iter_object.set_postfix_str(f"loss_g:{loss_g:.4f}, loss_d:{loss_d:.4f}")
        # end one epoch
        # ---------------------
        # -- save model ckp
        # ---------------------
        fid_score = compute_fid(params, train_dataloader, generator, fixed_fid_score_imgs)
        # print(f"type fid_score:{type(fid_score)}")   # <class 'numpy.float64'>
        if min_fid_score is None or min_fid_score > fid_score:
                min_fid_score = fid_score
                print("Save the best model...")
                torch.save(generator.state_dict(), params.g_ckp_path)
                torch.save(discriminator.state_dict(), params.d_ckp_path)
        # ---------------------
        # -- lr scheduler
        # ---------------------
        lr_scheduler_g.step()
        lr_scheduler_d.step()
        # ---------------------
        # -- training log
        # ---------------------
        avg_loss_g, avg_loss_d = avgUtil.getAllAvgLoss()
        # print log
        print(
            f'[{epoch}/{params.total_epochs}]\n'
            f'epoch_loss_g:{avg_loss_g:.4f}, '
            f'epoch_loss_d:{avg_loss_d:.4f}\n'
            f'fid_score:{fid_score:.4f}'
        )
        # clear avgUtil
        avgUtil.clearAllData()
        # visualizing all loss data
        # 前面几个epoch的数据可能离群, 导致折线图不好观察
        if epoch > -1: 
            lossUtil.append(
                loss_name=loss_name_list,
                loss_data=[
                    avg_loss_g,
                    avg_loss_d,
                    fid_score
                ]
            )
        else:
            lossUtil.append(
                loss_name=loss_name_list,
                loss_data=[
                    0,
                    0,
                    0
                ]
            )
        lossUtil.autoSaveFileAndImage()
        # ------------------------------
        # -- observe generated samples
        # ------------------------------
        with torch.no_grad():
            # setup model
            generator.eval()
            # inference
            fake_imgs = generator(fixed_noises)
            fake_imgs = denormalize(
                fake_imgs, norm_type=params.norm_type
            )
            # concat the real images
            vis_images = torch.cat((fake_imgs, fixed_real_vis_imgs), dim=0)
            # visualizing
            postfix = "png"
            save_image(
                vis_images,
                os.path.join(params.sample_root_path, f"sample_{epoch}.{postfix}"),
                nrow=4, padding=5
            )
        # end generate samples
        # end time for this epoch
        end_time = time.time()
        # 本轮次花费时间
        epoch_time = end_time - start_time
        print(formatSeconds(seconds=epoch_time, targetStr='epoch time'))
        print()
    # end all epochs
    print('---finish train convgan procedure---')


def trainConvGanMain():
    # ------------------------------
    # -- init the env
    # ------------------------------
    prepareEnv()
    # ------------------------------
    # -- init convgan train params
    # ------------------------------
    params = TrainConvGanParams()
    print(params)
    # ------------------------------
    # -- load data
    # ------------------------------
    print("=== load data ===")
    train_dataloader = getDataloader(
        params.dataset_name, 
        phase="train", 
        img_size=params.img_size, 
        batch_size=params.batch_size,
        norm_type=params.norm_type
    )
    # generate fixed noise for visualizing the training result
    fixed_noises = torch.randn((params.visualizing_sample_num, params.latent_dim)).cuda()
    # prepare the fixed imgs for computing the fid score
    fixed_fid_score_imgs = None
    for i in range(params.fid_score_sampleNum):
        current_imgs = train_dataloader.dataset[i][0].unsqueeze(0)
        if fixed_fid_score_imgs is None:
            fixed_fid_score_imgs = current_imgs
        else:
            fixed_fid_score_imgs = torch.cat((fixed_fid_score_imgs, current_imgs), dim=0)
    # prepare the fixed imgs for visulizing
    fixed_real_vis_imgs = denormalize(
        fixed_fid_score_imgs[0:params.visualizing_sample_num], 
        norm_type=params.norm_type
    ).cuda()

    # ------------------------------
    # -- get generator and discriminator
    # ------------------------------
    print("=== init model ===")
    generator = get_generator(params.latent_dim, params.dataset_name)
    discriminator = get_discriminator(params.dataset_name)
    # ------------------------------
    # -- train model
    # ------------------------------
    train_procedure(
        params,
        generator, discriminator,
        train_dataloader,
        fixed_noises,
        fixed_fid_score_imgs,
        fixed_real_vis_imgs
    )