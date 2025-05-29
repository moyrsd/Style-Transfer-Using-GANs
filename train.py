import torch
from Dataset import horse_zebra_dataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import Config
from torchvision.utils import save_image
from discriminator import discriminator
from generator import generator
from tqdm import tqdm
import numpy as np


def train(disc_h, disc_z, gen_z, gen_h, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)
    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(Config.device)
        horse = horse.to(Config.device)
        with torch.cuda.amp.autocast():
            fake_horse = gen_h(zebra)
            d_h_real = disc_h(horse)
            d_h_fake = disc_h(fake_horse.detach())
            d_h_real_loss = mse(d_h_real, torch.ones_like(d_h_real))
            d_h_fake_loss = mse(d_h_fake, torch.zeros_like(d_h_fake))
            d_h_loss = d_h_fake_loss+d_h_real_loss

            fake_zebra = gen_z(horse)
            d_z_real = disc_z(zebra)
            d_z_fake = disc_z(fake_zebra.detach())
            d_z_real_loss = mse(d_z_real, torch.ones_like(d_z_real))
            d_z_fake_loss = mse(d_z_fake, torch.zeros_like(d_z_fake))
            d_z_loss = d_z_fake_loss+d_z_real_loss

            d_loss = (d_h_loss+d_z_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            d_h_fake = disc_h(fake_horse)
            d_z_fake = disc_z(fake_zebra)
            loss_g_z = mse(d_z_fake, torch.ones_like(d_z_fake))
            loss_g_h = mse(d_h_fake, torch.ones_like(d_h_fake))

            cycle_horse = gen_h(fake_zebra)
            cycle_zebra = gen_z(fake_horse)
            cycle_zebra_loss = L1(zebra, cycle_zebra)
            cycle_horse_loss = L1(horse, cycle_horse)

            g_loss = loss_g_z+loss_g_h+cycle_horse_loss * \
                Config.lambda_cycle+cycle_zebra_loss*Config.lambda_cycle

        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 50 == 0:
            save_image(fake_horse*0.5+0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"saved_images/zebra_{idx}.png")


def main():
    disc_h = discriminator(in_channels=3).to(Config.device)
    disc_z = discriminator(in_channels=3).to(Config.device)
    gen_z = generator(in_channels=3).to(Config.device)
    gen_h = generator(in_channels=3).to(Config.device)

    opt_disc = optim.Adam(list(disc_h.parameters())+list(disc_z.parameters()),
                          lr=Config.learning_rate,
                          betas=(0.5, 0.999)
                          )

    opt_gen = optim.Adam(list(gen_h.parameters())+list(gen_z.parameters()),
                         lr=Config.learning_rate,
                         betas=(0.5, 0.999)
                         )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if Config.load_model:
        load_checkpoint(Config.checkpoint_gen_horse, gen_h,
                        opt_gen, Config.learning_rate)
        load_checkpoint(Config.checkpoint_gen_zebra, gen_z,
                        opt_gen, Config.learning_rate)
        load_checkpoint(Config.checkpoint_dis_horse, disc_h,
                        opt_disc, Config.learning_rate)
        load_checkpoint(Config.checkpoint_dis_zebra, disc_z,
                        opt_disc, Config.learning_rate)
    train_dataset = horse_zebra_dataset(root_horse=Config.train_dir+"/horse",
                                        root_zebra=Config.train_dir+"/zebra", transform=Config.transforms)
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=True)
    test_dataset = horse_zebra_dataset(root_horse=Config.test_dir+"/horse",
                                       root_zebra=Config.test_dir+"/zebra", transform=Config.transforms)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    for epoch in range(Config.epochs):
        print("Epoch is :{}".format(epoch))
        train(disc_h, disc_z, gen_z, gen_h,
              train_loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)

        if Config.save_model:
            save_checkpoint(
                gen_h, opt_gen, filename=Config.checkpoint_gen_horse)
            save_checkpoint(
                gen_z, opt_gen, filename=Config.checkpoint_gen_zebra)
            save_checkpoint(disc_h, opt_disc,
                            filename=Config.checkpoint_dis_horse)
            save_checkpoint(disc_z, opt_disc,
                            filename=Config.checkpoint_dis_zebra)


if __name__ == "__main__":
    main()
