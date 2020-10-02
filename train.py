import torch
import torch.nn as nn
from  model import Generator,Discriminator,weights_init
from preprocessdata import dataset
import argparse
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-r","--dataroot",required=True,help="root image directory")
ap.add_argument("-n","--nepochs",required=True,type = int,help="no of epochs to run")
ap.add_argument("--lr",nargs="?",default=0.0002,type=float,help="learning rate")
ap.add_argument("-sv","--savemod",required=True,help="directory to save model")
ap.add_argument("--beta",nargs="?",default=0.5,type=float,help="beta hyperparam for adam opt")
ap.add_argument("--bs",nargs="?",default= 128,type=int,help="batch size default 128")
#ap.add_argument("--device",nargs="?",default = "cpu" ,help="cpu or cuda")
ap.add_argument("--ngpu",nargs="?",default = 0,type=int,help="no of GPU")
ap.add_argument("--workers",nargs="?",default = 4,type = int,help="no of workers to load more workers==more memory usage==faster data loading")
ap.add_argument("--anm",nargs="?",default = True,type=bool,help="should create animatopn")



args = vars(ap.parse_args())
batch_size = args["bs"]
lr = args["lr"]
beta1=args["beta"]
dataroot = args["dataroot"]
num_workers = args["workers"]
ngpu = args["ngpu"]
save_path = args['savemod']
nz=100
num_epochs = args["nepochs"]
canimation = args["anm"]

print(args)

dataloader = torch.utils.data.DataLoader(dataset(dataroot), 
                                         batch_size=batch_size,
                                         shuffle=True ,
                                         num_workers=num_workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu") 

netG = Generator(ngpu).to(device)
netD = Discriminator(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD,list(range(ngpu)))
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)
netD.apply(weights_init)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))






# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        if canimation==True:
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

saveG = save_path + "/netG.pt"
saveD = save_path + "/netD.pt"
torch.save(netG.state_dict(),saveG)
torch.save(netD.state_dict(),saveD)
if canimation==True:
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("timeVisual.mp4",writer=writer)