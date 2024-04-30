import click
import os
import classifier_lib
import torch
import numpy as np
import dnnlib
import torchvision.transforms as transforms
import torch.utils.data as data
import random

class BasicDataset(data.Dataset):
  def __init__(self, x_np, y_np, transform=transforms.ToTensor()):
    super(BasicDataset, self).__init__()

    self.x = x_np
    self.y = y_np
    self.transform = transform

  def __getitem__(self, index):
    return self.transform(self.x[index]), self.y[index]

  def __len__(self):
    return len(self.x)


@click.command()

## Data configuration
@click.option('--feature_extractor',   help='Path of feature extractor',   metavar='STR',     type=str,                      default='/checkpoints/discriminator/feature_extractor/32x32_classifier.pt')
@click.option('--savedir',             help='Discriminator save directory', metavar='PATH',    type=str,                      default="/checkpoints/discriminator/cifar10/unbias_500/")
@click.option('--biasdir',             help='Bias data directory',         metavar='PATH',    type=str,                      default="/datasets/cifar10/discriminator_training/bias_10000/fake_data.npz")
@click.option('--refdir',              help='Real sample directory',       metavar='PATH',    type=str,                      default="/datasets/cifar10/discriminator_training/unbias_500/real_data.npz")

@click.option('--img_resolution',      help='Image resolution',            metavar='INT',     type=click.IntRange(min=1),    default=32)
@click.option('--real_mul',            help='Scaling imblance ',           metavar='STR',     type=click.IntRange(min=1),    default=20)

## Training configuration
@click.option('--batch_size',          help='Batch size',                  metavar='INT',     type=click.IntRange(min=1),    default=128)
@click.option('--iter',                help='Training iteration',          metavar='INT',     type=click.IntRange(min=1),    default=20000)
@click.option('--lr',                  help='Learning rate',               metavar='FLOAT',   type=click.FloatRange(min=0),  default=3e-4)
@click.option('--device',              help='Device',                      metavar='STR',     type=str,                      default='cuda:0')

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    path = os.getcwd()
    path_feat = path + opts.feature_extractor
    savedir = path + opts.savedir
    refdir = path + opts.refdir
    biasdir = path + opts.biasdir
    os.makedirs(savedir,exist_ok=True)

    ## Prepare real&fake data
    ref_data = np.load(refdir)['samples']
    for k in range(opts.real_mul):
        if k == 0:
            ref_datas = ref_data
        else:
            ref_datas = np.concatenate([ref_datas, ref_data])
    ref_data = ref_datas
    bias_data = np.load(biasdir)['samples']
    print("bias:", len(bias_data))
    print("scaled unbias:", len(ref_data))

    ## Loader
    transform = transforms.Compose([transforms.ToTensor()])
    ref_label = torch.ones(ref_data.shape[0])
    bias_label = torch.zeros(bias_data.shape[0])

    ref_data = BasicDataset(ref_data, ref_label, transform)
    bias_data = BasicDataset(bias_data, bias_label, transform)
    ref_loader = torch.utils.data.DataLoader(dataset=ref_data, batch_size=opts.batch_size, num_workers=0, shuffle=True, drop_last=True)
    bias_loader = torch.utils.data.DataLoader(dataset=bias_data, batch_size=opts.batch_size, num_workers=0, shuffle=True, drop_last=True)
    ref_iterator = iter(ref_loader)
    bias_iterator = iter(bias_loader)

    ## Extractor & Disciminator
    pretrained_classifier = classifier_lib.load_classifier(path_feat, opts.img_resolution, opts.device, eval=False)
    discriminator = classifier_lib.load_discriminator(None, opts.device, 0, eval=False)

    ## Prepare training
    vpsde = classifier_lib.vpsde()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=opts.lr, weight_decay=1e-7)
    loss = torch.nn.BCELoss()
    scaler = lambda x: 2. * x - 1.

    ## Training
    for i in range(opts.iter):
        if i % 500 == 0:
            outs = []
            cors = []
            num_data = 0
        try:
            r_inputs, r_labels = next(ref_iterator)
        except:
            ref_iterator = iter(ref_loader)
            r_inputs, r_labels = next(ref_iterator)
        try:
            f_inputs, f_labels = next(bias_iterator)
        except:
            bias_iterator = iter(bias_loader)
            f_inputs, f_labels = next(bias_iterator)

        inputs = torch.cat([r_inputs, f_inputs])
        labels = torch.cat([r_labels, f_labels])
        c = list(range(inputs.shape[0]))
        random.shuffle(c)
        inputs, labels = inputs[c], labels[c]

        optimizer.zero_grad()
        inputs = inputs.to(opts.device)
        labels = labels.to(opts.device)
        inputs = scaler(inputs)

        ## Data perturbation
        t, _ = vpsde.get_diffusion_time(inputs.shape[0], inputs.device, importance_sampling=True)
        mean, std = vpsde.marginal_prob(t)
        z = torch.randn_like(inputs)
        perturbed_inputs = mean[:, None, None, None] * inputs + std[:, None, None, None] * z

        ## Forward
        with torch.no_grad():
            pretrained_feature = pretrained_classifier(perturbed_inputs, timesteps=t, feature=True)
        label_prediction = discriminator(pretrained_feature, t, sigmoid=True).view(-1)

        ## Backward
        out = loss(label_prediction, labels)
        out.backward()
        optimizer.step()

        ## Report
        cor = ((label_prediction > 0.5).float() == labels).float().mean()
        outs.append(out.item())
        cors.append(cor.item())
        num_data += inputs.shape[0]
        print(f"{i}-th iter BCE loss: {np.mean(outs)}, correction rate: {np.mean(cors)}")

        if i % 500 == 0:
            torch.save(discriminator.state_dict(), savedir + f"/discriminator_{i+1}.pt")

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------