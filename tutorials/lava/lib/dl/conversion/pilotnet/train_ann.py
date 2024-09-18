import os
import glob
import argparse
from datetime import datetime
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F

from lava.lib.dl import slayer

class PilotNetDataset(Dataset):
    """PilotNet dataset class for ANN. This does not preserve temporal
    continuity. Returns image and ground truth value when the object is indexed.

    Parameters
    ----------
    path : str
        Path of the dataset folder. If the folder does not exists, the folder
        is created and the dataset is downloaded and extracted to the folder.
        Defaults to '../data'.
    train : bool
        Flag to indicate training or testing set. Defaults to True.
    visualize : bool
        If true, the train/test split is ignored and the temporal sequence of
        the data is preserved. Defaults to False.
    transform : lambda
        Transformation function to be applied to the input image.
        Defaults to None.

    Examples
    --------

    >>> dataset = PilotNetDataset()
    >>> images, gts = dataeset[0]
    >>> num_samples = len(dataset)
    """
    def __init__(
        self, path='data',
        train=True, visualize=False, transform=None,
        extract=True, download=True,
    ) -> None:
        self.path = path + '/driving_dataset/'

        id = '1Ue4XohCOV5YXy57S_5tDfCVqzLr101M7'
        dataset_link = 'https://docs.google.com/uc?export=download&id={id}'
        download_msg = f'''Please download dataset form \n{dataset_link}')
        and copy driving_dataset.zip to {path}/
        Note: create the folder if it does not exist.'''.replace(' ' * 8, '')

        # check if dataset is available in path. If not download it
        if len(glob.glob(self.path)) == 0:
            if download is True:
                os.makedirs(path, exist_ok=True)

                print('Dataset not available locally. Starting download ...')
                download_cmd = 'wget --load-cookies /tmp/cookies.txt '\
                    + '"https://docs.google.com/uc?export=download&confirm='\
                    + '$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate '\
                    + f"'https://docs.google.com/uc?export=download&id={id}' -O- | "\
                    + f"sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p\')&id={id}"\
                    + f'" -O {path}/driving_dataset.zip && rm -rf /tmp/cookies.txt'
                print(download_cmd)
                exec_id = os.system(download_cmd + f' >> {path}/download.log')
                if exec_id == 0:
                    print('Download complete.')
                else:
                    raise Exception(download_msg)

            if extract is True:
                if os.path.exists(path + '/driving_dataset.zip'):
                    print('Extracting data (this may take a while) ...')
                    exec_id = os.system(
                        f'unzip {path}/driving_dataset.zip -d {path} '
                        f'>> {path}/unzip.log'
                    )
                    if exec_id == 0:
                        print('Extraction complete.')
                    else:
                        print(
                            f'Could not extract file '
                            f'{path + "/driving_dataset.zip"}. '
                            f'Please extract it manually.'
                        )
                else:
                    print(f'Could not find {path + "/driving_dataset.zip"}.')
                    raise Exception(download_msg)
            else:
                print('Dataset does not exist. set extract=True')
                if not os.path.exists(path + '/driving_dataset.zip'):
                    raise Exception(download_msg)

        with open(path + 'data.txt', 'r') as data:
            all_samples = [line.split() for line in data]

        self.samples = all_samples

        if visualize is True:
            inds = np.arange(len(all_samples))
        else:
            inds = np.random.RandomState(seed=18).permutation(
                len(all_samples)
            )
        if train is True:
            self.ind_map = inds[
                :int(len(all_samples) * 0.8)
            ]
        else:
            self.ind_map = inds[
                -int(len(all_samples) * 0.2):
            ]

        self.transform = transform

    def __getitem__(self, index: int):
        path, gt = self.samples[self.ind_map[index]]
        image = Image.open(self.path + path)
        gt_val = float(gt) * np.pi / 180
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor([gt_val], dtype=image.dtype)

    def __len__(self) -> int:
        return len(self.ind_map)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.blocks = torch.nn.ModuleList([# sequential network blocks
                # convolution layers
                torch.nn.Conv2d(  3, 24, 3, padding=0, stride=2), torch.nn.BatchNorm2d(24), torch.nn.ReLU(),
                torch.nn.Conv2d( 24, 36, 3, padding=0, stride=2), torch.nn.BatchNorm2d(36), torch.nn.ReLU(),
                torch.nn.Conv2d( 36, 48, 3, padding=0, stride=2), torch.nn.BatchNorm2d(48), torch.nn.ReLU(),
                torch.nn.Conv2d( 48, 64, 3, padding=(1, 0), stride=(2, 1)), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
                torch.nn.Conv2d( 64, 64, 3, padding=0, stride=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
                # flatten layer
                torch.nn.Flatten(),
                # dense layers
                torch.nn.Linear( 64*40, 100), torch.nn.BatchNorm1d(100), torch.nn.ReLU(),
                torch.nn.Linear(   100,  50), torch.nn.BatchNorm1d( 50), torch.nn.ReLU(),
                torch.nn.Linear(    50,  10), torch.nn.BatchNorm1d( 10), torch.nn.ReLU(),
                # linear readout with sigma decoding of output
                torch.nn.Linear(    10,   1)
            ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu',   type=int,   default=[0],     help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',     type=int,   default=32,      help='batch size for dataloader')
    parser.add_argument('-lr',    type=float, default=0.001,   help='initial learning rate')
    parser.add_argument('-seq',   type=int,   default=1,       help='sequence of frames in a sample')
    parser.add_argument('-exp',   type=str,   default='',      help='experiment differentiater string')
    parser.add_argument('-seed',  type=int,   default=None,    help='random seed of the experiment')
    parser.add_argument('-epoch', type=int,   default=100,     help='number of epochs to run')
    parser.add_argument('-step',  type=int,   default=[60, 120, 160], help='milestones for learing rate scheduler', nargs='+')

    args = parser.parse_args()

    identifier = 'ANN_' + args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{args.seed}'

    trained_folder = 'Trained' + identifier
    logs_folder    = 'Logs' + identifier
    print(trained_folder)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder   , exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    if len(args.gpu) == 1:
        net = Network(args.maxrate).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(args.maxrate).to(device), device_ids=args.gpu)
        module = net.module

    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(), lr = args.lr, weight_decay=1e-5)

    # Dataset and dataLoader instances.
    training_set = PilotNetDataset(
        train=True,
        transform=transforms.Compose([
            transforms.Resize([66, 200]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )
    testing_set = PilotNetDataset(
        train=False,
        transform=transforms.Compose([
            transforms.Resize([66, 200]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )

    visualization_set = PilotNetDataset(
        visualize=True,
        transform=transforms.Compose([
            transforms.Resize([66, 200]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
    )

    train_loader = DataLoader(dataset=training_set, batch_size=args.b, shuffle=True, num_workers=8)
    test_loader  = DataLoader(dataset=testing_set , batch_size=args.b, shuffle=True, num_workers=8)

    # Learning stats instance.
    stats = slayer.utils.LearningStats()

    # training loop
    for epoch in range(args.epoch):
        tSt = datetime.now()

        if epoch in args.step:
            for param_group in optimizer.param_groups:
                print('Learning rate reduction from', param_group['lr'])
                stats.new_line()
                param_group['lr'] /= 10/3

        # Training loop.
        for i, (input, ground_truth) in enumerate(train_loader, 0):
            net.train()

            input  = input.to(device)
            ground_truth = ground_truth.to(device)

            output = net.forward(input)

            stats.training.num_samples += input.shape[0]

            loss = F.mse_loss(output.flatten(), ground_truth.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats.training.loss_sum += loss.cpu().data.item() * output.shape[0]

            # stats.linesPrinted = 0
            # Display training stats.
            stats.print(
                epoch, i,
                (datetime.now() - tSt).total_seconds() / (i+1) / train_loader.batch_size,
                dataloader = train_loader,
            )

        # Testing loop.
        for i, (input, ground_truth) in enumerate(test_loader, 0):
            net.eval()

            with torch.no_grad():
                input  = input.to(device)
                ground_truth = ground_truth.to(device)

                output = net.forward(input)

                stats.testing.num_samples += len(ground_truth)

                loss = F.mse_loss(output.flatten(), ground_truth.flatten())
                # loss_mae = F.l1_loss(output, ground_truth)
                stats.testing.loss_sum += loss.cpu().data.item() * output.shape[0]

            # stats.linesPrinted = 0
            # Display training stats.
            stats.print(
                epoch, i,
                dataloader = test_loader,
            )


        if stats.testing.best_loss:
            torch.save(module.state_dict(), trained_folder + '/network.pt')

        # Update stats.
        stats.update()
        stats.plot(path=trained_folder + '/')
        stats.save(trained_folder + '/')

        if epoch%10 == 0:
            torch.save(
                {
                    'net': module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                logs_folder + '/checkpoint%d.pt'%(epoch)
            )

