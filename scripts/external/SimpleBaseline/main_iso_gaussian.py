from __future__ import print_function, absolute_import, division

import os
import datetime
import argparse
import numpy as np
import os.path as path

import torch
from torch.utils.data import DataLoader

from models.linear_model import LinearModel, init_weights

from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe


from models.iso_gaussian_model import IsoGaussianModel, FullGaussianModel


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=10, type=int, help='save models for every #snapshot epochs (default: 20)')

    # Model arguments
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-z', '--hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('-l', '--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('--non_local', dest='non_local', action='store_true', help='if use non-local layers')
    parser.set_defaults(non_local=False)
    parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')

    # Experimental
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')

    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args


def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample
    # cudnn.benchmark = True
    device = torch.device("cpu")

    # Create model
    print("==> Creating model...")
    num_joints = dataset.skeleton().num_joints()
    model_pos = LinearModel(num_joints * 2, (num_joints - 1) * 3).to(device)
    model_pos.apply(init_weights)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))
    model_gauss = IsoGaussianModel((num_joints - 1) * 3).to(device)

    optimizer = torch.optim.Adam(model_gauss.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location=device)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = 0
            lr_now = args.lr

            # fix "nonlocal" to "non_local" in state dict
            new_state_dict = {}
            for k, v in ckpt['state_dict'].items():
                new_state_dict[k.replace('nonlocal', 'non_local')] = v
            ckpt['state_dict'] = new_state_dict

            model_pos.load_state_dict(ckpt['state_dict'])
            # optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))

        if args.evaluate:
            gauss_path = './ckpt_iso_gaussian_log_prob.pth.tar'
            if path.isfile(gauss_path):
                ckpt = torch.load(gauss_path, map_location=device)
                model_gauss.load_state_dict(ckpt['state_dict'])

    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = args.lr
        ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().isoformat())

        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

        logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])

    if args.evaluate:
        print('==> Evaluating...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(len(action_filter))

        for i, action in enumerate(action_filter):
            print(action)
            poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, [action], stride)
            valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)

            errors_p1[i] = evaluate(valid_loader, model_pos, model_gauss, device)

            print('\tminMPJPE: {:.2f}'.format(errors_p1[i]))

        print('Protocol #1  (minMPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))

        exit(0)

    poses_train, poses_train_2d, actions_train = fetch(subjects_train, dataset, keypoints, action_filter, stride)
    train_loader = DataLoader(PoseGenerator(poses_train, poses_train_2d, actions_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))
        print('doing train...')
        # Train for one epoch
        train(train_loader, model_pos, model_gauss, optimizer, device)

    logger.close()

    return


def train(data_loader, model_mean, gauss_model, optimizer, device):
    model_mean.eval()

    bar = Bar('Train', max=len(data_loader))
    bar.suffix = f"({0}/{len(data_loader)}) minMPJPE (mm): "
    bar.next()
    lr_now = 0.001
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        optimizer.zero_grad()
        num_poses = targets_3d.size(0)

        targets_3d, inputs_2d = targets_3d[:, 1:, :].to(device), inputs_2d.to(device)  # Remove hip joint for 3D poses

        outputs_3d_mean = model_mean(inputs_2d.view(num_poses, -1))

        loss = -gauss_model.log_prob(outputs_3d_mean.view(num_poses, -1), targets_3d.view(num_poses, -1)).mean()

        # samples = gauss_model.sample(outputs_3d_mean.view(num_poses, -1), n_samples=(num_poses, 200))
        # samples = samples.view(*(num_poses, 200), -1, 3).swapaxes(0, 1)
        #
        # mpjpe_error = mpjpe(samples, targets_3d.unsqueeze(0).repeat(200, 1, 1, 1)).min(0).values
        # loss = mpjpe_error.mean()
        loss.backward()

        optimizer.step()

        bar.suffix = f"({i}/{len(data_loader)}) minMPJPE (mm): {loss.item() * 1000:.4f}"
        bar.next()

        if i % 1000 == 0:
            print('saving checkpoint')
            save_ckpt({'state_dict': gauss_model.state_dict()}, './', suffix='iso_gaussian_log_prob')

    bar.finish()

    return loss.item()


def evaluate(data_loader, model_mean, gauss_model, device):
    epoch_loss_3d_pos = AverageMeter()

    torch.set_grad_enabled(False)
    model_mean.eval()
    gauss_model.eval()


    bar = Bar('Eval ', max=len(data_loader))
    bar.suffix = f"({0}/{len(data_loader)}) minMPJPE (mm): "
    bar.next()
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        num_poses = targets_3d.size(0)

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)  # Remove hip joint for 3D poses

        outputs_3d_mean = model_mean(inputs_2d.view(num_poses, -1))

        samples = gauss_model.sample(outputs_3d_mean.view(num_poses, -1), n_samples=(num_poses, 200))

        samples = samples.view(*(num_poses, 200), -1, 3).swapaxes(0, 1)
        samples = np.insert(samples, 0, 0, axis=2)

        mpjpe_error = mpjpe(samples, targets_3d.unsqueeze(0).repeat(200, 1, 1, 1)).min(0).values
        mpjpe_error = mpjpe_error.mean().item() * 1000

        epoch_loss_3d_pos.update(mpjpe_error, num_poses)

        bar.suffix = f"({i}/{len(data_loader)}) minMPJPE (mm): {mpjpe_error:.4f}"
        bar.next()

    return epoch_loss_3d_pos.avg


if __name__ == '__main__':
    main(parse_args())