import argparse
import json
import time
from time import gmtime, strftime
import test
from models import *
from shutil import copyfile
from utils.datasets import JointDataset, collate_fn
from utils.utils import *
from utils.log import logger
from torchvision.transforms import transforms as T


def train(
        cfg,
        data_cfg,
        weights_from="",
        weights_to="",
        save_every=10,
        img_size=(1088, 608),
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None,
):
    timme = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    timme = timme[5:-3].replace('-', '_')
    timme = timme.replace(' ', '_')
    timme = timme.replace(':', '_')
    weights_to = osp.join(weights_to, 'run' + time)
    mkdir_if_missing(weights_to)
    if resume:
        latest_resume = osp.join(weights_from, 'latest.pt')

    torch.backends.cudnn.benchmark = True

    f = open(data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()

    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(dataset_root, trainset_paths, img_size, augment = True, transform = transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    model = Darknet(cfg, dataset.nID)

    cutoff = -1
    start_epoch = 0
    if resume:
        checkpoint = torch.load(latest_resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.cuda().train()

        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9)

        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

        del checkpoint

    else:
        if cfg.endswith('yolov3.cfg'):
            load_darknet_weights(model, osp.join(weights_from, 'darknet53.conv.74'))
            cutoff = 75
        elif cfg.endswith('yolov3-tiny.cfg'):
            load_darknet_weights(model, osp.join(weights_from, 'yolov3-tiny.conv.15'))
            cutoff = 15

        model.cuda().train()

        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=.9, weight_decay=1e-4)

    model = torch.nn.DataParallel(model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * opt.epochs), int(0.75 * opt.epochs)], gamma=0.1)


    if not opt.unfreeze_bn:
        for i, (name, p) in enumerate(model.named_parameters()):
            p.requires_grad = False if 'batch_norm' in name else True

    t0 = time.time()
    for epoch in range(epochs):
        epoch += start_epoch
        logger.info(('%8s%12s' + '%10s' * 6) % ('Epoch', 'Batch', 'box', 'conf', 'id', 'total', 'nTargets', 'time'))

        if freeze_backbone and (epoch < 2):
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[2]) < cutoff:
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)
        optimizer.zero_grad()
        for i, (imgs, targets, _, _, targets_len) in enumerate(dataloader):
            if sum([len(x) for x in targets]) < 1:
                continue

            burnin = min(1000, len(dataloader))
            if (epoch == 0) & (i <= burnin):
                lr = opt.lr * (i / burnin) ** 4
                for g in optimizer.param_groups:
                    g['lr'] = lr

            loss, components = model(imgs.cuda(), targets.cuda(), targets_len.cuda())
            components = torch.mean(components.view(-1, 5), dim=0)
            loss = torch.mean(loss)
            loss.backward()

            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()

            ui += 1

            for ii, key in enumerate(model.module.loss_names):
                rloss[key] = (rloss[key] * ui + components[ii] / (ui + 1))
                
