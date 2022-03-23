import os
import cv2
import time
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from shutil import copyfile
from tqdm import tqdm

from Data.augmentation import siamcenter_search_augpipe
from Data.ILSVRCDataSet import ILSVRCDataSet
from Data.DataSet import SearchTemplateDataset, ResumableSampler

from SiamCenterNet import SiamCenterNet_ResNet
from CenterLoss import CenterLoss
from CenterNet_Utils import get_bboxs
from Utils import set_seed
from log import TrainingLogger

import matplotlib
matplotlib.use('Agg')
from Visualizer import plt, plot_multiImage


SAVE_FOLDER = 'Save/SiamCenterNet_ResNet/r18/Adam/baseline-withaug'
SAVE_SAMPLES = os.path.join(SAVE_FOLDER, 'epoch_samples')
CHECKPOINT_FILE = os.path.join(SAVE_FOLDER, 'checkpoint.tar')
BEST_FILE = os.path.join(SAVE_FOLDER, 'best-model.pth')
SAVE_INTERVAL = 200
SAMPLE_INTERVAL = 2000

TRAIN_PAIRS = 'dataset/train_pairs_withneg_e{}.npy'
VALID_PAIRS = 'dataset/valid_pairs_withneg_e{}.npy'

RESNET_SLUG = 'r18'
SEARCH_SIZE = (255, 255)
TEMPLATE_SIZE = (127, 127)
CONTEXT_AMOUNT = 0.5
FRAME_RANGE = 1000

AUGMENT_FN = siamcenter_search_augpipe()

EPOCHS = 1
BATCH_SIZE = 16
ALPHA = 1.0
BETAS = 1.0
GAMMA = 1.0

logger = TrainingLogger(os.path.join(SAVE_FOLDER, 'history.txt'))


def main():
    torch.backends.cudnn.benchmark = True

    model = SiamCenterNet_ResNet(RESNET_SLUG).cuda()
    out_shape = model.get_output_shape((1, 3, *TEMPLATE_SIZE), (1, 3, *SEARCH_SIZE))[0][-2:]

    train_set = SearchTemplateDataset(sub_dataset=ILSVRCDataSet(mode='train'),
                                      search_size=SEARCH_SIZE,
                                      template_size=TEMPLATE_SIZE,
                                      hm_size=tuple(out_shape),
                                      context_amount=CONTEXT_AMOUNT,
                                      max_frame_range=FRAME_RANGE,
                                      norm_wh=True,
                                      augment_fn=AUGMENT_FN,
                                      pairs_idx_file=TRAIN_PAIRS.format(0))
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=6,
                              sampler=ResumableSampler(train_set, 0))

    valid_set = SearchTemplateDataset(sub_dataset=ILSVRCDataSet(mode='val'),
                                      search_size=SEARCH_SIZE,
                                      template_size=TEMPLATE_SIZE,
                                      hm_size=tuple(out_shape),
                                      context_amount=CONTEXT_AMOUNT,
                                      max_frame_range=FRAME_RANGE,
                                      norm_wh=True,
                                      pairs_idx_file=VALID_PAIRS.format(0))
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    losser = CenterLoss(alpha=ALPHA, beta=BETAS, gamma=GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0
    if os.path.exists(CHECKPOINT_FILE):
        print('Continue training...')
        state = torch.load(CHECKPOINT_FILE)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optim_state_dict'])
        logger.last = state['last_log']
        logger.reset_checkpoint(logger.last)

        start_epoch = logger.last_epcoh

    for e in range(start_epoch, EPOCHS):
        set_seed(e + 1)
        print('Epoch {}/{}'.format(e, EPOCHS - 1))
        train_loader.dataset.random_pairs(TRAIN_PAIRS.format(e))
        valid_loader.dataset.random_pairs(VALID_PAIRS.format(e))

        ep_st = time.time()
        train(model, train_loader, losser, optimizer, e)
        torch.save(model.state_dict(), os.path.join(SAVE_FOLDER, 'model-{}.pth'.format(e)))

        validate(model, valid_loader, losser, e)

        ep_et = time.time() - ep_st
        print('Epoch total-time used: %.0f h : %.0f m : %.0f s' %
              (ep_et // 3600, ep_et // 60, ep_et % 60))


def train(model, dataloader, losser, optimizer, e):
    model.train()
    dataloader.sampler.start_idx = logger.last_idx
    ini = int(np.ceil(len(dataloader) * (logger.last_idx / len(dataloader.dataset))))
    with tqdm(dataloader, desc='Training', initial=ini) as iterator:
        for i, (t_imgs, s_imgs, hm_gt, wh_gt, offset_gt, ct, _, idxs) in enumerate(iterator, start=ini):
            model.zero_grad()

            t_imgs = t_imgs.cuda()
            s_imgs = s_imgs.cuda()
            hm_gt = hm_gt.cuda()
            wh_gt = wh_gt.cuda()
            offset_gt = offset_gt.cuda()

            hm_pd, wh_pd, offset_pd = model(t_imgs, s_imgs)
            loss_cent, loss_regr = losser((hm_pd, wh_pd, offset_pd),
                                          (hm_gt, wh_gt, offset_gt, ct))
            loss = loss_cent + loss_regr
            loss.backward()
            optimizer.step()

            iterator.set_postfix_str(' Loss {:.4f}| center: {:.4f}, regr: {:.4f}'.format(
                loss.detach().item(), loss_cent.detach().item(), loss_regr.detach().item()
            ))
            iterator.update()

            logger.write('t', e, i, idxs[-1], {'center': loss_cent.item(), 'regr': loss_regr.item()})
            if i % SAVE_INTERVAL or i == len(dataloader) - 1:
                states = {'last_log': logger.last,
                          'model_state_dict': model.state_dict(),
                          'optim_state_dict': optimizer.state_dict()}
                torch.save(states, CHECKPOINT_FILE)
                plot_history(e, 't')
            if i % SAMPLE_INTERVAL or i == len(dataloader) - 1:
                samples(t_imgs.cpu(), s_imgs.cpu(), hm_pd.detach().cpu(), wh_pd.detach().cpu(),
                        offset_pd.detach().cpu(), loss.detach().item(),
                        save_path=os.path.join(SAVE_SAMPLES, 'train', '{}_{}.jpg'.format(e, i)))

    logger.reset_last()


def validate(model, dataloader, losser, e):
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, desc='Validation') as iterator:
            for i, (t_imgs, s_imgs, hm_gt, wh_gt, offset_gt, ct, _, idxs) in enumerate(iterator, start=ini):
                t_imgs = t_imgs.cuda()
                s_imgs = s_imgs.cuda()
                hm_gt = hm_gt.cuda()
                wh_gt = wh_gt.cuda()
                offset_gt = offset_gt.cuda()

                hm_pd, wh_pd, offset_pd = model(t_imgs, s_imgs)
                loss_cent, loss_regr = losser((hm_pd, wh_pd, offset_pd),
                                              (hm_gt, wh_gt, offset_gt, ct))
                loss = loss_cent + loss_regr

                iterator.set_postfix_str(' Loss {:.4f}| center: {:.4f}, regr: {:.4f}'.format(
                    loss.detach().item(), loss_cent.detach().item(), loss_regr.detach().item()
                ))
                iterator.update()

                logger.write('v', e, i, idxs[-1], {'center': loss_cent.item(), 'regr': loss_regr.item()})
                if i % SAMPLE_INTERVAL or i == len(dataloader) - 1:
                    samples(t_imgs.cpu(), s_imgs.cpu(), hm_pd.detach().cpu(), wh_pd.detach().cpu(),
                            offset_pd.detach().cpu(), loss.detach().item(),
                            save_path=os.path.join(SAVE_SAMPLES, 'valid', '{}_{}.jpg'.format(e, i)))
    plot_history(e, 'v')


def plot_history(e, mode):
    hist = logger.get_history()
    mt = hist[e][mode]
    for n, v in mt.items():
        avg = np.mean(v)
        std = np.std(v)

        plt.figure()
        plt.plot(v)
        plt.axhline(avg, color='black')
        plt.text(len(v) * 0.1, avg * 1.1, '{:.5f}'.format(avg))

        plt.title('({}) epoch: {} metric: {}'.format(mode, e, n))
        plt.xlabel('batch')
        plt.ylabel(n)
        plt.xlim([0, max(100, len(v))])
        plt.ylim([0, (avg + std) * 1.2])

        path = os.path.join(SAVE_FOLDER, '{}_e{}_{}.jpg'.format(mode, e, n))
        plt.savefig(path)
        plt.close()


def samples(t_imgs, s_imgs, hm_pd, wh_pd, offset_pd, loss, save_path):
    cm = plt.get_cmap('viridis')
    sc_h, sc_w = hm_pd.shape[1] / SEARCH_SIZE[0], hm_pd.shape[2] / SEARCH_SIZE[1]
    qx, qy = int((SEARCH_SIZE[1] - TEMPLATE_SIZE[1]) / 2), \
             int((SEARCH_SIZE[0] - TEMPLATE_SIZE[0]) / 2)

    bboxs, score, _ = get_bboxs(hm_pd, wh_pd, offset_pd, 1, norm_wh=True)
    samples = []
    for i in range(len(bboxs)):
        bb = bboxs[i].squeeze().numpy()

        x1 = int(max(0, bb[0] / sc_w))
        y1 = int(max(0, bb[1] / sc_h))
        x2 = int(min(bb[2] / sc_w, SEARCH_SIZE[1]))
        x2 = x1 + 1 if x2 - x1 <= 0 else x2
        y2 = int(min(bb[3] / sc_h, SEARCH_SIZE[0]))
        y2 = y1 + 1 if y2 - y1 <= 0 else y2

        res = cv2.rectangle(
            cv2.UMat(s_imgs[i].permute(1, 2, 0).mul(255.).numpy().astype(np.uint8)),
            (x1, y1), (x2, y2), (0, 255, 0), 2).get()

        qry = np.zeros_like(res)
        qry[qy:qy + TEMPLATE_SIZE[1], qx:qx + TEMPLATE_SIZE[0], :] = \
            t_imgs[i].permute(1, 2, 0).mul(255.).numpy().astype(np.uint8)

        hm = cv2.resize(hm_pd[i, 0, ...].mul(255.).numpy().astype(np.uint8),
                        (SEARCH_SIZE[1], SEARCH_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        hm = (cm(hm)[:, :, :3] * 255.).astype(np.uint8)

        samples.append(np.concatenate([qry, res, hm], axis=1))

    samples = np.stack(samples, axis=0)
    plot_multiImage(samples, labels=['{:.4f}'.format(sc) for sc in score.squeeze()],
                    title='loss: {:.4f}'.format(loss), fig_size=(15, 10), tight_layout=True,
                    save=save_path)


if __name__ == '__main__':
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        os.makedirs(os.path.join(SAVE_SAMPLES, 'train'))
        os.makedirs(os.path.join(SAVE_SAMPLES, 'valid'))
    copyfile('_train.py', os.path.join(SAVE_FOLDER, '_train.py'))
    copyfile('Models/SiamCenterNet.py', os.path.join(SAVE_FOLDER, 'SiamCenterNet.py'))
    copyfile('Criterion/CenterLoss.py', os.path.join(SAVE_FOLDER, 'CenterLoss.py'))
    copyfile('Data/augmentation.py', os.path.join(SAVE_FOLDER, 'augmentation.py'))
    copyfile('Data/DataSet.py', os.path.join(SAVE_FOLDER, 'DataSet.py'))

    main()