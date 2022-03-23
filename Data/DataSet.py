import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from Data.prepro_siamese import crop_and_pad, get_context_size
from Data.prepro_center import make_single_hm_umich
from Utils import resize_padding, corner2center

import matplotlib.pyplot as plt


class SearchTemplateDataset(Dataset):
    def __init__(self, sub_dataset, search_size, template_size, hm_size, context_amount,
                 max_frame_range=100, norm_wh=True, augment_fn=None, pairs_idx_file='./dataset/idx_pairs.npy'):
        self.data = sub_dataset.data

        self.search_size = search_size if isinstance(search_size, (tuple, list)) else (search_size, search_size)
        self.template_size = template_size if isinstance(template_size, (tuple, list)) else (template_size, template_size)
        self.hm_size = hm_size if isinstance(hm_size, (tuple, list)) else (hm_size, hm_size)

        self.context_amount = context_amount
        self.max_frame_range = max_frame_range
        self.norm_wh = norm_wh

        self.augment_fn = augment_fn

        self.pairs_idx_file = pairs_idx_file
        self.pairs = np.zeros((len(self.data), 3), dtype=int)
        self.random_pairs(self.pairs_idx_file)

    def __len__(self):
        return len(self.data)

    def random_pairs(self, save_pair_file, overwrite=False):
        if not os.path.exists(save_pair_file) or overwrite:
            for tmp_idx in tqdm(range(self.__len__()), desc='Random new pairs'):
                neg = np.random.rand() > 0.5  # random with uniform for equality.
                self.pairs[tmp_idx, :] = self.get_template_search_pair(tmp_idx, neg)

            np.random.shuffle(self.pairs)
            np.save(save_pair_file, self.pairs)
            print('Save paired index file to: {}'.format(save_pair_file))
        else:
            print('Load paired index file from: {}'.format(save_pair_file))
            self.pairs = np.load(save_pair_file)

    def get_template_search_pair(self, tmp_idx, negative=False):
        # Assume column is [seq, path, is_val, frame, id, name, bbox, occluded, generated]
        tmp_id = self.data.iloc[tmp_idx, 4]
        seq = self.data.iloc[tmp_idx, 0]

        vid_idx = np.where((self.data['seq'] == seq) & (self.data['id'] == tmp_id))[0]

        if negative:
            src_idx = np.random.randint(0, len(self.data))
            while src_idx in vid_idx:
                src_idx = np.random.randint(0, len(self.data))
        else:
            f_num = len(vid_idx)
            vid_idx_tmp = np.where(vid_idx == tmp_idx)[0][0]
            left = max(0, vid_idx_tmp - self.max_frame_range)
            right = min(vid_idx_tmp + self.max_frame_range, f_num - 1)
            vid_idx_src = np.random.randint(left, right + 1)

            src_idx = vid_idx[vid_idx_src]
        return tmp_idx, src_idx, negative

    def get_data(self, idx):
        path = self.data.iloc[idx, 1]
        bb = self.data.iloc[idx, 6]
        img = cv2.imread(path)
        cbb = corner2center(bb)
        return img, bb, cbb

    def __getitem__(self, idx):
        tmp, src, neg = self.pairs[idx]

        t_img, _, t_cbb = self.get_data(tmp)
        s_img, s_bb, _ = self.get_data(src)

        t_sz = get_context_size(t_cbb, self.context_amount)
        t_img, _ = crop_and_pad(t_img, t_cbb[0], t_cbb[1], self.template_size, (t_sz, t_sz))

        s_img, sc, (pw, ph) = resize_padding(
            s_img, self.search_size[1], self.search_size[0],
            pad_value=tuple(map(int, s_img.mean(axis=(0, 1)))))
        s_bb = np.array([s_bb[0] * sc + pw, s_bb[1] * sc + ph,
                         s_bb[2] * sc + pw, s_bb[3] * sc + ph])

        if self.augment_fn is not None:
            t_img = self.augment_fn(image=t_img)
            s_img, s_bb = self.augment_fn(image=s_img, bounding_boxes=s_bb[None, None, ...])
            s_bb = s_bb.squeeze()

        cbb = corner2center(s_bb)

        if neg:
            hm = np.zeros((self.hm_size[0], self.hm_size[1]), dtype=np.float32)
            wh = np.zeros(2, dtype=np.float32)
            offset = np.zeros(2, dtype=np.float32)
            ct = np.array((self.hm_size[0] // 2, self.hm_size[1] // 2), dtype=np.float32)
        else:
            hm, wh, offset, ct = make_single_hm_umich(
                self.search_size, cbb, self.hm_size, self.norm_wh)

        t_img = torch.tensor(t_img, dtype=torch.float32).permute(2, 0, 1).contiguous().div(255.)
        s_img = torch.tensor(s_img, dtype=torch.float32).permute(2, 0, 1).contiguous().div(255.)
        hm = torch.tensor(hm[None, ...], dtype=torch.float32).contiguous()
        wh = torch.tensor(wh, dtype=torch.float32)
        offset = torch.tensor(offset, dtype=torch.float32)
        ct = torch.tensor(ct, dtype=torch.long)
        cbb = torch.tensor(cbb, dtype=torch.float32)
        return t_img, s_img, hm, wh, offset, ct, cbb, idx

    def show_sample(self, idx):
        t_img, s_img, hm, wh, offset, ct, cbb, idx = self.__getitem__(idx)
        t_img = t_img.permute(1, 2, 0).cpu().mul(255.).numpy().astype('uint8')
        s_img = s_img.permute(1, 2, 0).cpu().mul(255.).numpy().astype('uint8')
        hm = hm.permute(1, 2, 0).squeeze().cpu().numpy()
        cbb = cbb.cpu().numpy().astype(int)

        bb = (cbb[0] - (cbb[2] // 2), cbb[1] - (cbb[3] // 2),
              cbb[0] + (cbb[2] // 2), cbb[1] + (cbb[3] // 2))
        s_img = cv2.rectangle(cv2.UMat(s_img), (bb[0], bb[1]), (bb[2], bb[3]),
                              (0, 255, 0), 2).get()

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(t_img)
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.imshow(s_img)
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.imshow(hm)
        plt.tight_layout()
        print('wh: ', wh)
        print('offset: ', offset)


class ResumableSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, start_idx=0):
        super(ResumableSampler, self).__init__(data_source)
        self.data_source = data_source
        self.start_idx = start_idx

    def __iter__(self):
        return iter(range(self.start_idx, len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class Test:
    def __init__(self, num):
        self.data = list(range(num))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':
    from tqdm import tqdm
    from ILSVRCDataSet import ILSVRCDataSet
    from torch.utils.data import DataLoader
    from augmentation import siamcenter_search_augpipe
    from Visualizer import plot_multiImage
    from Utils import set_seed

    dataset = SearchTemplateDataset(ILSVRCDataSet('train'), 255, 127, 26, 0.5, max_frame_range=1000,
                                    augment_fn=siamcenter_search_augpipe(),
                                    pairs_idx_file=os.path.join('dataset', 'train_pairs_e0.npy'))
    #dataset.show_sample(0)

    #dataset = Test(1000)
    #start_idx = 200
    #sampler = ResumableSampler(dataset, start_idx)

    #loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=4, shuffle=False)
    #batch = next(iter(loader))

    """with tqdm(loader, desc='test loader') as iterator:
        for i, (t_imgs, s_imgs, hm_gt, wh_gt, offset_gt, ct, _, idxs) in enumerate(iterator):
            t_imgs = t_imgs.cuda()
            s_imgs = s_imgs.cuda()
            hm_gt = hm_gt.cuda()
            wh_gt = wh_gt.cuda()
            offset_gt = offset_gt.cuda()

            print(idxs)
            iterator.update()"""

    """ini = int(np.ceil(len(loader) * (start_idx / len(dataset))))
    for e in range(3):
        #set_seed()
        print(len(loader))
        with tqdm(loader, desc='test', initial=ini, total=len(loader)) as iterator:
            for i, bat in enumerate(iterator):
                #if i < 5:
                iterator.update()
                time.sleep(0.2)
                #print(bat)

        loader.sampler.start_idx = 0
        ini = 0
        print()"""

    """for e in range(3):
        #set_seed()
        for bat in loader:
            img = bat[0]
            plot_multiImage(img.permute(0, 2, 3, 1))
            break"""

