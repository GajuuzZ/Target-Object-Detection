import os
import cv2
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from contextlib import closing

from Data.prepro_siamese import get_instance_image


def _read_anno(folder):
    seq, folder = folder
    img_files = glob(os.path.join(folder, '*.JPEG'))
    img_files = sorted(img_files, key=lambda x: int(
        os.path.split(x)[-1].split('.')[0]))

    frame = 0
    anno = []
    for path in img_files:
        xml_path = path.replace('Data', 'Annotations').replace('.JPEG', '.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        is_val = True if 'val' in path else False

        o = 0
        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))

            trk = {
                'seq': seq,
                'path': path,
                'is_val': is_val,
                'frame': frame,
                'id': int(obj.find('trackid').text),
                'name': obj.find('name').text,
                'bbox': bbox,
                'occluded': int(obj.find('occluded').text),
                'generated': int(obj.find('generated').text)
            }
            anno.append(trk)
            o += 1
        if o > 0:
            frame += 1
    return anno


def to_bboxarray(str_list):
    num = list(map(int, str_list[1:-1].split(',')))
    return np.array(num)


class ILSVRCDataSet(object):
    """Data set loader for ImageNet Large Scale Visual Recognition Challenge (ILSVRC)"""
    def __init__(self, mode='train', path_to_dataset='./dataset/ILSVRC', list_file='./dataset/ilsvrc_list.csv'):
        self.mode = mode
        self.path_to_dataset = path_to_dataset
        self.list_file = list_file

        print('Loading info list...')
        if not os.path.exists(self.list_file):
            print('  Cant find pre-create list file, Creating new one!...')
            self.data = self.load_info()
        else:
            self.data = pd.read_csv(self.list_file, converters={'bbox': to_bboxarray})

        if mode == 'train':
            self.data = self.data.loc[self.data['is_val'] == False].reset_index(drop=True)
        elif mode == 'val':
            self.data = self.data.loc[self.data['is_val'] == True].reset_index(drop=True)

    def load_info(self):
        """Load annotation info into csv file. """
        vid_dir = os.path.join(self.path_to_dataset, 'Data', 'VID')
        all_videos = glob(os.path.join(vid_dir, 'train', 'ILSVRC2015_VID_train_0000', '*')) + \
                     glob(os.path.join(vid_dir, 'train', 'ILSVRC2015_VID_train_0001', '*')) + \
                     glob(os.path.join(vid_dir, 'train', 'ILSVRC2015_VID_train_0002', '*')) + \
                     glob(os.path.join(vid_dir, 'train', 'ILSVRC2015_VID_train_0003', '*')) + \
                     glob(os.path.join(vid_dir, 'val', '*'))
        seq = list(range(1, len(all_videos) + 1))
        all_videos = list(zip(seq, all_videos))

        df = []
        # Use multi-processing to read massive files.
        with closing(Pool(processes=4)) as pool:
            for ret in tqdm(pool.imap_unordered(_read_anno, all_videos), total=len(all_videos)):
                df.extend(ret)
            pool.terminate()

        df = pd.DataFrame(df)
        df = df.sort_values(by=['seq', 'frame', 'id'], ignore_index=True)
        df.to_csv(self.list_file, index=False)
        return df

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def show_video_sample(self, seq, instance_size, exemplar_size, max_translate, scale_resize,
                          context_amount, delay=50):
        df = self.data.loc[self.data['seq'] == seq]

        if len(df) > 0:
            instance_crop_size = int(np.ceil(
                (instance_size + max_translate * 2) * (1 + scale_resize))
            )

            i = 0
            num_frame = df['frame'].iloc[-1]
            while True:
                print('{}/{}'.format(i, num_frame))
                if i >= num_frame:
                    break

                rows = df.loc[df['frame'] == i]

                img = cv2.imread(rows['path'].iloc[0])
                for _, r in rows.iterrows():
                    id_ = r.loc['id']
                    bb = list(map(int, r.loc['bbox']))
                    occ = r.loc['occluded']
                    gen = r.loc['generated']

                    cbb = np.array([(bb[2] + bb[0]) / 2, (bb[3] + bb[1]) / 2,
                                    bb[2] - bb[0] + 1, bb[3] - bb[1] + 1])
                    instance_img, w, h, _ = get_instance_image(
                        img, cbb, exemplar_size, instance_crop_size, context_amount)
                    ct = instance_crop_size / 2
                    res = cv2.rectangle(instance_img, (int(ct - (w / 2)), int(ct - (h / 2))),
                                        (int(ct + (w / 2)), int(ct + (h / 2))), (0, 255, 0), 2)
                    cv2.imshow('obj:{}'.format(id_), res)

                    img = cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                    img = cv2.putText(img, 'occ: {}, gen: {}'.format(occ, gen),
                                      (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (255, 0, 0), 2)

                cv2.imshow('frame', img)
                if cv2.waitKey(delay) == ord('q'):
                    break
                i += 1

            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    exemplar_size = 127
    instance_size = 255
    max_translate = 0
    scale_resize = 0.5
    context_amount = 0.1

    dataset = ILSVRCDataSet()
    dataset.show_video_sample(2, instance_size, exemplar_size, max_translate, scale_resize,
                              context_amount, delay=25)




