import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def siamcenter_search_augpipe():
    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.Noop(),
            iaa.GaussianBlur(sigma=(0.0, 3.0)),
            iaa.MedianBlur(k=(1, 7)),
            iaa.AverageBlur(k=((3, 5), (3, 5))),
            iaa.MotionBlur(k=(3, 5), angle=(0, 360))
        ]),
        iaa.SomeOf(
            (0, None),
            [
                iaa.Fliplr(),
                iaa.Multiply(mul=(0.8, 1.2), per_channel=0.2),
                iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255), per_channel=0.5),
                iaa.SigmoidContrast(gain=(3, 6), cutoff=(0.4, 0.6)),
                iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.0, 0.8)),
                iaa.OneOf([
                    iaa.HistogramEqualization(),
                    iaa.CLAHE(),
                    iaa.LinearContrast(),
                    iaa.GammaContrast()
                ])
            ]
        ),
        ## Affect BBox.
        iaa.Sometimes(0.5, iaa.Affine(
            rotate=(-10, 10),
            shear=(-5, 5),
            scale=(0.9, 1.1),
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            mode='edge'
        )),
    ])

    return seq


if __name__ == '__main__':
    import cv2
    import numpy as np
    from ILSVRCDataSet import ILSVRCDataSet
    from DataSet import SearchTemplateDataset
    from Visualizer import plot_multiImage

    dataset = SearchTemplateDataset(ILSVRCDataSet(mode='val'),
                                    search_size=255,
                                    template_size=127,
                                    hm_size=26,
                                    context_amount=0.5,
                                    max_frame_range=1000,
                                    augment_fn=siamcenter_search_augpipe())

    n = 25
    t_imgs, s_imgs = [], []
    for i in range(n):
        idx = np.random.randint(0, len(dataset))
        print(idx)

        t_img, s_img, _, _, _, _, cbb = dataset.__getitem__(idx)
        t_img = t_img.permute(1, 2, 0).cpu().mul(255.).numpy().astype('uint8')
        s_img = s_img.permute(1, 2, 0).cpu().mul(255.).numpy().astype('uint8')

        cbb = cbb.cpu().numpy().astype(int)
        bb = (cbb[0] - (cbb[2] // 2), cbb[1] - (cbb[3] // 2),
              cbb[0] + (cbb[2] // 2), cbb[1] + (cbb[3] // 2))
        s_img = cv2.rectangle(cv2.UMat(s_img), (bb[0], bb[1]), (bb[2], bb[3]),
                              (0, 255, 0), 2).get()

        t_imgs.append(t_img)
        s_imgs.append(s_img)

    t_imgs = np.stack(t_imgs, axis=0)
    s_imgs = np.stack(s_imgs, axis=0)

    plot_multiImage(t_imgs, tight_layout=True)
    plot_multiImage(s_imgs, tight_layout=True)
