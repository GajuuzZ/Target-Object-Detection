import os
import cv2
import time
import torch
import argparse

from Models.SiamCenterNet import SiamCenterNet_ResNet
from Models.CenterNet_Utils import get_bboxs
from Utils import select_target, get_template_image, resize_padding


def parse_args():
    parser = argparse.ArgumentParser(description='Target Object Detection tester.')
    parser.add_argument('-v', '--video', type=str, required=True, help='video file to search.')
    parser.add_argument('-i', '--image', type=str, default=None, help='target image if not from video')
    parser.add_argument('-m', '--model', type=str, required=True, help='model path. (.pth or .tar)')
    parser.add_argument('-s', '--slug', type=str, default='r18', help='model slug should match the model path')
    parser.add_argument('-c', '--context_amount', type=float, default=0.5, help='size of croped object in target image')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='cpu or cuda to run.')
    return parser.parse_args()


class Tracker:
    def __init__(self, model_path, model_slug='r18', device='cuda'):
        self.device = device
        self.target_size = (127, 127)
        self.search_size = (255, 255)

        self.model = SiamCenterNet_ResNet(model_slug).to(device)
        ext = os.path.splitext(model_path)[-1]
        if ext == '.tar':
            self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        elif ext == '.pth':
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        out_shape = self.model.get_output_shape((1, 3, *self.target_size),
                                                (1, 3, *self.search_size))[0][-2:]
        self.out_sc = out_shape[0] / self.search_size[0], out_shape[1] / self.search_size[1]  # h, w

        self.target_image = None
        self.target = None

    def set_target(self, image, bbox, context_amount=0.5):
        img = get_template_image(image, bbox, self.target_size, context_amount)
        self.target_image = img
        self.target = torch.tensor(img, dtype=torch.float32).div(255.)
        self.target = self.target.permute(2, 0, 1)[None, ...].contiguous().cuda()

    def searh(self, image):
        assert self.target is not None, 'Must set the Target first!'

        h, w = image.shape[:2]
        img, sc, (pw, ph) = resize_padding(image, self.search_size[1], self.search_size[0],
                                           pad_value=tuple(map(int, image.mean(axis=(0, 1)))))
        img = torch.tensor(img, dtype=torch.float32).div(255.)
        img = img.permute(2, 0, 1)[None, ...].contiguous().cuda()

        with torch.no_grad():
            hm_pd, wh_pd, offset_pd = self.model(self.target.to(self.device), img.to(self.device))
            hm_pd, wh_pd, offset_pd = hm_pd.detach().cpu(), wh_pd.detach().cpu(), offset_pd.detach().cpu()

        bb_pd, score, _ = get_bboxs(hm_pd, wh_pd, offset_pd, 1, norm_wh=True)
        score = score.squeeze(1)

        # Scale bb to model input size.
        bb_pd = bb_pd[:, 0, :]
        bb_pd[:, [0, 2]] = torch.clip(bb_pd[:, [0, 2]] / self.out_sc[1], 0, self.search_size[1])
        bb_pd[:, [1, 3]] = torch.clip(bb_pd[:, [1, 3]] / self.out_sc[0], 0, self.search_size[0])
        # Scale bb to image size.
        bb_pd[:, [0, 2]] = torch.clip((bb_pd[:, [0, 2]] - pw) / sc, 0, w)
        bb_pd[:, [1, 3]] = torch.clip((bb_pd[:, [1, 3]] - ph) / sc, 0, h)

        return bb_pd.int().numpy(), score.numpy()


if __name__ == '__main__':
    args = parse_args()
    fil = args.video

    model = Tracker(model_path=args.model, model_slug=args.slug, device=args.device)

    image, bb = select_target(fil if args.image is None else args.image)
    model.set_target(image, bb, args.context_amount)

    cap = cv2.VideoCapture(fil)
    fps = 0
    cv2.imshow('target', model.target_image)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            bbs, scores = model.searh(frame)

            for i, bb in enumerate(bbs):
                clr = (0, 255, 0) if scores[i].item() > 0.2 else (0, 0, 255)
                frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), clr, 2)
                fream = cv2.putText(frame, 'score: {:.2f}%'.format(scores[i].item() * 100),
                                    (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)

            frame = cv2.putText(frame, 'FPS: {:.2f}'.format(1 / (time.time() - fps)),
                                (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('frame', frame)
            fps = time.time()
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


