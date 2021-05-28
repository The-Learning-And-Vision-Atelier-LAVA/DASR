from model.blindsr import BlindSR
import torch
import numpy as np
import imageio
import argparse
import os
import utility
import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='D:/LongguangWang/Data/test.png',
                        help='image directory')
    parser.add_argument('--scale', type=str, default='2',
                        help='super resolution scale')
    parser.add_argument('--resume', type=int, default=600,
                        help='resume from specific checkpoint')
    parser.add_argument('--blur_type', type=str, default='iso_gaussian',
                        help='blur types (iso_gaussian | aniso_gaussian)')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.blur_type == 'iso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_iso'
    elif args.blur_type == 'aniso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_aniso'

    # path to save sr images
    save_dir = dir + '/results'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    DASR = BlindSR(args).cuda()
    DASR.load_state_dict(torch.load(dir + '/model/model_' + str(args.resume) + '.pt'), strict=False)
    DASR.eval()

    lr = imageio.imread(args.img_dir)
    lr = np.ascontiguousarray(lr.transpose((2, 0, 1)))
    lr = torch.from_numpy(lr).float().cuda().unsqueeze(0).unsqueeze(0)

    # inference
    sr = DASR(lr[:, 0, ...])
    sr = utility.quantize(sr, 255.0)

    # save sr results
    img_name = args.img_dir.split('.png')[0].split('/')[-1]
    sr = np.array(sr.squeeze(0).permute(1, 2, 0).data.cpu())
    sr = sr[:, :, [2, 1, 0]]
    cv2.imwrite(save_dir + '/' + img_name + '_sr.png', sr)


if __name__ == '__main__':
    with torch.no_grad():
        main()