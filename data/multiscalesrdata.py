import os
import glob

from data import common
import pickle
import numpy as np
import imageio

import torch
import torch.utils.data as data


class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.scale = args.scale
        self.idx_scale = 0

        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr = self._scan()
        if args.ext.find('bin') >= 0:
            # Binary files are stored in 'bin' folder
            # If the binary file exists, load it. If not, make it.
            list_hr = self._scan()
            self.images_hr = self._check_and_load(
                args.ext, list_hr, self._name_hrbin()
            )
        else:
            if args.ext.find('img') >= 0 or benchmark:
                self.images_hr = list_hr
            elif args.ext.find('sep') >= 0:
                os.makedirs(
                    self.dir_hr.replace(self.apath, path_bin),
                    exist_ok=True
                )

                self.images_hr = []
                for h in list_hr:
                    b = h.replace(self.apath, path_bin)
                    b = b.replace(self.ext[0], '.pt')
                    self.images_hr.append(b)
                    self._check_and_load(
                        args.ext, [h], b, verbose=True, load=False
                    )

        if train:
            self.repeat = args.test_every // (len(self.images_hr) // args.batch_size)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        print(len(names_hr))

        return names_hr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = ('.png', '.png')

    def _name_hrbin(self):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_HR.pt'.format(self.split)
        )

    def _name_lrbin(self, scale):
        return os.path.join(
            self.apath,
            'bin',
            '{}_bin_LR_X{}.pt'.format(self.split, scale)
        )

    def _check_and_load(self, ext, l, f, verbose=True, load=True):
        if os.path.isfile(f) and ext.find('reset') < 0:
            if load:
                if verbose: print('Loading {}...'.format(f))
                with open(f, 'rb') as _f:
                    ret = pickle.load(_f)
                return ret
            else:
                return None
        else:
            if verbose:
                if ext.find('reset') >= 0:
                    print('Making a new binary: {}'.format(f))
                else:
                    print('{} does not exist. Now making binary...'.format(f))
            b = [{
                'name': os.path.splitext(os.path.basename(_l))[0],
                'image': imageio.imread(_l)
            } for _l in l]
            with open(f, 'wb') as _f:
                pickle.dump(b, _f)
            return b

    def __getitem__(self, idx):
        hr, filename = self._load_file(idx)
        hr = self.get_patch(hr)
        hr = [common.set_channel(img, n_channels=self.args.n_colors) for img in hr]
        hr_tensor = [common.np2Tensor(img, rgb_range=self.args.rgb_range)
                     for img in hr]

        return torch.stack(hr_tensor, 0), filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        if self.args.ext.find('bin') >= 0:
            filename = f_hr['name']
            hr = f_hr['image']
        else:
            filename, _ = os.path.splitext(os.path.basename(f_hr))
            if self.args.ext == 'img' or self.benchmark:
                hr = imageio.imread(f_hr)
            elif self.args.ext.find('sep') >= 0:
                with open(f_hr, 'rb') as _f:
                    hr = np.load(_f)[0]['image']

        return hr, filename

    def get_patch(self, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            out = []
            hr = common.augment(hr) if not self.args.no_augment else hr
            # extract two patches from each image
            for _ in range(2):
                hr_patch = common.get_patch(
                    hr,
                    patch_size=self.args.patch_size,
                    scale=scale
                )
                out.append(hr_patch)
        else:
            out = [hr]
        return out

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale

