import numpy as np
from PIL import Image


class GridMask(object):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        if np.random.rand() > self.prob:
            return sample
        h = img.shape[0]
        w = img.shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        # d = self.d
        #        self.l = int(d*self.ratio+0.5)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask.astype(np.float), axis=2)
        mask = np.tile(mask, [1, 1, 3])
        if self.offset:
            offset = np.float(2 * (np.random.rand(h, w) - 0.5))
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask
        return {'img': img, 'annot': annots}
