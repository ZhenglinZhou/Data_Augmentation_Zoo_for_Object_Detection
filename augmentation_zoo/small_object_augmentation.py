import numpy as np
import random

class copysmallobjects(object):
    def __init__(self, thresh=64*64, prob=0, copy_times=3, all_objects=False, one_object=False):
        """
        img = [height, width, 3]
        annot = [xmin, ymin, xmax, ymax, label]
        thresh：小目标边界

        """
        self.thresh = thresh
        self.prob = prob
        self.copy_times = copy_times
        self.all_objects = all_objects
        self.one_object = one_object
        if self.all_objects or self.one_object:
            self.copy_times = 1

    def issmallobject(self, h, w):
        if h * w <= self.thresh:
            return True
        else:
            return False

    def compute_overlap(self, annot_a, annot_b):
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, (right_min-left_max)) * max(0, (bottom_min-top_max))
        if inter != 0:
            return True
        else:
            return False

    def donot_overlap(self, copy_annot, annots):
        for annot in annots:
            if self.compute_overlap(copy_annot, annot): return False
        return True

    def create_copy_annot(self, h, w, annot, annots):
        annot = annot.astype(np.int)
        annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
        random_x, random_y = np.random.randint(int(annot_w/2), int(w-annot_w/2)), \
                             np.random.randint(int(annot_h/2), int(h-annot_h/2))

        # print(random_x, random_y)
        if np.int(random_x - annot_w/2) < 0 or np.floor(random_x + annot_w/2) > w or \
                np.int(random_y - annot_h/2) < 0 or np.floor(random_y + annot_h/2) > h:
            return self.create_copy_annot(h ,w, annot, annots)

        copy_annot = list()
        xmin, ymin = random_x - annot_w/2, random_y - annot_h/2
        xmax, ymax = xmin + annot_w, ymin + annot_h
        copy_annot.append(xmin), copy_annot.append(ymin)
        copy_annot.append(xmax), copy_annot.append(ymax)
        copy_annot.append(annot[4])

        copy_annot = np.array(copy_annot).astype(np.int)

        if self.donot_overlap(copy_annot, annots) is False:
            return self.create_copy_annot(h, w, annot, annots)

        return copy_annot

    def add_patch_in_img(self, annot, copy_annot, image):
        copy_annot = copy_annot.astype(np.int)
        image[annot[1]:annot[3], annot[0]:annot[2], :] = image[copy_annot[1]:copy_annot[3], copy_annot[0]:copy_annot[2], :]
        return image

    def __call__(self, sample):
        if self.all_objects and self.one_object: return sample
        if np.random.rand() < self.prob: return sample

        img, annots = sample['img'], sample['annot']
        h, w, l = img.shape[0], img.shape[1], annots.shape[0]

        copy_object_num = np.random.randint(0, l)
        if self.all_objects:
            copy_object_num = l
        if self.one_object:
            copy_object_num = 1
        random_list = random.sample(range(l), copy_object_num)
        select_annots = annots[random_list, :]
        annots = annots.tolist()
        for idx in range(copy_object_num):
            annot = select_annots[idx]
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]

            if self.issmallobject(annot_h, annot_w) is False: continue

            for i in range(self.copy_times):
                new_annot = self.create_copy_annot(h, w, annot, annots)
                img = self.add_patch_in_img(new_annot, annot, img)
                annots.append(new_annot)

        return {'img': img, 'annot': np.array(annots)}

