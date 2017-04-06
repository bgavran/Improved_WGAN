from src.utils import *
from scipy.misc import imresize

datapath = ProjectPath("log")


class Data:
    def __init__(self, img_size, crop_size=128):
        self.img_size = img_size
        self.crop_size = crop_size
        images_folder_path = os.path.join(datapath.base, "data", "lfw-deepfunneled")
        self.images_path = []
        for (dirpath, dirnames, fnames) in os.walk(images_folder_path):
            for fname in fnames:
                self.images_path.append(os.path.join(dirpath, fname))

        self.images_path = self.images_path[:500]  # training only on 500 images for now, for speed and memory reasons
        self.images = np.zeros((len(self.images_path), self.img_size, self.img_size, 3))
        for i, img_path in enumerate(self.images_path):
            if i % 100 == 0:
                print(i)
            self.images[i] = self.get_image(img_path, resize_dim=self.img_size)

    def next_batch_real(self, size):
        locations = np.random.randint(0, len(self.images_path), size)
        return np.array([self.images[loc] for loc in locations])

    def next_batch_fake(self, batch_size, z_size):
        return np.random.rand(batch_size, z_size)

    def get_image(self, path, resize_dim=None):
        img = Data.read_image(path)
        img = Data.center_crop(img, crop_h=self.crop_size)
        if resize_dim is not None:
            rev_rat = self.crop_size / resize_dim  # ratio
            assert rev_rat.is_integer()
            rev_rat = int(rev_rat)
            img = img[::rev_rat, ::rev_rat]
        return img

    @staticmethod
    def read_image(path):
        # dividing with 256 because we need to get it in the [0, 1] range
        return scipy.misc.imread(path).astype(np.float) / 256

    @staticmethod
    def center_crop(x, crop_h, crop_w=None):
        if crop_w is None:
            crop_w = crop_h
        h, w = x.shape[:2]
        j = round((h - crop_h) / 2)
        i = round((w - crop_w) / 2)
        return x[j:j + crop_h, i:i + crop_w]
