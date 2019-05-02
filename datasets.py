import torch
from torch.utils.data import Dataset
import h5py
import cv2
import json
import numpy as np
from torchvision.transforms.functional import resize
from torchvision.transforms import ToPILImage, ToTensor
from model.utils.config import cfg
from model.utils.blob import im_list_to_blob

cv2.setNumThreads(0)


class CaptionDetectionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, data_specs, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

        self.split = split
        assert self.split in {'train', 'val', 'test'}

        # Open hdf5 file where images are stored
        self.h = h5py.File("%s/%s/%s_%s_%s.hdf5" % (data_folder, data_name, data_name, split, data_specs), 'r')
        self.imgs = self.h['image_data']

        # Captions per image
        self.cpi = self.h.attrs['num_caption_per_image']

        # Load encoded captions (completely into memory)
        with open("%s/%s/%s_%s_CAPTIONS_%s.json" % (data_folder, data_name, data_name, split, data_specs), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open("%s/%s/%s_%s_CAPLENS_%s.json" % (data_folder, data_name, data_name, split, data_specs), 'r') as j:
            self.caplens = json.load(j)

        # Load image names (completely into memory)
        with open("%s/%s/%s_%s_IMAGENAMES.json" % (data_folder, data_name, data_name, split), 'r') as j:
            self.imgnames = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = np.array(self.imgs[i // self.cpi]).transpose(1, 2, 0)
        img = img[:, :, ::-1]
        blobs, im_scales = self._get_image_blob(img)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs

        im_data_pt = torch.from_numpy(im_blob)
        img = torch.squeeze(im_data_pt.permute(0, 3, 1, 2))

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        imgname = self.imgnames[i // self.cpi]

        if self.split is 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions, imgname

    def __len__(self):
        return self.dataset_size


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, data_specs, split, scale=256, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.scale = scale
        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

        self.split = split
        assert self.split in {'train', 'val', 'test'}

        # Open hdf5 file where images are stored
        self.h = h5py.File("%s/%s/%s_%s_%s.hdf5" % (data_folder, data_name, data_name, split, data_specs), 'r')
        self.imgs = self.h['image_data']

        # Captions per image
        self.cpi = self.h.attrs['num_caption_per_image']

        # Load encoded captions (completely into memory)
        with open("%s/%s/%s_%s_CAPTIONS_%s.json" % (data_folder, data_name, data_name, split, data_specs), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open("%s/%s/%s_%s_CAPLENS_%s.json" % (data_folder, data_name, data_name, split, data_specs), 'r') as j:
            self.caplens = json.load(j)

        # Load image names (completely into memory)
        with open("%s/%s/%s_%s_IMAGENAMES.json" % (data_folder, data_name, data_name, split), 'r') as j:
            self.imgnames = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        if self.scale != 256:
            img = self.to_pil(img)
            img = resize(img, (self.scale, self.scale))
            img = self.to_tensor(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        imgname = self.imgnames[i // self.cpi]

        if self.split is 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions, imgname

    def __len__(self):
        return self.dataset_size