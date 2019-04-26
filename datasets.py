import torch
from torch.utils.data import Dataset
import h5py
import json
from torchvision.transforms.functional import resize
from torchvision.transforms import ToPILImage, ToTensor

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
