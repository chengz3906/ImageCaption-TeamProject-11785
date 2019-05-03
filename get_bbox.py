import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Detector, Decoder, EncoderForDetector
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from datetime import datetime

# Data parameters
data_folder = '../data'  # folder with data files saved by create_input_files.py
dataset_name = 'coco_val2014'  # base name shared by data files
max_cap_len = 100
min_word_freq = 3
num_caption_per_image = 5

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 3
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 1  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = "%s/%s/%s_WORDMAP_min_word_freq_%d.json" % (data_folder, dataset_name, dataset_name, min_word_freq)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    detector = Detector(dataset_name)
    detector.fine_tune(False)
    detector = detector.to(device)

    # Custom dataloaders
    data_specs = "max_cap_%d_min_word_freq_%d" % (max_cap_len, min_word_freq)
    train_loader = torch.utils.data.DataLoader(
        CaptionDetectionDataset(data_folder, dataset_name, data_specs, 'train'),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDetectionDataset(data_folder, dataset_name, data_specs, 'val'),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)


    # One epoch's training
    get_train(train_loader=train_loader,
              detector=detector)

    # One epoch's validation
    get_validate(val_loader=val_loader,
                 detector=detector)


def get_train(train_loader, detector):

    detector.eval()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time

    start = time.time()

    all_boxes = []
    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)

        # Forward prop.
        stacked_imgs, num_boxes, boxes = detector(imgs)
        all_boxes += boxes

        # Keep track of metrics
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('{0}\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})'.format(datetime.now(),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time))
    with open("%s/%s/%s_train_boxes.json"
              % (data_folder, dataset_name, dataset_name), 'w') as j:
        for i in range(len(all_boxes)):
            all_boxes[i] = all_boxes[i].tolist()
        json.dump(all_boxes, j)


def get_validate(val_loader, detector):
    detector.eval()

    batch_time = AverageMeter()

    start = time.time()

    all_boxes = []
    # Batches
    for i, (imgs, caps, caplens, allcaps, imgname) in enumerate(val_loader):

        # Move to device, if available
        imgs = imgs.to(device)

        # Forward prop.
        stacked_imgs, num_boxes, boxes = detector(imgs)
        all_boxes += boxes

        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(i, len(val_loader),
                                                                                  batch_time=batch_time))
    with open("%s/%s/%s_val_boxes.json"
              % (data_folder, dataset_name, dataset_name), 'w') as j:
        for i in range(len(all_boxes)):
            all_boxes[i] = all_boxes[i].tolist()
        json.dump(all_boxes, j)

if __name__ == '__main__':
    main()
