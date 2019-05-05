import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from models import *
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
import os

# Parameters
gpu = torch.cuda.is_available()
batch_size = 3
data_folder = '../data'  # folder with data files saved by create_input_files.py
dataset_name = 'coco_val2014'
rst_path = '../results'
max_cap_len = 100
min_word_freq = 3
num_caption_per_image = 1
epoch = 16
checkpoint = '../save/BEST_checkpoint_%s_epoch_%d_max_cap_%d_min_word_freq_%d.pth.tar' % (dataset_name, epoch, max_cap_len, min_word_freq)  # model checkpoint
word_map_file = "%s/%s/%s_WORDMAP_min_word_freq_%d.json" % (data_folder, dataset_name, dataset_name, min_word_freq)  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if gpu else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

if not os.path.exists(rst_path):
    os.makedirs(rst_path)

# Load model
detector = Detector(dataset_name)
detector.fine_tune(False)
detector.eval()
if gpu:
    checkpoint = torch.load(checkpoint)
else:
    checkpoint = torch.load(checkpoint, map_location='cpu')
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
encoder = checkpoint['encoder']
encoder = encoder.to(device)
encoder.eval()

# Load word map (word2ix)
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    data_specs = "max_cap_%d_min_word_freq_%d" % (max_cap_len, min_word_freq)
    loader = torch.utils.data.DataLoader(
        CaptionDetectionDataset(data_folder, dataset_name, data_specs, 'test', transform=transforms.Compose([normalize])),
    batch_size=batch_size)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()
    rst_hypo = dict()
    rst_refer = dict()

    for i, (imgs, caps, caplens, allcaps, imgname, imgs_d) in enumerate(tqdm(loader)):

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        stacked_imgs, num_boxes, _ = detector(imgs, imgs_d)
        features, sorted_idx, num_boxes = encoder(stacked_imgs, num_boxes)
        caps = caps[sorted_idx]
        caplens = caplens[sorted_idx]
        # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(features, caps, caplens, num_boxes)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)


        # References
        imgname = np.asarray(imgname)[sorted_idx][sort_ind.numpy()]
        allcaps = allcaps[sorted_idx][sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

            img_captions_text = [[rev_word_map[w] for w in cap] for cap in img_captions]
            rst_refer[imgname[j]] = img_captions_text

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypo_captions = [[w for w in pred if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}] for pred in preds]
        hypotheses.extend(hypo_captions)
        hypo_captions_text = [[rev_word_map[w] for w in hypo_caption] for hypo_caption in hypo_captions]
        for idx, name in enumerate(imgname):
            rst_hypo[name] = hypo_captions_text[idx]

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    print('\n * BLEU-4 - {bleu}\n'.format(bleu=bleu4))

    with open("%s/%s_REFER_%s.json"
              % (rst_path, dataset_name, data_specs), 'w') as j:
        json.dump(rst_refer, j)

    with open("%s/%s_HYPO_%s.json"
              % (rst_path, dataset_name, data_specs), 'w') as j:
        json.dump(rst_hypo, j)

    return bleu4


if __name__ == '__main__':
    beam_size = 3
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
