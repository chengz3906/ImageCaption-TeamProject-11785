import h5py
import json
from collections import Counter
from tqdm import tqdm
from random import choice, sample, seed
from scipy.misc import imread, imresize
import numpy as np

max_cap_len = 100
min_word_freq = 3
num_caption_per_image = 1
available_datasets = ['coco_val2014', 'flickr8k', 'flickr30k']


def preprocess(dataset_name, image_path, caption_path, output_path):
    assert dataset_name in available_datasets
    image_path = image_path[:-1] if image_path[-1] == '/' else image_path
    caption_path = caption_path[:-1] if caption_path[-1] == '/' else caption_path
    with open("%s/dataset_%s.json" % (caption_path, dataset_name)) as f_cap:
        caption_json = json.load(f_cap)['annotations']
    with open("%s/dataset_%s.json" % (caption_path, dataset_name)) as f_cap:
        image_json = json.load(f_cap)['images']

    # make a caption dict with key-value pair as image_id-caption(s)
    caption_dict = {}
    for caption_entry in caption_json:
        image_id = caption_entry['image_id']
        caption_in_words = caption_entry['caption'].strip().replace('.', ' ').lower().split()
        caption_dict[image_id] = caption_in_words

    # Read the captions and bind them with corresponding images
    image_with_caption = dict()
    word_count = Counter()
    for image_entry in image_json:
        image_name = image_entry['file_name']
        image_id = image_entry['id']

        caption_words = caption_dict[image_id]
        if not caption_words:
            continue
        if len(caption_words) > max_cap_len:
            continue
        word_count.update(caption_words)
        image_with_caption[image_name] = [caption_words]

    # make train-val-test split (6:1:1)
    train_image_with_caption = {}
    val_image_with_caption = {}
    test_image_with_caption = {}

    seed(11785)
    # train set
    train_len = int(len(image_with_caption) * 6 / 8)
    keys = sample(list(dict.fromkeys(image_with_caption)), train_len)
    for k in keys:
        train_image_with_caption[k] = image_with_caption[k]
        del image_with_caption[k]
    # val set
    val_len = int(len(image_with_caption) / 2)
    keys = sample(list(dict.fromkeys(image_with_caption)), val_len)
    for k in keys:
        val_image_with_caption[k] = image_with_caption[k]
        del image_with_caption[k]
    # test set
    for k in image_with_caption.keys():
        test_image_with_caption[k] = image_with_caption[k]
    assert image_with_caption == test_image_with_caption
    print(len(train_image_with_caption), len(val_image_with_caption), len(test_image_with_caption))

    # Form the vocabulary and word map
    vocabulary = [word for word in word_count if word_count[word] >= min_word_freq]
    word_map = {k: v+1 for v, k in enumerate(vocabulary)}
    word_map['<pad>'] = 0
    word_map['<unk>'] = len(word_map)
    word_map['<start>'] = len(word_map)
    word_map['<end>'] = len(word_map)
    with open("%s/%s/%s_WORDMAP_min_word_freq_%d.json" % (output_path, dataset_name, dataset_name, min_word_freq), 'w') as j:
        json.dump(word_map, j)
    
    # Organize caption data and save them with hdf5
    data_splits = [(train_image_with_caption, 'train'), 
                   (val_image_with_caption, 'val'), 
                   (test_image_with_caption, 'test')]
    for image_with_caption, split in data_splits:
        with h5py.File("%s/%s/%s_%s_max_cap_%d_min_word_freq_%d.hdf5"
                       % (output_path, dataset_name, dataset_name,
                          split, max_cap_len, min_word_freq), 'a') as h5f:
            dset = h5f.create_dataset("image_data",
                                      (len(image_with_caption), 3, 256, 256),
                                      dtype='uint8')
            h5f.attrs['num_caption_per_image'] = num_caption_per_image

            # Load images into hdf5 dataset
            complete_captions = list()
            caption_lengths = list()
            for i, image_name in enumerate(tqdm(image_with_caption.keys())):
                cur_captions = image_with_caption[image_name]

                # Sample captions
                if len(cur_captions) < num_caption_per_image:
                    captions = cur_captions + [choice(cur_captions)
                                               for _ in range(num_caption_per_image - len(cur_captions))]
                else:
                    captions = sample(cur_captions, k=num_caption_per_image)

                # Read images
                image_fullpath = "%s/%s/%s" % (image_path, dataset_name, image_name)
                img = imread(image_fullpath)
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)

                # Save image to HDF5 file
                dset[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    com_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_cap_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    complete_captions.append(com_c)
                    caption_lengths.append(c_len)

            # Save encoded captions and their lengths to JSON files
            with open("%s/%s/%s_%s_CAPTIONS_max_cap_%d_min_word_freq_%d.json"
                      % (output_path, dataset_name, dataset_name, split, max_cap_len, min_word_freq), 'w') as j:
                json.dump(complete_captions, j)

            with open("%s/%s/%s_%s_CAPLENS_max_cap_%d_min_word_freq_%d.json"
                      % (output_path, dataset_name, dataset_name, split, max_cap_len, min_word_freq), 'w') as j:
                json.dump(caption_lengths, j)

            with open("%s/%s/%s_%s_IMAGENAMES.json"
                      % (output_path, dataset_name, dataset_name, split), 'w') as j:
                json.dump(list(image_with_caption.keys()), j)


def main():
    preprocess(dataset_name='coco_val2014',
               image_path='../image_rawdata',
               caption_path='../caption_rawdata',
               output_path='../data')


if __name__ == '__main__':
    main()
