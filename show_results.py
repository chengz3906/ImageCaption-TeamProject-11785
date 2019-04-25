import json

hypo_path = '../results/flickr8k_HYPO_max_cap_100_min_word_freq_3.json'
refer_path = '../results/flickr8k_REFER_max_cap_100_min_word_freq_3.json'


def main():
    with open(hypo_path, 'r') as j:
        hypo_dict = json.load(j)

    with open(refer_path, 'r') as j:
        refer_dict = json.load(j)

    for img in hypo_dict:
        hypo = hypo_dict[img]
        refer = refer_dict[img]
        print(img)
        print(hypo)
        print(refer)


if __name__ == '__main__':
    main()