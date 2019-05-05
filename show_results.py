import json

hypo_path = '../results/coco_val2014_HYPO_max_cap_100_min_word_freq_3.json'
refer_path = '../results/coco_val2014_REFER_max_cap_100_min_word_freq_3.json'

hypo_b_path = '../results/coco_val2014_baseline_HYPO_max_cap_100_min_word_freq_3.json'
result_txt = "../results/result.txt"

def main():
    with open(hypo_path, 'r') as j:
        hypo_dict = json.load(j)

    with open(hypo_b_path, 'r') as j:
        hypo_b_dict = json.load(j)

    with open(refer_path, 'r') as j:
        refer_dict = json.load(j)

    with open(result_txt, 'w') as r:
        for img in hypo_dict:
            hypo = hypo_dict[img]
            hypo_b = hypo_b_dict[img]
            refer = refer_dict[img]
            r.write("image name: %s\n" % img)
            r.write("model: %s\n" % " ".join(hypo))
            r.write("baseline: %s\n" % " ".join(hypo_b))
            r.write("ground truth: %s\n\n" % " ".join(refer[0]))


if __name__ == '__main__':
    main()