from collections import OrderedDict

import numpy as np

from .aspect import ASPECT_RATIOS, get_closest_ratio


def find_approximate_hw(hw, hw_dict, approx=0.8):
    for k, v in hw_dict.items():
        if hw >= v * approx:
            return k
    return None


def find_closet_smaller_bucket(t, t_dict, frame_interval):
    # process image
    if t == 1:
        if 1 in t_dict:
            return 1
        else:
            return None
    # process video
    for k, v in t_dict.items():
        if t >= v * frame_interval and v != 1:
            return k
    return None


def check_frames_len(T, probs):
    prob_sum = 0.0
    revalue = False
    new_probs = probs.copy()
    for t, prob in probs.items():
        if T < t:
            new_probs.pop(t, None)
            revalue = True
        else:
            prob_sum += prob

    if prob_sum == 0 and revalue:
        raise ValueError(f"The frame length {T} should larger than the assigned length.")
    
    if revalue:
        for t, prob in new_probs.items():
            new_probs[t] = prob / prob_sum

    return new_probs


class Bucket:
    def __init__(self, bucket_config):
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        # wrap config with OrderedDict
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_hw_probs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)
        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key][1].keys(), key=lambda x: x, reverse=True)
            bucket_hw_probs[key] = bucket_config[key][0]
            bucket_probs[key] = OrderedDict({k: bucket_config[key][1][k][0] for k in bucket_time_names})
            bucket_bs[key] = OrderedDict({k: bucket_config[key][1][k][1] for k in bucket_time_names})

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        bucket_id = OrderedDict()
        bucket_id_cnt = 0
        for k1, v1 in bucket_probs.items():
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            bucket_id[k1] = dict()
            for k2, _ in v1.items():
                t_criteria[k1][k2] = k2
                bucket_id[k1][k2] = bucket_id_cnt
                bucket_id_cnt += 1
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_hw_probs = bucket_hw_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket
        print(f"Number of buckets: {num_bucket}")
    

    def get_bucket_id(self, T, H, W, seed=None):
        chosen_hw_id = chosen_t_id = chosen_ar_id = None  # Init H, W, frames, and aspect ratio with None

        fail = True
        hw_probs_sum = 0.0
        rng = np.random.default_rng(seed)
        random_hw = rng.random()
        for hw_id, hw_probs in self.bucket_hw_probs.items():
            hw_probs_sum += hw_probs
            if random_hw <= hw_probs_sum:
                chosen_hw_id = hw_id
                fail = False
                break

        chosen_bucket_probs = self.bucket_probs[hw_id].copy()
        if T == 1 and 1 in chosen_bucket_probs.keys():  # if sample is an image
            chosen_t_id = 1
            fail = False
        elif T > 1:                                     # if sample is a video
            if 1 in chosen_bucket_probs.keys():
                chosen_bucket_probs.pop(1, None)

            # Check the frame length and revalue the chosen_bucket_probs
            chosen_bucket_probs = check_frames_len(T, chosen_bucket_probs)

            t_probs_sum = 0.0
            random_t = rng.random()
            for t, t_prob in chosen_bucket_probs.items():
                t_probs_sum += t_prob
                if random_t <= t_probs_sum:
                    chosen_t_id = t
                    fail = False
                    break

        if fail:
            return None

        # get aspect ratio id
        ar_criteria = self.ar_criteria[chosen_hw_id][chosen_t_id]
        chosen_ar_id = get_closest_ratio(H, W, ar_criteria)

        return chosen_hw_id, chosen_t_id, chosen_ar_id

    def get_thw(self, bucket_id):
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self):
        return self.num_bucket


def closet_smaller_bucket(value, bucket):
    for i in range(1, len(bucket)):
        if value < bucket[i]:
            return bucket[i - 1]
    return bucket[-1]


class BucketBackup:
    def __init__(self, bucket_config):
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        # wrap config with OrderedDict
        bucket_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)
        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)
            bucket_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in bucket_time_names})
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in bucket_time_names})

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        bucket_id = OrderedDict()
        bucket_id_cnt = 0
        for k1, v1 in bucket_probs.items():
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            bucket_id[k1] = dict()
            for k2, _ in v1.items():
                t_criteria[k1][k2] = k2
                bucket_id[k1][k2] = bucket_id_cnt
                bucket_id_cnt += 1
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        self.bucket_probs = bucket_probs
        self.bucket_bs = bucket_bs
        self.bucket_id = bucket_id
        self.hw_criteria = hw_criteria
        self.t_criteria = t_criteria
        self.ar_criteria = ar_criteria
        self.num_bucket = num_bucket
        print(f"Number of buckets: {num_bucket}")

    def get_bucket_id(self, T, H, W, frame_interval=1, seed=None):
        resolution = H * W
        approx = 0.8

        fail = True
        for hw_id, t_criteria in self.bucket_probs.items():
            if resolution < self.hw_criteria[hw_id] * approx:
                continue

            # if sample is an image
            if T == 1:
                if 1 in t_criteria:
                    rng = np.random.default_rng(seed + self.bucket_id[hw_id][1])
                    if rng.random() < t_criteria[1]:
                        fail = False
                        t_id = 1
                        break
                else:
                    continue

            # otherwise, find suitable t_id for video
            t_fail = True
            for t_id, prob in t_criteria.items():
                if T > t_id * frame_interval and t_id != 1:
                    t_fail = False
                    break
            if t_fail:
                continue

            # leave the loop if prob is high enough
            rng = np.random.default_rng(seed + self.bucket_id[hw_id][t_id])
            if prob == 1 or rng.random() < prob:
                fail = False
                break
        if fail:
            return None

        # get aspect ratio id
        ar_criteria = self.ar_criteria[hw_id][t_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, t_id, ar_id

    def get_thw(self, bucket_id):
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self):
        return self.num_bucket


def closet_smaller_bucket(value, bucket):
    for i in range(1, len(bucket)):
        if value < bucket[i]:
            return bucket[i - 1]
    return bucket[-1]