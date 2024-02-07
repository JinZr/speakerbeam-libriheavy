# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import random
from collections import defaultdict

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


def read_enrollment_csv(csv_path):
    data = defaultdict(dict)
    with open(csv_path, "r") as f:
        f.readline()  # csv header

        for line in f:
            mix_id, utt_id, *aux = line.strip().split(",")
            aux_it = iter(aux)
            aux = [
                (auxpath, int(float(length))) for auxpath, length in zip(aux_it, aux_it)
            ]
            data[mix_id][utt_id] = aux
    return data


class LibriheavyMixInformed(Dataset):
    def __init__(
        self,
        mixscp: str,
        mix2spk: str,
        spk2src: str,
        spk2spk: str,
        enrollments: str,
        sample_rate=8000,
        segment=4,
        segment_aux=4,
        train=True,
    ):
        def lines_to_dict(file_path):
            res = dict()
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                key, value = line.strip().split()
                res[key] = value
            return res

        def lines_to_dict_spk2src(file_path):
            res = dict()
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                key, value = line.strip().split()
                res[key] = value.split(",")
            return res

        def lines_to_dict_spk2spk(file_path):
            res = dict()
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                key, value = line.strip().split()
                wavid, spkid = key.split("+")
                if wavid not in res:
                    res[wavid] = dict()
                    res[wavid][spkid] = value
                else:
                    res[wavid][spkid] = value
            return res

        def enrollments_to_dict(file_path):
            res = dict()
            with open(file_path, "r") as f:
                lines = f.readlines()
            for line in lines:
                key, _, spkid, enroll, index = line.strip().split(",")
                res[key] = (
                    spkid,
                    f"/star-data/rui/libriheavy_ovlp_src_reverb/dev_2spk/{enroll}/{index}.flac",
                )
            return res

        self.train = train
        self.sample_rate = sample_rate
        self.segment = segment
        self.segment_aux = segment_aux

        if self.train:
            self.dvec_list = lines_to_dict_spk2src(spk2src)
            self.target_wav_list = lines_to_dict_spk2spk(spk2spk)
            self.mix2spk = lines_to_dict(mix2spk)
            self.mix2spk_keys = list(self.mix2spk.keys())
            self.mixed_wav_list = lines_to_dict(mixscp)
            assert len(self.dvec_list) != 0, "no training file found"
        else:
            self.enrollments = enrollments_to_dict(enrollments)
            self.enrollments_keys = list(self.enrollments.keys())
            self.target_wav_list = lines_to_dict_spk2spk(spk2spk)
            self.mix2spk = lines_to_dict(mix2spk)
            self.mixed_wav_list = lines_to_dict(mixscp)
            self.segment = 8
            self.segment_aux = 8

    def __len__(self):
        if self.train:
            return len(self.mix2spk_keys)
        else:
            return len(self.enrollments)

    def _get_segment_start_stop(self, seg_len, length):
        if seg_len is not None:
            if length < seg_len:
                return 0, length
            start = random.randint(0, length - seg_len)
            stop = start + seg_len
        else:
            start = 0
            stop = None
        return start, stop

    def __getitem__(self, idx):
        if self.train:
            mix_key = self.mix2spk_keys[idx]
            mixed_path = self.mixed_wav_list[mix_key]
            target_spk = self.mix2spk[mix_key]
            target_path = self.target_wav_list[mix_key][target_spk]
            dvec_path = random.choice(self.dvec_list[target_spk])

            enroll, _ = librosa.load(dvec_path, sr=self.sample_rate)
            mixture, _ = librosa.load(mixed_path, sr=self.sample_rate)
            source, _ = librosa.load(target_path, sr=self.sample_rate)

            assert len(mixture) == len(source), f"{len(mixture)} != {len(source)}"

            start, stop = self._get_segment_start_stop(
                self.segment * self.sample_rate, len(mixture)
            )
            index = 0
            while np.count_nonzero(source[start:stop]) < (
                self.sample_rate * (self.segment - 1)
            ):
                index += 1
                if index > 10:
                    break
                else:
                    start, stop = self._get_segment_start_stop(
                        self.segment * self.sample_rate, len(mixture)
                    )
            mixture = torch.from_numpy(mixture[start:stop])
            source = torch.from_numpy(source[start:stop])

            e_start, e_stop = self._get_segment_start_stop(
                self.segment_aux * self.sample_rate, len(enroll)
            )
            enroll = torch.from_numpy(enroll[e_start:e_stop])

            mixture = torch.nn.functional.pad(
                mixture,
                (0, self.segment * self.sample_rate - mixture.shape[0]),
                value=0,
            )
            source = torch.nn.functional.pad(
                source,
                (0, self.segment * self.sample_rate - source.shape[0]),
                value=0,
            )
            enroll = torch.nn.functional.pad(
                enroll,
                (0, self.segment_aux * self.sample_rate - enroll.shape[0]),
                value=0,
            )

            assert mixture.shape == source.shape, f"{mixture.shape} != {source.shape}"
            assert mixture.shape == enroll.shape, f"{mixture.shape} != {enroll.shape}"
        else:
            enroll_key = self.enrollments_keys[idx]
            spkid, dvec_path = self.enrollments[enroll_key]
            enroll, _ = librosa.load(dvec_path, sr=self.sample_rate)

            source, _ = librosa.load(
                self.target_wav_list[enroll_key][spkid], sr=self.sample_rate
            )
            mixture, _ = librosa.load(
                self.mixed_wav_list[enroll_key], sr=self.sample_rate
            )

            # mixture = torch.from_numpy(mixture[: self.sample_rate * 12])
            # source = torch.from_numpy(source[: self.sample_rate * 12])
            # enroll = torch.from_numpy(enroll[: self.sample_rate * self.segment_aux])

            mixture = torch.from_numpy(mixture)
            source = torch.from_numpy(source)
            enroll = torch.from_numpy(enroll)

            # mixture = torch.nn.functional.pad(
            #     mixture,
            #     (0, self.sample_rate * 12 - mixture.shape[0]),
            #     value=0,
            # )
            # source = torch.nn.functional.pad(
            #     source,
            #     (0, self.sample_rate * 12 - source.shape[0]),
            #     value=0,
            # )
            # enroll = torch.nn.functional.pad(
            #     enroll,
            #     (0, self.segment_aux * self.sample_rate - enroll.shape[0]),
            #     value=0,
            # )

            assert mixture.shape == source.shape, f"{mixture.shape} != {source.shape}"
            # assert mixture.shape == enroll.shape, f"{mixture.shape} != {enroll.shape}"

        return mixture, source.unsqueeze(0), enroll

    def get_infos(self):
        return "LibriheavyMix"
