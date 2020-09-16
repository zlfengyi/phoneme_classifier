
import glob
import numpy as np
import librosa
from settings.hparam import hparam as hp
from trainers.trainer import trainer_logger
from data.data_utils import get_mfccs_phones
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor
import os
import utils

def get_cache_data(mode):
    wav_files = glob.glob(getattr(hp, mode).data_path)

    print("Total %d wav files" % len(wav_files))
    result = []
    for index, file in enumerate(wav_files):
        if index % 100 == 0:
            print("%d files processed." % index)

        cache_file = file.replace("wav", hp.cache_version).replace("WAV", hp.cache_version)
        if os.path.exists(cache_file):
            mp = np.load(cache_file)
            result.append(mp)
            continue
        wav_data, _ = librosa.load(file, sr=hp.default.sr)
        phn_file = file.replace("wav", "PHN").replace("WAV", "PHN")
        mfccs, phns = get_mfccs_phones(wav_data, phn_file,
                                       random_crop=False, trim=False)
        mp = {'mfccs': mfccs, 'phns': phns}
        np.save(cache_file, mp)
        result.append(mp)

    return result

class Train1Dataset(Dataset):
    def __init__(self, mode='train', random_crop=True):
        self.mode = mode
        self.random_crop = random_crop
        self.cache_data = get_cache_data(mode)

    def __getitem__(self, idx):
        # Random crop
        if self.random_crop:
            n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1
            start = np.random.choice(range(int(np.maximum(1, len(self.cache_data[idx]['mfccs']) - n_timesteps))), 1)[0]
            end = start + n_timesteps
        else:
            start = 0
            end = len(self.cache_data[idx]['mfccs'])

        return FloatTensor(self.cache_data[idx]['mfccs'][start:end]), \
               LongTensor(self.cache_data[idx]['phns'][start:end])

    def __len__(self):
        return len(self.cache_data)
