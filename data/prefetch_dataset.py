
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

logger = utils.get_logger('prefetch_dataset')

def get_cache_data(mode):
    wav_files = glob.glob(getattr(hp, mode).data_path)

    logger.info("Total %d wav files" % len(wav_files))
    result = []
    for index, file in enumerate(wav_files):
        if index % 100 == 0:
            logger.info("%d files processed." % index)

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
    def __init__(self, mode='train'):
        self.mode = mode
        self.cache_data = get_cache_data(mode)

    def __getitem__(self, idx):
        return FloatTensor(self.cache_data[idx]['mfccs']), LongTensor(self.cache_data[idx]['phns'])

    def __len__(self):
        return len(self.cache_data)
