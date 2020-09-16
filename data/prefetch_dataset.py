
import glob
import numpy as np
import librosa
from settings.hparam import hparam as hp
from trainers.trainer import trainer_logger
from data.data_utils import get_mfccs_phones
from torch.utils.data import Dataset
from torch import FloatTensor, LongTensor


def get_cache_data(mode):
    wav_files = glob.glob(getattr(hp, mode).data_path)

    trainer_logger.info("Total %d wav files" % len(wav_files))
    result = []
    for index, file in enumerate(wav_files):
        wav_data, _ = librosa.load(file, sr=hp.default.sr)
        phn_file = file.replace("wav", "PHN").replace("WAV", "PHN")
        mfccs, phns = get_mfccs_phones(wav_data, phn_file,
                                       random_crop=True, trim=False)
        result.append([FloatTensor(mfccs), LongTensor(phns)])
        if index % 100 == 0:
            trainer_logger.info("%d files processed." % index)

    return result

class VoiceDataset(Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        self.cache_data = get_cache_data(mode)

    def __getitem__(self, idx):
        return self.cache_data[idx][0], self.cache_data[idx][1]

    def __len__(self):
        return len(self.cache_data)
