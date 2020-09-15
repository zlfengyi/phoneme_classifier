
import glob
import data
import numpy as np
import librosa
from data.audio import preemphasis, amp_to_db
from settings.hparam import hparam as hp
from trainers.trainer import trainer_logger
from torch import FloatTensor, LongTensor

def get_dataset():
    wav_files = glob.glob(getattr(hp, "train").data_path)

    trainer_logger.info("Total %d wav files" % len(wav_files))
    dataset = []
    for index, file in enumerate(wav_files):
        wav_data, _ = librosa.load(file, sr=hp.defalut.sr)
        phn_file = file.replace("wav", "PHN").replace("WAV", "PHN")
        mfccs, phns = data.data_util.get_mfccs_phones(wav_data, phn_file,
                                       random_crop=False, trim=False)
        dataset.append([FloatTensor(mfccs), LongTensor(phns)])
        if index % 100 == 0:
            trainer_logger.info("%d files processed." % index)

    return dataset
