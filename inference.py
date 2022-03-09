import yaml
import numpy as np

from graphs.deep_blstm import create_blstm
from data.vocabs import GRAPHEMES, PHONEMES
from data.data_prepare import get_sequence_one_hots


def get_inputs(words, tokens_gr):
    inputs = []
    for word in words:
        graphemes = list(word)
        inputs.append(get_sequence_one_hots(graphemes, tokens_gr))
    return np.array(inputs, dtype=np.float32)


def get_spells(predicts, tokens_ph):
    spells = []
    for sequence in predicts:
        phonemes = []
        for phoneme_distrib in sequence:
            phonemes.append(tokens_ph[phoneme_distrib.argmax()])
        spells.append(''.join(phonemes))
    return spells


if __name__ == '__main__':
    with open('configs/inference_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open('models/deep_blstm/config_deep_blstm.yaml', 'r') as f:
        config_model = yaml.load(f, Loader=yaml.FullLoader)
    model = create_blstm(config_model)
    model.load_weights(config['path_checkpoint'])
    examples = ['привет']
    inputs = get_inputs(examples, GRAPHEMES)
    predicts = model.predict(inputs)
    spells = get_spells(predicts, PHONEMES)
    print(spells)
