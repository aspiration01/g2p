import yaml
import numpy as np
from graphs.deep_blstm import create_blstm
from tensorflow.keras.models import load_model

from data_prepare import get_tokens, get_sequence_one_hots


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
    with open('configs/train_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open('configs/deep_blstm.yaml', 'r') as f:
        config_model = yaml.load(f, Loader=yaml.FullLoader)
    model_name = 'g2p_model_deep_blstm'
    path_load_model = f"{config['path_to_work_dir']}/{model_name}"
    checkpoint_path = f'{config["path_to_work_dir"]}/ckpt_{model_name}/best.ckpt'
    # model = load_model(path_load_model)
    model = create_blstm(config_model)
    model.load_weights(checkpoint_path)
    path_to_tokens_grapheme = f'{config["path_to_sources"]}/data/graphemes.txt'
    path_to_tokens_phoneme = f'{config["path_to_sources"]}/data/phonemes.txt'
    tokens_grapheme, tokens_phoneme = get_tokens(path_to_tokens_grapheme, path_to_tokens_phoneme)
    examples = ['привет']
    inputs = get_inputs(examples, tokens_grapheme)
    predicts = model.predict(inputs)
    spells = get_spells(predicts, tokens_phoneme)
    print(spells)
