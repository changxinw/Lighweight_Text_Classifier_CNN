import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
import random
import numpy as np
from transformers import BertTokenizer

SEED = 1234

def load_dataset(data_name, device, BATCH_SIZE = 64, include_lengths = False, batch_first = True):
    TEXT = data.Field(tokenize = 'spacy', 
                      tokenizer_language = 'en_core_web_sm',
                      batch_first = batch_first,
                      include_lengths = include_lengths)
    LABEL = data.LabelField(dtype = torch.float)
    if data_name == 'IMDB':
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        train_data, valid_data = train_data.split(random_state = random.seed(SEED), split_ratio = 0.9)
        MAX_VOCAB_SIZE = 25_000
    if data_name == 'TREC':
        train_data, test_data = datasets.TREC.splits(TEXT, LABEL)
        train_data, valid_data = train_data.split(random_state = random.seed(SEED), split_ratio = 0.9)
        MAX_VOCAB_SIZE = 6000
    if data_name == 'SST':
        train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL,train_subtrees=True,filter_pred=lambda ex: ex.label != 'neutral')
        MAX_VOCAB_SIZE = 50000
    TEXT.build_vocab(train_data, 
                     max_size = MAX_VOCAB_SIZE, 
                     vectors = "glove.840B.300d", 
                     unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE, 
        device = device)
    return train_iterator, valid_iterator, test_iterator, TEXT, LABEL


def load_dataset_bert(data_name, device, BATCH_SIZE = 64, include_lengths = False, batch_first = True):
    # pass
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    def tokenize_and_cut(sentence, max_input_length = max_input_length):
        tokens = tokenizer.tokenize(sentence) 
        tokens = tokens[:max_input_length-2]
        return tokens
    TEXT = data.Field(batch_first = batch_first,
                      use_vocab = False,
                      tokenize = tokenize_and_cut,
                      preprocessing = tokenizer.convert_tokens_to_ids,
                      init_token = init_token_idx,
                      eos_token = eos_token_idx,
                      pad_token = pad_token_idx,
                      unk_token = unk_token_idx)
    LABEL = data.LabelField(dtype = torch.float)
    if data_name == 'IMDB':
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        train_data, valid_data = train_data.split(random_state = random.seed(SEED), split_ratio = 0.9)
        MAX_VOCAB_SIZE = 25_000
    if data_name == 'TREC':
        train_data, test_data = datasets.TREC.splits(TEXT, LABEL)
        train_data, valid_data = train_data.split(random_state = random.seed(SEED), split_ratio = 0.9)
        MAX_VOCAB_SIZE = 6000
    if data_name == 'SST':
        train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL,train_subtrees=True,filter_pred=lambda ex: ex.label != 'neutral')
        MAX_VOCAB_SIZE = 50000
    LABEL.build_vocab(train_data)
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE, 
        device = device)
    return train_iterator, valid_iterator, test_iterator, TEXT, LABEL
