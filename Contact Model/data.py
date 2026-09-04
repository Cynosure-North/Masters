import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np
import random
from random import randint
from torch.nn.utils.rnn import pad_sequence
import csv
import ast

np.random.seed(1)
random.seed(1)
chars = ['[PAD]', '[MASK]', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', \
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.']
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


class DataVariableLength(Dataset):
    """Data loader for the long-term decoder.
    DataLongTerm parses the raw data and extracts
    (seq. of user input, g.t. seq. of characters) pairs of each full sentence.
    """
    def __init__(self, csv_path, min_length=13, full_sentence=False, augment=False):
        df = pd.read_csv(csv_path)
        self.full_sentence = full_sentence
        self.min_length = min_length
        data_file = df[df['length'] >= self.min_length]
        self.df = data_file
        self.data_length = self.df.shape[0]
        self.min_x = 0
        self.min_y = 0
        self.augment = augment

    def process_csv(self, data, idx):
        one_sample = data.iloc[idx]
        selected_length = random.randint(self.min_length, one_sample['length'])
        # width_one = one_sample['width']
        # height_one = one_sample['height']
        width_one = one_sample['width'] * 1.1
        height_one = one_sample['height'] * 1.4

        var_x = width_one - max([float(x) for x in one_sample['x_list'].split(',')])
        var_y = height_one - max([float(y) for y in one_sample['y_list'].split(',')])
        shift_x = random.randint(-3, 3)
        shift_y = random.randint(-3, 3)
        # if var_x < self.min_x:
        #     self.min_x = var_x
        #     print('x: ' + str(var_x))
        # if var_y < self.min_y:
        #     self.min_y = var_y
        #     print('y: ' + str(var_y))

        if self.full_sentence:
            cropped_chars = one_sample['sentence']
            norm_x = np.expand_dims(np.array([(float(x) + random.uniform(-3, 3)) / width_one if random.random() < 0.3
                                              else (float(x)) / width_one for x in one_sample['x_list'].split(',')]), 1)
            norm_y = np.expand_dims(np.array([(float(y) + random.uniform(-3, 3)) / height_one if random.random() < 0.3
                                              else (float(y)) / height_one for y in one_sample['y_list'].split(',')]), 1)
        else:
            cropped_chars = one_sample['sentence'][0:selected_length]
            if self.augment:
                random.seed(0)
                norm_x = np.expand_dims(np.array([(float(x) + random.uniform(-3, 3)) / width_one
                                                  if random.random() < 0.3 else (float(x)) / width_one for x in one_sample['x_list'].split(',')][0:selected_length]), 1)
                norm_y = np.expand_dims(np.array([(float(y) + random.uniform(-3, 3)) / height_one
                                                  if random.random() < 0.3 else (float(y)) / height_one for y in one_sample['y_list'].split(',')][0:selected_length]), 1)
            else:
                norm_x = np.expand_dims(np.array(
                    [float(x) / width_one for x in one_sample['x_list'].split(',')][
                    0:selected_length]), 1)
                norm_y = np.expand_dims(np.array(
                    [float(y) / height_one for y in one_sample['y_list'].split(',')][
                    0:selected_length]), 1)
        char_np = np.array([char_to_idx[char] for char in cropped_chars])
        input_char_np = np.expand_dims(char_np, 1)
        # char_idx_list = np.copy(char_np)
        # label = np.copy(char_np)
        # assuming we have user input
        width = np.expand_dims(np.array([(one_sample['width'])] * len(char_np)) / 1000, 1)
        # print("width is ", width)
        height = np.expand_dims(np.array([(one_sample['height'])] * len(char_np)) / 1000, 1)
        # print("height is ", height)
        char_idx_list = torch.tensor(np.concatenate((input_char_np, norm_x, norm_y, width, height), axis=1))
        label = torch.tensor(np.copy(char_np))

        return char_idx_list, label

    def __getitem__(self, idx):
        data_arr, label = self.process_csv(self.df, idx)
        return data_arr, label

    def __len__(self):
        return self.data_length


class MaskedLM(IterableDataset):
    """Data loader for the long-term decoder.
    DataLongTerm parses the raw data and extracts
    (seq. of user input, g.t. seq. of characters) pairs of each full sentence.

    """
    def __init__(self, file_path, min_length=13, full_sentence=False, inference=False,
                 max_length=512):
        self.file_path = file_path
        self.min_length = min_length
        self.inference = inference
        self.max_length = max_length

    # TODO: I need to figure out how to use dataloaders more effectively, to avoid loading the entire dataset into memory, although maybe that is possible
    # TODO: I should also figure out whether to store the tensorised or untensorised data
    @staticmethod
    def process(read_path, write_path, min_length=13, full_sentence=False, inference=False):
        with open(read_path, 'r', encoding='utf-8') as input_file, open(write_path, 'a', newline='') as output_file:
            writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
            for line in input_file:
                if len(line) < min_length:
                    continue

                if full_sentence:
                    cropped_chars = line
                else:
                    selected_length = random.randint(min_length, len(line))
                    cropped_chars = line[:selected_length]
                masked_sentence = []
                label = []
                mask = []
                full_label = []
                for char in cropped_chars:
                    if char not in chars:
                        continue
                    else:
                        prob = random.random()
                        if prob < 0.85:
                            masked_sentence += [char_to_idx[char]]
                            if inference:
                                label += [char_to_idx[char]]
                                mask += [1]
                            else:
                                label += [0]
                                mask += [0]
                        elif prob < 0.97:
                            masked_sentence += [char_to_idx['[MASK]']]
                            label += [char_to_idx[char]]
                            mask += [1]
                        elif prob < 0.985:
                            masked_sentence += [randint(0, len(chars) - 1)]
                            label += [char_to_idx[char]]
                            mask += [1]
                        else:
                            masked_sentence += [char_to_idx[char]]
                            label += [char_to_idx[char]]
                            mask += [1]

                        full_label += [char_to_idx[char]]

                # masked_char = torch.tensor(masked_sentence)
                # label = torch.tensor(label)
                # mask = torch.tensor(mask)
                # full_label = torch.tensor(full_label)

                writer.writerow([masked_sentence] + [label] + [mask] + [full_label])
                # output_file.write(f'"{masked_sentence}", "{label}", "{mask}", "{full_label}"\n')


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        worker_count = worker_info.num_workers if worker_info is not None else 1

        with open(self.file_path, newline='', encoding='utf-8') as input_file:
            reader = csv.DictReader(input_file)
            for row_number, row in enumerate(reader):
                if row_number % worker_count != worker_id:
                    continue

                masked_char = ast.literal_eval(row['masked_sentence'])
                label = ast.literal_eval(row['label'])
                mask = ast.literal_eval(row['mask'])
                full_label = ast.literal_eval(row['full_label'])
                if not masked_char:
                    continue

                for start in range(0, len(masked_char), self.max_length):
                    end = start + self.max_length
                    yield (
                        torch.tensor(masked_char[start:end], dtype=torch.long),
                        torch.tensor(label[start:end], dtype=torch.long),
                        torch.tensor(mask[start:end], dtype=torch.long),
                        torch.tensor(full_label[start:end], dtype=torch.long),
                    )


def pad_variable(batch):
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)             # sort the batch in descending order
    sequences = [x[0] for x in sorted_batch]                                            # length of sequence
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)     # pad the sequence
    lengths = [len(x) for x in sequences]
    labels = [x[1] for x in sorted_batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    if len(sorted_batch[0]) > 2:
        masks = [x[2] for x in sorted_batch]
        masks_padded = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)

        full_labels = [x[3] for x in sorted_batch]
        full_labels_padded = torch.nn.utils.rnn.pad_sequence(full_labels, batch_first=True)

    else:
        masks_padded = lengths
        full_labels_padded = labels_padded

    return sequences_padded, labels_padded.long(), lengths, masks_padded, full_labels_padded.long()

def get_dataloader(data_path, batch_size, test=False, masked_LM=False):
    if masked_LM:
        dataset = MaskedLM(data_path, full_sentence=False, min_length=9, inference=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_variable)
        return data_loader
    else:
        dataset = DataVariableLength(data_path, full_sentence=False, min_length=9, augment=False)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=not test, collate_fn=pad_variable)
        return data_loader