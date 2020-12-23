from torch.utils.data import Dataset
import linecache
import os
import numpy as np

def parse_a_line(line):
    words = line.strip().split("\t")

    sampleid = words[0]
    drugid = words[1]
    atcid = words[2]

    features1 = np.array([float(x) for x in words[3].split(",")]).reshape(1, -1)
    features2 = np.array([float(x) for x in words[4].split(",")]).reshape(1, -1)
    features3 = np.array([float(x) for x in words[5].split(",")]).reshape(1, -1)
    features4 = np.array([float(x) for x in words[6].split(",")]).reshape(1, -1)
    features5 = np.array([float(x) for x in words[7].split(",")]).reshape(1, -1)
    features6 = np.array([float(x) for x in words[8].split(",")]).reshape(1, -1)
    features7 = np.array([float(x) for x in words[9].split(",")]).reshape(1, -1)
    features = np.concatenate((features1, features2, features3, features4, features5, features6, features7), axis=0).reshape(1, 7, -1)

    label = int(words[9])

    return sampleid, drugid, atcid, features, label


class DataLoader(Dataset):
    def __init__(self, filename, transform=None):
        self._filename = os.path.abspath(filename)
        self._total_data = 0
        self._transform = transform
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())

    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        if line == "":
            # return None
            raise RuntimeError("empty batch")
        else:
            output = parse_a_line(line)
            if self._transform is not None:
                output = self._transform(output)
            return output

    def __len__(self):
        return self._total_data

    def clear_cache(self):
        linecache.clearcache()
