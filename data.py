import os
from torch.utils.data import Dataset

class AnnotatedVoiceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.names = []
        self.labels = []

        self.gather_labels()

    def gather_labels(self):
        names = []
        labels = []
        for root, dirs, files in os.walk(self.root_dir, topdown=True):
            for name in files:
                if name.endswith('.txt'):
                    with open(os.path.join(root, name)) as file:
                        string = ''
                        in_name = True

                        while True:
                            c = file.read(1)
                            if not c:
                                string = string.strip()
                                labels.append(string)
                                #print("label: " + string)
                                #print("End of file")
                                break

                            if in_name:
                                if c.isspace():
                                    in_name = False
                                    string = string.strip()
                                    names.append(os.path.join(root, string) + ".flac")
                                    #print("name: " + os.path.join(root, string) + ".flac")
                                    string = ''
                            else:
                                if c.isdigit():
                                    in_name = True
                                    string = string.strip()
                                    labels.append(string)
                                    #print("label: " + string)
                                    string = ''

                            string += c

        self.names = names
        self.labels = labels

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if 0 <= idx < self.__len__():
            return self.names[idx], self.labels[idx]

