from keras.preprocessing.sequence import pad_sequences
class prepareOneHot:
    def __init__(self, label_text):
        '''
        key == y
        label_text[key] == X
        text -> label
        :param txt_label:
        '''
        self.y = []
        self.X_temp = []
        for k in label_text.keys():
            for d in label_text[k]:
                self.X_temp.append(d.strip())
                self.y.append(k)

        self.create_dict()
        self.create_X()

    def create_dict(self):
        # dict = open(txt_path + '1_point.txt', 'rb').read()
        # dict += open(txt_path + '2_point.txt', 'rb').read()
        # dict += open(txt_path + '5_point.txt', 'rb').read()
        X_dict = ''.join(self.X_temp)
        dict_list = set(list(X_dict.decode('utf8')))
        self.dicts = {}
        for i, d in enumerate(dict_list):
            self.dicts[d] = i
        # return dicts

    def create_X(self):
        self.len_seq = []
        self.X_sequences = []
        for line in self.X_temp:
            if line == '\n':
                continue
            line = line.strip()
            l = list(line.decode('utf8'))
            sequence = [self.dicts[char] for char in l]
            self.len_seq.append(len(sequence))
            self.X_sequences.append(sequence)

    def get_X_y(self):
        return self.X_sequences, self.y, self.len_seq,self.dicts

    def get_pad_X_y(self, max_seq):
        X_sequences = pad_sequences(self.X_sequences, maxlen=max_seq)

        return X_sequences, self.y, self.len_seq, self.dicts



def prepare_X_y():
    y = []

    txt_path = '../data/corpus/reviews/'

    with open(txt_path+'1_point.txt', 'rb') as fp:
        lines = fp.readlines()
    len_temp_lines = len(lines)
    for i in range(len(lines)):
        y.append(1)

    with open(txt_path+'5_point.txt', 'rb') as fp:
        lines += fp.readlines()
    for i in range(len(lines[len_temp_lines:])):
        y.append(0)

    def create_dict():
        dict = open(txt_path+'1_point.txt', 'rb').read()
        # dict += open(txt_path + '2_point.txt', 'rb').read()
        dict += open(txt_path+'5_point.txt', 'rb').read()
        dict_list = set(list(dict.decode('utf8')))
        dicts = {}
        for i, d in enumerate(dict_list):
            dicts[d] = i
        return dicts

    def create_X(lines):
        len_seq = []
        dicts = create_dict()
        sequences = []
        for line in lines:
            if line == '\n':
                continue
            line = line.strip()
            l = list(line.decode('utf8'))
            sequence = [dicts[char] for char in l]
            len_seq.append(len(sequence))
            sequences.append(sequence)
        return sequences, len_seq, dicts


    X_sequences, len_seq, dicts = create_X(lines)

    max_seq = max(len_seq)/2
    print 'max_seq:', max_seq


    max_features = len(dicts) + 1
    print 'max_features:', max_features


    data_X = pad_sequences(X_sequences, maxlen=max_seq)

    return data_X, y, dicts, max_seq, max_features

if __name__== '__main__':
    # txt_path = '../data/corpus/reviews/'
    txt_path = './'

    label_text = {}
    with open(txt_path+'1_point.txt', 'rb') as fp:
        lines = fp.readlines()

    label_text[0] = lines

    with open(txt_path+'5_point.txt', 'rb') as fp:
        lines += fp.readlines()

    label_text[1] = lines

    oh = prepareOneHot(label_text)

