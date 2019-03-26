from codecs import open

class RawCorpora:

    def __init__(self, corpora_file):

        f = open(corpora_file, 'rt')
        self.sentences = f.readlines()

    def extract_vocabulary(self):

        vocab = set([])
        for sentence in self.sentences:
            for word in sentence.strip().split():
                vocab.add(word)

        return vocab

    def extract_vocabulary_to_file(self, output_file):
        
        vocab = self.extract_vocabulary()
        f = open(output_file, 'wt')
        for word in vocab:
            f.write(word + '\n')
        f.close()


class AnnotatedCorpora:

    def __init__(self, corpora_file):

        file_content = open(corpora_file, 'rt').readlines()
        self.sentences = []
        
        if corpora_file.endswith("conllu"):

            sentence = []
            for line in file_content:
                if line.strip() == '':
                    self.sentences.append(sentence)
                    sentence = []
                else:
                    fields = line.strip().split('\t')
                    word = fields[1]
                    tag = fields[3]
                    sentence.append((word, tag))
        else:

            if corpora_file.endswith("conll"):

                file_content = open(corpora_file, 'rt').readlines()
                self.sentences = []

                sentence = []
                for line in file_content:
                    if line.strip() == '':
                        self.sentences.append(sentence)
                        sentence = []
                    else:
                        fields = line.strip().split('\t')
                        word = fields[0]
                        tag = fields[-1]
                        sentence.append((word, tag))

            else:

                for line in file_content:
                    sentence = []
                    for word in line.strip().split():
                        sentence.append((word, "--xxx--"))
                    self.sentences.append(sentence)


    def write_conllu(self, output_file):
        
        f = open(output_file, 'wt')

        for sentence in self.sentences:
            i = 1
            for (word, tag) in sentence:
                f.write('\t'.join([str(i), word, '-', tag, '-', '-', '-', '-', '-', '-']) + '\n')
                i += 1
            f.write('\n')

    def write_raw(self, output_file):
        
        f = open(output_file, 'wt')

        for sentence in self.sentences:
            for word, tag in sentence:
                f.write(word + ' ')
            f.write('\n')

    def write_yaset_format(self, output_file):

        f = open(output_file, 'wt')

        for sentence in self.sentences:
            i = 1
            for (word, tag) in sentence:
                f.write('\t'.join([word, tag]) + '\n')
                i += 1
            f.write('\n')

