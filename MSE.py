import sys
import subprocess
from codecs import open

from Corpora import AnnotatedCorpora

class MSE:

    def __init__(self, working_dir, iterative=False):
        
        self.working_dir = working_dir + "/"
        # subprocess.call(["mkdir", self.working_dir])
        self.iterative = iterative
        self.n_iter = 0

    def create_vocab(self, data_files, vocab_init):
        
        full_vocab_file = self.working_dir + "full_lex.txt"
        vocab = set([])
        
        fin = open(vocab_init, 'rt')
        for line in fin.readlines():
            vocab.add(line.strip())

        vocab.add('$$UNK$$')

        for corpus in data_files:
            corpus_file = open(corpus, 'rt')
            for line in corpus_file.readlines():
                if line.strip() != "":
                    word = line.strip().split('\t')[0]
                    vocab.add(word)

        fout = open(full_vocab_file, 'wt')
        for word in vocab:
            fout.write(word + '\n')
        fout.close()

        return full_vocab_file

    def learn_embeddings(self, raw_data_conll_file, morpho_file, noccmin=2, ndim=50, use_mimick=False, vocab_file=None):

        if use_mimick and not vocab_file:
            print("need vocab file")
            sys.exit(1)

        if self.iterative:
            self.n_iter += 1
            self.working_dir_iter = self.working_dir + "/iter" + str(self.n_iter) + "/"
            subprocess.call(["mkdir", self.working_dir_iter])
        else:
            self.working_dir_iter = self.working_dir

        print("raw data")
        print(raw_data_conll_file)
        subprocess.call(["python", \
                         "mse/train_keras.py", \
                         raw_data_conll_file, \
                         morpho_file, \
                         str(noccmin), \
                         str(ndim), \
                         self.working_dir_iter])

        embeddings_file = self.working_dir_iter + "emb_train_vocab.vec"
        dataset_file = self.working_dir_iter + "mimick_model"
        output_file = self.working_dir_iter + "emb_full_vocab.vec"

        if use_mimick:

            subprocess.call(["python", \
                             "mimick/make_dataset.py", \
                             "--vectors", embeddings_file, \
                             "--output", dataset_file, \
                             "--vocab", vocab_file, \
                             "--w2v-format"])

            subprocess.call(["python", \
                             "mimick/model.py", \
                             "--dataset", dataset_file, \
                             "--output", output_file, \
                             "--vocab", vocab_file, \
                             "--dynet-mem", "4096"])

            return output_file

        else:

            return embeddings_file
