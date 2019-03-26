import subprocess
from codecs import open

from Corpora import RawCorpora

MORPH = "\ue00b"

class Morfessor:

    def __init__(self, working_dir):
        self.working_dir = working_dir + "/"
        # subprocess.call(["mkdir", self.working_dir])

    def make_morpho(self, raw_data_file):

        raw_corpora = RawCorpora(raw_data_file)
        vocab_file = self.working_dir + "lex.txt"
        raw_corpora.extract_vocabulary_to_file(vocab_file)

        temp_file = self.working_dir + "morpho.txt"
        subprocess.call(["morfessor", \
                         "-t", raw_data_file, \
                         "-d", "ones", \
                         "-T", vocab_file, \
                         "--output-format", "{compound}\\t{analysis}\\n"], \
                        stdout=open(temp_file, 'wt'))

        morpho_content = open(temp_file, 'rt').readlines()
        f = open(self.working_dir + "morpho.txt", 'wt')
        for line in morpho_content:

            w, morph = line.strip().split('\t')
            morph_set = morph.split(' ')
            if len(morph_set) > 1:
                morph_set[0] = '%%' + morph_set[0]
                morph_set[-1] = morph_set[-1] + '%%'
                morph_new = MORPH.join(morph_set)
                f.write(w + '\t' + morph_new + '\n')
            if len(morph_set) > 1:
                f.write(w + '\t\n')
        
        return self.working_dir + "morpho.txt", vocab_file
