import subprocess
from codecs import open
import re

class Yaset:

    def __init__(self, working_dir, iterative=False):

        self.working_dir = working_dir + "/"
        self.iterative = iterative
        self.n_iter = 0

    def train_model(self, embeddings_file, train_file, dev_file, yaset_patience='100'):

        if self.iterative:
            self.n_iter += 1
            self.working_dir_iter = self.working_dir + "/iter" + str(self.n_iter) + "/"
            subprocess.call(["mkdir", self.working_dir_iter])
        else:
            self.working_dir_iter = self.working_dir
        
        config_file = self.create_config(embeddings_file, train_file, dev_file, yaset_patience)

        subprocess.call(["yaset", \
                         "LEARN", \
                         "--config", config_file])
        
        subprocess.call(["rm", config_file])
        working_dir_content = subprocess.check_output(["ls", self.working_dir_iter]).decode("utf-8")
        model_path = working_dir_content.strip().split('\n')[-1]
        
        subprocess.call(["mv", self.working_dir_iter + model_path, self.working_dir_iter + "yaset_model"])

        return self.working_dir_iter + "yaset_model"

    def apply_model(self, data_file, model_path):

        if self.iterative:
            self.working_dir_iter = self.working_dir + "/iter" + str(self.n_iter) + "/"
            subprocess.call(["mkdir", self.working_dir_iter])
        else:
            self.working_dir_iter = self.working_dir

        subprocess.call(["yaset", \
                         "APPLY", \
                         "--input-file", data_file, \
                         "--working-dir", self.working_dir_iter,
                         "--model-path", model_path])

        working_dir_content = subprocess.check_output(["ls", self.working_dir_iter]).decode("utf-8")
        decoded_data_path = re.findall(r"yaset-apply-\d{8}-\d{6}", working_dir_content)[-1]

        subprocess.call(["cp", \
                         self.working_dir_iter + decoded_data_path + "/output-model-001.conll", \
                         self.working_dir_iter + "train.raw.retagged.conll"])

        subprocess.call(["rm", "-r", self.working_dir_iter + decoded_data_path])

        return self.working_dir_iter + "train.raw.retagged.conll"


    def create_config(self, embeddings_file, train_file, dev_file, yaset_patience):

       config_txt = " = ".join("\n".join(\
                    "[general] \
                    batch_mode = false \
                    batch_iter = 5\
                    experiment_name = name\
                    [data] \
                    train_file_path = {1} \
                    dev_file_use = true \
                    dev_file_path = {2} \
                    dev_random_ratio = 0.2 \
                    dev_random_seed_use = false \
                    dev_random_seed_value = 42 \
                    preproc_lower_input = false \
                    preproc_replace_digits = false \
                    feature_data = false \
                    feature_columns = 1,2,3 \
                    embedding_model_type = word2vec \
                    embedding_model_path = {0} \
                    embedding_oov_strategy = map \
                    embedding_oov_map_token_id = $$UNK$$ \
                    embedding_oov_replace_rate = 0.0 \
                    working_dir = {3} \
                    [training] \
                    model_type = bilstm-char-crf \
                    max_iterations = 500 \
                    patience = {4} \
                    store_matrices_on_gpu = true \
                    bucket_use = false \
                    dev_metric = accuracy \
                    trainable_word_embeddings = false \
                    cpu_cores = 12 \
                    batch_size = 16 \
                    use_last = false \
                    opt_algo = adam \
                    opt_lr = 0.001 \
                    opt_gc_use = false \
                    opt_gc_type = clip_by_norm \
                    opt_gs_val = 5.0 \
                    opt_decay_use = false \
                    opt_decay_rate = 0.9 \
                    opt_decay_iteration = 1 \
                    feature_use = false \
                    feature_embeddings_size = 10 \
                    [bilstm-char-crf] \
                    hidden_layer_size = 30 \
                    dropout_rate = 0.5 \
                    use_char_embeddings = true \
                    char_hidden_layer_size = 15 \
                    char_embedding_size = 15 \
                    ".format(embeddings_file, train_file, dev_file, \
                             self.working_dir_iter, yaset_patience).split()).split("\n=\n"))

       config_file = self.working_dir_iter + "config.ini"
       f = open(config_file, 'wt')
       f.write(config_txt)
       f.close()

       return config_file
