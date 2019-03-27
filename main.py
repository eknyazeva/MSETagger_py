import sys
from codecs import open
import subprocess
import argparse
import datetime

from Corpora import RawCorpora, AnnotatedCorpora
from Morfessor import Morfessor
from MSE import MSE
from Yaset import Yaset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--embeddings', default=None)
    parser.add_argument('--tagger_output', default="test.tagged.conllu")
    parser.add_argument('--use_mimick', default='True')
    parser.add_argument('--iter_number', default=1)
    parser.add_argument('--temp_files_dir', default=None)
    parser.add_argument('--emb_dim', default=50)
    parser.add_argument('--emb_min_occ_num', default=1)
    parser.add_argument('--yaset_patience', default=100)


    args = parser.parse_args()

    working_dir = args.work_dir
    work_space = '/' + args.temp_files_dir + '/' if args.temp_files_dir else "/workspace/"
    subprocess.call(["mkdir", working_dir + work_space])

    raw_data_file = working_dir + "/train.raw.txt"
    train_data_file = working_dir + "/train.conllu"
    dev_data_file = working_dir + "/dev.conllu"
    test_data_file = working_dir + "/gold.conllu"

    raw_data_txt_file = working_dir + work_space + "/train.raw.txt"
    AnnotatedCorpora(raw_data_file).write_raw(raw_data_txt_file)
    raw_data_conllu_file = working_dir + work_space + "/train.raw.conllu"
    AnnotatedCorpora(raw_data_file).write_conllu(raw_data_conllu_file)

    # yaset needs a particular conll-type format
    train_data_yaset_file = working_dir + work_space + "/train.conll"
    AnnotatedCorpora(train_data_file).write_yaset_format(train_data_yaset_file)
    dev_data_yaset_file = working_dir + work_space + "/dev.conll"
    AnnotatedCorpora(dev_data_file).write_yaset_format(dev_data_yaset_file)        
    test_data_yaset_file = working_dir + work_space + "/gold.conll"
    AnnotatedCorpora(test_data_file).write_yaset_format(test_data_yaset_file)
    raw_data_yaset_file = working_dir + work_space + "/train.raw.conll"
    AnnotatedCorpora(raw_data_file).write_yaset_format(raw_data_yaset_file)
  
    if args.mode == "train_and_test_with_embeddings":
        
        assert args.embeddings, "You need to specify the embeddings file to use this mode"
        embeddings_file = working_dir + args.embeddings

        yaset = Yaset(working_dir + work_space)
        model_path = yaset.train_model(embeddings_file, train_data_yaset_file, dev_data_yaset_file, \
                                       yaset_patience=args.yaset_patience)
        output_file = yaset.apply_model(test_data_yaset_file, model_path)

        AnnotatedCorpora(output_file).write_conllu(working_dir + '/' + args.tagger_output)

    if args.mode == "iterative_train_and_test":

        morfessor = Morfessor(working_dir + work_space)
        mse = MSE(working_dir + work_space, iterative=True)
        yaset = Yaset(working_dir + work_space, iterative=True)

        morpho_file, train_vocab_file = morfessor.make_morpho(raw_data_file)

        if args.use_mimick == 'True':
            full_vocab_file = mse.create_vocab([train_data_file, \
                                                dev_data_file, \
                                                test_data_file, \
                                                raw_data_conllu_file], \
                                               train_vocab_file)

        for i in range(1, int(args.iter_number) + 1):

            if args.use_mimick == 'True':
                embeddings_file = mse.learn_embeddings(raw_data_conllu_file, \
                                                       morpho_file, \
                                                       noccmin=int(args.emb_min_occ_num), \
                                                       ndim=int(args.emb_dim), \
                                                       vocab_file=full_vocab_file, \
                                                       use_mimick=True)
            else:
                embeddings_file = mse.learn_embeddings(raw_data_conllu_file, \
                                                       morpho_file, \
                                                       noccmin=int(args.emb_min_occ_num), \
                                                       ndim=int(args.emb_dim), \
                                                       vocab_file=None, \
                                                       use_mimick=False)

            model_path = yaset.train_model(embeddings_file, train_data_yaset_file, dev_data_yaset_file, \
                                           yaset_patience=args.yaset_patience)
            
            retagged_raw = yaset.apply_model(raw_data_yaset_file, model_path)
            
            retagged_raw_conllu_file = working_dir + work_space + "/iter" + str(i) + "/train.raw.retagged.conllu"
            AnnotatedCorpora(retagged_raw).write_conllu(retagged_raw_conllu_file)
            raw_data_conllu_file = retagged_raw_conllu_file

            output_file = yaset.apply_model(test_data_yaset_file, model_path)            
            AnnotatedCorpora(output_file).write_conllu(working_dir + work_space + '/iter' + str(i) + '/' + args.tagger_output)

        output_file = yaset.apply_model(test_data_yaset_file, model_path)
        AnnotatedCorpora(output_file).write_conllu(working_dir + args.tagger_output)

    subprocess.call(["rm", train_data_yaset_file, dev_data_yaset_file, test_data_yaset_file, raw_data_yaset_file])
    subprocess.call(["rm", raw_data_txt_file])
    subprocess.call(["rm", raw_data_conllu_file])
