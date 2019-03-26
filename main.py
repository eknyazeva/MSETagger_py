import sys
from codecs import open
import subprocess
import argparse

from Corpora import RawCorpora, AnnotatedCorpora
from Morfessor import Morfessor
from MSE import MSE
from Yaset import Yaset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--raw_data', default=None)
    parser.add_argument('--train_data', default=None)
    parser.add_argument('--dev_data', default=None)
    parser.add_argument('--gold_data', default=None)
    parser.add_argument('--yaset_model', default="yaset_model_final")
    parser.add_argument('--embeddings', default=None)
    parser.add_argument('--output_model', default="yaset_model_final")
    parser.add_argument('--output_embeddings', default="embeddings.vec")
    parser.add_argument('--tagger_output', default="test.conllu")
    parser.add_argument('--use_mimick', default='True')
    parser.add_argument('--iter_number', default=1)
    parser.add_argument('--temp_files_dir', default=None)


    args = parser.parse_args()

    working_dir = args.work_dir

    if args.mode == "train_yaset":

        train_data_file = working_dir + "/train.conllu" if not args.train_data else args.train_data
        dev_data_file = working_dir + "/dev.conllu" if not args.dev_data else args.dev_data

        work_space = "/_train_yaset_workspace" if not args.temp_files_dir else '/' + args.temp_files_dir
        subprocess.call(["mkdir", working_dir + work_space])

        yaset = Yaset(working_dir + work_space)

        train_data_yaset_file = working_dir + work_space + "/train.conll"
        AnnotatedCorpora(train_data_file).write_yaset_format(train_data_yaset_file)

        dev_data_yaset_file = working_dir + work_space + "/dev.conll"
        AnnotatedCorpora(dev_data_file).write_yaset_format(dev_data_yaset_file)
        
        embeddings_file = working_dir + "/embeddings.vec" if not args.embeddings else args.embeddings
        model_path = yaset.train_model(embeddings_file, train_data_yaset_file, dev_data_yaset_file)

        print(model_path)
        subprocess.call(["cp", '-r', \
                         model_path, \
                         working_dir + '/' + args.output_model])

    if args.mode == "apply_yaset":

        work_space = "/_apply_workspace" if not args.temp_files_dir else '/' + args.temp_files_dir
        subprocess.call(["mkdir", working_dir + work_space])
        yaset = Yaset(working_dir + work_space)

        test_data_file = working_dir + "/gold.conllu" if not args.gold_data else args.gold_data
        test_data_yaset_file = working_dir + work_space + "/gold.conll"
        AnnotatedCorpora(test_data_file).write_yaset_format(test_data_yaset_file)

        yaset_model_path = working_dir + '/' + args.yaset_model 
        output_file = yaset.apply_model(test_data_yaset_file, yaset_model_path)

        AnnotatedCorpora(output_file).write_conllu(working_dir + '/' + args.tagger_output)

    if args.mode == "train_embeddings":

        raw_data_file = working_dir + "/train.raw.txt" if not args.raw_data else args.raw_data
        train_data_file = working_dir + "/train.conllu" if not args.train_data else args.train_data
        dev_data_file = working_dir + "/dev.conllu" if not args.dev_data else args.dev_data
        test_data_file = working_dir + "/gold.conllu" if not args.gold_data else args.gold_data

        work_space = "/_train_embeddings_workspace" if not args.temp_files_dir else '/' + args.temp_files_dir
        subprocess.call(["mkdir", working_dir + work_space])

        morfessor = Morfessor(working_dir + work_space)
        mse = MSE(working_dir + work_space)

        raw_data_txt_file = working_dir + work_space + "/train.raw.txt"
        raw_data_conllu_file = working_dir + work_space + "/train.raw.conllu"
        AnnotatedCorpora(raw_data_file).write_raw(raw_data_txt_file)
        AnnotatedCorpora(raw_data_file).write_conllu(raw_data_conllu_file)

        morpho_file, train_vocab_file = morfessor.make_morpho(raw_data_txt_file)

        if args.use_mimick == 'True':
            full_vocab_file = mse.create_vocab([train_data_file, \
                                                dev_data_file, \
                                                test_data_file, \
                                                raw_data_conllu_file], \
                                               train_vocab_file)

        if args.use_mimick == 'True':
            embeddings_file = mse.learn_embeddings(raw_data_conllu_file, \
                                                   morpho_file, \
                                                   vocab_file=full_vocab_file, \
                                                   use_mimick=True)
        else:
            embeddings_file = mse.learn_embeddings(raw_data_conllu_file, \
                                                   morpho_file, \
                                                   vocab_file=None, \
                                                   use_mimick=False)
        subprocess.call(["cp", '-r', \
                         embeddings_file, \
                         working_dir + '/' + args.output_embeddings])

    if args.mode.startswith("iterative_train"):

        raw_data_conllu_file = working_dir + "/train.raw.txt" if not args.raw_data else args.raw_data
        train_data_file = working_dir + "/train.conllu" if not args.train_data else args.train_data
        dev_data_file = working_dir + "/dev.conllu" if not args.dev_data else args.dev_data
        test_data_file = working_dir + "/gold.conllu" if not args.gold_data else args.gold_data

        work_space = "/_iterative_train_workspace" if not args.temp_files_dir else '/' + args.temp_files_dir
        subprocess.call(["mkdir", working_dir + work_space])

        morfessor = Morfessor(working_dir + work_space)
        mse = MSE(working_dir + work_space, iterative=True)
        yaset = Yaset(working_dir + work_space, iterative=True)

        raw_data_file = working_dir + work_space + "/train.raw.txt"
        AnnotatedCorpora(raw_data_conllu_file).write_raw(raw_data_file)
        raw_data_yaset_file = working_dir + work_space + "/train.raw.conll"
        AnnotatedCorpora(raw_data_file).write_yaset_format(raw_data_yaset_file)

        train_data_yaset_file = working_dir + work_space + "/train.conll"
        AnnotatedCorpora(train_data_file).write_yaset_format(train_data_yaset_file)

        dev_data_yaset_file = working_dir + work_space + "/dev.conll"
        AnnotatedCorpora(dev_data_file).write_yaset_format(dev_data_yaset_file)

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
                                                       vocab_file=full_vocab_file, \
                                                       use_mimick=True)
            else:
                embeddings_file = mse.learn_embeddings(raw_data_conllu_file, \
                                                       morpho_file, \
                                                       vocab_file=None, \
                                                       use_mimick=False)

            model_path = yaset.train_model(embeddings_file, train_data_yaset_file, dev_data_yaset_file)
            
            retagged_raw = yaset.apply_model(raw_data_yaset_file, model_path)
            
            retagged_raw_conllu_file = working_dir + work_space + "/iter" + str(i) + "/train.raw.retagged.conllu"
            AnnotatedCorpora(retagged_raw).write_conllu(retagged_raw_conllu_file)
            raw_data_conllu_file = retagged_raw_conllu_file

        if args.mode == "iterative_train":

            subprocess.call(["cp", '-r', \
                             model_path, \
                             working_dir + '/' + args.output_model])

        if args.mode == "iterative_train_and_test":

            test_data_file = working_dir + "/gold.conllu" if not args.gold_data else args.gold_data
            test_data_yaset_file = working_dir + work_space + "/gold.conll"
            AnnotatedCorpora(test_data_file).write_yaset_format(test_data_yaset_file)

            work_space = "/_iterative_train_workspace" if not args.temp_files_dir else '/' + args.temp_files_dir
            yaset = Yaset(working_dir + work_space)
            output_file = yaset.apply_model(test_data_yaset_file, model_path)
            
            AnnotatedCorpora(output_file).write_conllu(working_dir + '/' + args.tagger_output)
