Python re-implementation of the system described in [this paper](https://hal.archives-ouvertes.fr/LIMSI/hal-01793092v1) and implemented in [this repository](https://github.com/a-tsioh/MSETagger).

# MSETagger
POS tagging for low-resource languages, using specialized MorphoSyntactic Embeddings.

# Dependencies

The tagger is built on top of [yaset](https://github.com/jtourille/yaset) for the Bi-LSTM tagger part and [mimick](https://github.com/yuvalpinter/Mimick) to compute embeddings of out-of-vocabulary words.

# Requirements

We need a python3 environnement (you will probably want to create a separated virtual environnement).

All the modules except yaset can be installed by executing
```bash
pip install -r requirements.txt
```

For yaset installation, please follow the [installation instructions](https://jtourille.github.io/yaset/).

# Quickstart

Please clone the repository and go inside.

The end-to-end use of MSETagger includes 3 main steps:

* train morphosyntaxique embeddings
* train the Bi-LSTM tagger using the previous embeddings
* apply the trained tagger to test data

For that, we need 4 data files:

* The raw train corpus file, as a preprocessed tokenised text
* The annotated train corpus in [conllu](https://universaldependencies.org/format.html) format (it is ok if not all the columns have significant values: only the word and the tag columns will be used (the 1st and the 4th columne respectively, starting from 0, see our toy example))
* The annotated dev corpus in conllu format
* The test corpus in conllu format (not need to be annotated, it can have "-" in the tag column)

The results will be stored in a working directory which you choose.

You need to put this files (or the link to) in a working directory with default names train.raw.txt, train.conll, dev.conll, test.conll. Now, for running the default configuration, just do 
```bash
python main.py --mode iterative_train_and_test --work_dir working/directory
```
replacing working/directory by the directory you chosed.

The tagged output will be placed to the working directory with the name test.conllu. The working directory will also contain the embeddings and the tagger model produced at each iteration. These are placed in a _iterative_train_workspace subdirectory.


