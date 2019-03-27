Python re-implementation of the system described in [this paper](https://hal.archives-ouvertes.fr/LIMSI/hal-01793092v1) and implemented in [this repository](https://github.com/a-tsioh/MSETagger).

# MSETagger
POS tagging for low-resource languages, using specialized MorphoSyntactic Embeddings.

# Dependencies

The tagger is built on top of [yaset](https://github.com/jtourille/yaset) for the Bi-LSTM tagger part and [mimick](https://github.com/yuvalpinter/Mimick) to compute embeddings of out-of-vocabulary words.

# Requirements

You need a python3 environnement (you will probably want to create a separated virtual environnement).

All the modules needed except yaset can be installed by executing
```bash
pip install -r requirements.txt
```

For yaset installation, please follow the [installation instructions](https://jtourille.github.io/yaset/).
In case of problem, do
```bash
git clone https://github.com/eknyazeva/yaset.git
cd yaset
pip install .
```

# Quickstart

Please clone the repository and go inside its directory.

End-to-end use of MSETagger includes 3 main steps:

* train morphosyntactic embeddings
* train Bi-LSTM tagger using the previous embeddings
* apply trained tagger to test data

For that, we need 4 data files:

* A raw train corpus file in the form of preprocessed tokenised text
* An annotated train corpus in [conllu](https://universaldependencies.org/format.html) format (it is ok if not all columns have significant values: only word and tag columns will be used (the 2nd and the 4th columns respectively).)
* An annotated dev corpus in conllu format
* A test corpus in conllu format (no need for it to be annotated, it can have a "-" in the tag column)

Results will be stored in a working directory which you choose.

You need to put these files (or links to them) in a working directory with default names train.raw.txt, train.conll, dev.conll, test.conll. Now, in order to run default end-to-end configuration, just do 
```bash
python main.py --mode iterative_train_and_test --work_dir path/to/the/working/directory
```
replacing path/to/the/working/directory properly.

Tagged output will be placed in the working directory with the name test-tagged.conllu. The working directory will also contain the embeddings and the tagger model produced at each iteration. These are placed in an _iterative_train_workspace subdirectory.


