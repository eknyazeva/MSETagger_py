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

# Toy example

You can execute
```bash
python main.py --mode iterative_train_and_test --work_dir toy-example --yaset_patience 3
```
to try MSETagger on a small amount of data in order to check if all requirements are satisfied.

# End-to-end MSETagger

End-to-end use of MSETagger includes 3 main steps:

* train morphosyntactic embeddings
* train Bi-LSTM tagger using the previous embeddings
* apply trained tagger to test data

For that, we need 4 data files:

* A raw train corpus file in the form of preprocessed tokenised text
* An annotated train corpus in [conllu](https://universaldependencies.org/format.html) format (it is ok if not all columns have significant values: only word and tag columns will be used (the 2nd and the 4th columns respectively).)
* An annotated dev corpus in conllu format
* A test corpus in conllu format (no need for it to be annotated, it can have a "-" in the tag column)

You need to put these files (or links to them) in a working directory with default names train.raw.txt, train.conllu, dev.conllu, test.conllu. Now, in order to run default end-to-end configuration, just do 
```bash
python main.py --mode iterative_train_and_test --work_dir path/to/the/working/directory
```
replacing path/to/the/working/directory properly.

Tagged output will be placed in the working directory with the name test-tagged.conllu. The working directory will also contain the embeddings and the tagger model produced at each iteration. These are placed in a models subdirectory if your working directory. You can change the name if this directory with --models_dir parameter. You can also choose not to keep all these model files by passing --keep_models False.

# Training with prepared embeddings

You can you MSETagger with embeddings issu from another system, such as [Fasttext](https://fasttext.cc). In this case, the MSETagger will

* train Bi-LSTM tagger using the provided embeddings
* apply trained tagger to test data

As embeddings will not be trained, you don't need any raw data, but you need an embeddings file instead. So, the required data are
* An embeddings file
* An annotated train corpus in conllu format
* An annotated dev corpus in conllu format
* A test corpus in conllu format

You need to put these files (or links to them) in a working directory with default names embeddings.vec, train.conllu, dev.conllu, test.conllu. Now, in order to run training with provided embeddings, do
```bash
python main.py --mode train_and_test_with_embeddings --work_dir path/to/the/working/directory
```
replacing path/to/the/working/directory properly.

Tagged output will be placed in the working directory with the name test-tagged.conllu.