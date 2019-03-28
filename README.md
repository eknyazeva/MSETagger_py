# MSETagger
POS tagging for low-resource languages, using specialised MorphoSyntactic Embeddings.

# Dependencies

The tagger is built on top of [yaset](https://github.com/jtourille/yaset) for the Bi-LSTM tagger part and [mimick](https://github.com/yuvalpinter/Mimick) to compute embeddings of out-of-vocabulary words.

# Requirements

You need a python3 environnement (you will probably want to create a dedicated virtual environnement).

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

# Toy example

Please clone the repository and go inside its directory.

You can execute
```bash
python main.py --mode iterative_train_and_test --work_dir toy-example --yaset_patience 3
```
to try MSETagger on a small amount of data in order to check if all requirements are satisfied. Note: in this example, the yaset_patience parameter is equals to 3 to go faster. For real data, this parameter shall not be so small, values such as 75 or 100 are more typical (for more details about the different parameters see the Options section... which is not yet written).

# End-to-end MSETagger

End-to-end use of MSETagger includes 3 main steps:

* Train morphosyntactic embeddings
* Train Bi-LSTM tagger using the previous embeddings
* Apply trained tagger to test data

For that, we need 4 data files:

* A raw train corpus file in the form of preprocessed tokenised text
* An annotated train corpus in [conllu](https://universaldependencies.org/format.html) format (it is ok if not all columns have significant values: only word and tag columns will be used (the 2nd and the 4th columns respectively).)
* An annotated dev corpus in conllu format
* A test corpus in conllu format (no need for it to be annotated, it can have a "-" in the tag column)

You need to put these files (or links to them) in a working directory with default names train.raw.txt, train.conllu, dev.conllu, gold.conllu. Now, in order to run default end-to-end configuration, just do
```bash
python main.py --mode iterative_train_and_test --work_dir path/to/the/working/directory
```
replacing path/to/the/working/directory properly.

Tagged output will be placed in the working directory with the name test-tagged.conllu. The working directory will also contain the embeddings and the tagger model produced at each iteration. These are placed in a models subdirectory if your working directory. You can change the name if this directory with --models_dir parameter. You can also choose not to keep all these model files by passing --keep_models False.

# Training with prepared embeddings

You can use MSETagger with embeddings from another system, such as [Fasttext](https://fasttext.cc). In this case, the MSETagger will

* Train a Bi-LSTM tagger using the provided embeddings
* Apply trained tagger to test data

As embeddings will not be trained, you don't need any raw data, but you need an embeddings file instead. So, the required data are
* An embeddings file in [Word2Vec](https://github.com/dav/word2vec) text format (The first line contains the number of words then a space then the dimentionality of embedding vectors. The other lines contain the word followed by it's representation, all fileds are separated by spaces.)
* An annotated train corpus in conllu format
* An annotated dev corpus in conllu format
* A test corpus in conllu format

You need to put the data files (or links to them) in a working directory with default names train.conllu, dev.conllu, gold.conllu. The embeddings file must also be put in a working directory, with an arbitrary name which will be given as a parameter during execution (for the case you have several embeddings to try on this data). Now, in order to run training with provided embeddings, do
```bash
python main.py --mode train_and_test_with_embeddings --work_dir path/to/the/working/directory --embeddings the/name/of/the/embedding/file
```
replacing path/to/the/working/directory properly. You can also give a special name to your experience with the parameter --models_dir.

Tagged output will be placed in the working directory with the name test-tagged.conllu.

This is an exemple of using the provided embeddings on toy data:
```bash
python main.py --mode train_and_test_with_embeddings --work_dir toy-example --embeddings embeddings_example.vec --yaset_patience 3 --models_dir exp_with_provided_embeddings
```
