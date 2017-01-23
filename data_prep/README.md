## Creating the Dependency Parsing Dataset
Please follow these steps to recreate the dependency parsing dataset:

- Obtain Stanford dependency trees by running the Stanford Dependency Converter, e.g.,

```java -cp stanford-parser-3.3.0.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile  ~/Projects/group/data/wsj/train.txt     -conllx -basic > train.conll.txt```

- Use dep/convert.py to produce target sequences from the conll files by providing the desired conll file as the first argument, as follows:

```python convert.py train.conll.txt```

This will write a file to /tmp/targetparses.txt.

- Use dep/convert.py to replace digits with '#' and remove final ROOT symbol and action, as follows:

```python preproc.py /tmp/targetparses.txt ```

This will create files /tmp/src-targetparses.txt and /tmp/targ-targetparses.txt .

- If interested, you can create hdf5 tensors (which bso_train.lua consumes) in the following way:

```python preproc_pp.py --srcfile src-targetparses.txt --targetfile targ-targetparses.txt --srcvalfile src-targetparses_val.txt --targetvalfile targ-targetparses_val.txt --outputfile dep/dep --seqlength 283 --srcminfreq 2 ```

This will replace any tokens occuring fewer than twice with an UNK token. The "--seqlength 283" argument just happens to be the length of the longest target sequence; no sentences should be discarded.

- To obtain CONLL format files from predicted target sequences, use dep/convert_back_noroot.py, e.g.,

```python convert_back_noroot.py < dev-preds.out > dev-preds.out.conll```


## Creating the Word-ordering Dataset
Use Allen Schmaltz's data generation [script](https://github.com/allenschmaltz/word_ordering/tree/master/data/preprocessing) (with instructions [here](https://github.com/allenschmaltz/word_ordering/tree/master/data/preprocessing)). The only files needed are ```${splitname}_words_fullyshuffled_no_eos.txt``` (which is the source), and ```${splitname}_words_no_eos.txt``` (which is the target), and so the script may be edited to produce only these. The script creates files with unk's already substituted and so preproc.py should not do any unk replacement. See instructions [here](https://github.com/allenschmaltz/word_ordering/blob/master/Usage.txt) for randomly replacing unks and evaluating at test time.


## Creating the MT Dataset
To creat the MIXER dataset, I used the data-preparation code at https://github.com/facebookresearch/MIXER, but modified it to output text files (which the BSO code consumes), rather than torch Tensors. The modified files (prepareData.sh, makedata.lua, and tokenizer.lua) are in the MT/ directory, and you should be able to just run:

```bash prepareData.sh```

This will create a directory MT/prep/, which will contain the original train, val, and test files, as well as files suffixed with '.wmixerprep'. These \*.wmixerprep files put unks in the same place as MIXER does internally, and these should be used for training. Final results should use the non-*.wmixerprep files.

As above, preproc_pp.py can be used to form hdf5 tensors from the text data. No words should be replaced with UNK (since the preprocessing already did this), and no sentences should be discarded.

## Notes
preproc_pp.py is a very slight modification of [Yoon Kim's preprocessing script](https://github.com/harvardnlp/seq2seq-attn/blob/master/preprocess.py) .
