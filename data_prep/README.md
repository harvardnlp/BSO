## Creating the Word-ordering Dataset
I'm waiting for permission to upload files/scripts...will update soon!

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

- To obtain CONLL format files from predicted target sequences, use dep/convert_back_noroot.py, e.g.,

```python convert_back_noroot.py < dev-preds.out > dev-preds.out.conll```

## Creating the MT Dataset
To creat the MIXER dataset, I used the data-preparation code at https://github.com/facebookresearch/MIXER, but modified it to output text files (which the BSO code consumes), rather than torch Tensors. The modified files (prepareData.sh, makedata.lua, and tokenizer.lua) are in the MT/ directory, and you should be able to just run:

```bash prepareData.sh```

This will create a directory MT/prep/, which will contain the original train, val, and test files, as well as files suffixed with '.wmixerprep'. These *.wmixerprep files put unks in the same place as MIXER does internally, and these should be used for training. Final results should use the non-*.wmixerprep files.


