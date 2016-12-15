## Creating the Word-ordering Dataset
I'm waiting for permission to upload files/scripts...will update soon!

## Creating the Dependency Parsing Dataset
Please follow these steps to recreate the dependency parsing dataset:

- Obtain Stanford dependency trees by running the Stanford Dependency Converter, e.g.,

```java -cp stanford-parser-3.3.0.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile  ~/Projects/group/data/wsj/test.txt     -conllx -basic > test.conll.txt```

- Use dep/convert.py to produce target sequences from the conll files

- Remove the final ROOT word and @R_ROOT from the generated target sequences

- To obtain CONLL format files from predicted target sequences, use dep/convert_back_noroot.py, e.g.,

```python convert_back_noroot.py < dev-preds.out > dev-preds.out.conll```

## Creating the MT Dataset
To creat the MIXER dataset, I used the data-preparation code at https://github.com/facebookresearch/MIXER, but modified it to output text files (which the BSO code consumes), rather than torch Tensors. The modified files (prepareData.sh, makedata.lua, and tokenizer.lua) are in the MT/ directory, and you should be able to just run:

```bash prepareData.sh```


