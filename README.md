Code for [Sequence-to-Sequence Learning as Beam-Search Optimization](http://aclweb.org/anthology/D/D16/D16-1137.pdf) (Wiseman and Rush, 2016).

This code is adapted from a much earlier version of Yoon Kim's [seq2seq-attn code](https://github.com/harvardnlp/seq2seq-attn).

For questions/concerns/bugs feel free to contact swiseman at seas.harvard.edu.

## Running Experiments
First prepare the data as in [data_prep/](https://github.com/harvardnlp/BSO/tree/master/data_prep).

All seq2seq baselines use the [seq2seq-attn code](https://github.com/harvardnlp/seq2seq-attn).

### Word-ordering Experiments
Pretrain with

```th pretrain.lua -data_file wo-train.hdf5 -val_data_file wo-val.hdf5 -savefile wopt -num_layers 2 -rnn_size 256 -word_vec_size 256 -save_after 10 -adagrad -layer_etas 0.02,0.01,0.2 -epochs 10 -curriculum 1 -dropout 0.2```

Unconstrained train with

```th bso_train.lua -data_file wo-train.hdf5 -val_data_file wo-val.hdf5 -savefile wosave -num_layers 2 -rnn_size 256 -word_vec_size 256 -adagrad -layer_etas 0.02,0.02,0.2 -curriculum 0 -epochs 39  -train_from wopt_epoch10.00_*.t7 -dropout 0.2 -max_beam_size 6 -beam_size 2```

Constrained training is accomplished by adding the argument '-con wo' to the above.

Predict with

``th predict.lua -val_data_file wo-val.hdf5 -model wosave_epoch39*.t7 -src_file wo-src-val.txt -src_dict wo.src.dict -targ_dict wo.targ.dict -beam_size 5 -con wo -output_file val-unconstrwo-preds.out```

Train the seq2seq baseline as

```th train.lua -data_file wo-train.hdf5 -val_data_file wo-val.hdf5 -savefile wos2s -num_layers 2 -rnn_size 256 -word_vec_size 256 -save_after 10 -param_init 0.1 -adagrad -layer_lrs 0.02,0.01,0.2 -lr_decay 1 -epochs 30 -curriculum 1 -dropout 0.2```

(and use the epoch with the lowest validation perplexity)

### Dependency Parsing Experiments
Pretrain with 

```th pretrain.lua -data_file dep-train.hdf5 -val_data_file dep-val.hdf5 -savefile deppt -num_layers 2 -rnn_size 300 -word_vec_size 300 -save_after 5 -adagrad -layer_etas 0.02,0.02,0.2 -epochs 5 -curriculum 1 -dropout 0.3 -pre_word_vecs_enc dep_src_w2v.h5 -pre_word_vecs_dec dep_targ_w2v.h5```

Constrained train with

```th bso_train.lua -data_file dep-train.hdf5 -val_data_file dep-val.hdf5 -savefile condep -num_layers 2 -rnn_size 300 -word_vec_size 300 -save_after 16 -adagrad -curriculum 0 -epochs 16 -train_from deppt_epoch5_*.t7 -dropout 0.3 -max_beam_size 6 -beam_size 2 -layer_etas 0.02,0.02,0.1 -ignore_eos -src_dict dep.src.dict -targ_dict dep.targ.dict -con sr ```

(Unconstrained training can be accomplished by leaving out the '-con sr' argument)

Predict with

```th predict.lua -val_data_file dep-val.hdf5 -model condep_epoch16_*.t7 -gpuid 1 -src_file dep-src-val.txt -src_dict dep.src.dict -targ_dict dep.targ.dict -beam_size 5 -con sr -output_file val-condepb5-preds.out```

Train the seq2seq baseline as

```th train.lua -data_file dep-train.hdf5 -val_data_file dep-val.hdf5 -savefile deps2s -num_layers 2 -rnn_size 300 -word_vec_size 300 -adagrad -layer_lrs 0.02,0.02,0.2 -lr_decay 1 -epochs 25 -curriculum 1 -dropout 0.3 -pre_word_vecs_enc dep_src_w2v.h5 -pre_word_vecs_dec dep_targ_w2v.h5```

(and use the epoch with the lowest validation perplexity)

### MT Experiments
Pretrain with

```th pretrain.lua -data_file mixer-train.hdf5 -val_data_file mixer-val.hdf5 -savefile mixerpt -num_layers 1 -rnn_size 256 -word_vec_size 256 -save_after 3 -adagrad -layer_etas 0.02,0.02,0.2 -epochs 3 -curriculum 1 -dropout 0.2```

Train with

```th bso_train.lua -data_file mixer-train.hdf5 -val_data_file mixer-val.hdf5 -savefile mixersave -num_layers 1 -rnn_size 256 -word_vec_size 256 -save_after 21 -adagrad -curriculum 0 -epochs 21 -train_from mixerpt_epoch3_*.t7 -dropout 0.2 -max_beam_size 6 -beam_size 2 -layer_etas 0.02,0.02,0.1 -mt_delt_multiple 1```

Predict with

```th predict.lua -val_data_file mixer-val.hdf5 -model mixersave_epoch21_*.t7 -src_file valid.de-en.de -src_dict mixer.src.dict -targ_dict mixer.targ.dict -beam_size 5 -output_file val-mixer-preds.out```

Train the seq2seq baseline as

```th pretrain.lua -data_file mixer-train.hdf5 -val_data_file mixer-val.hdf5 -savefile mixers2s -num_layers 1 -rnn_size 256 -word_vec_size 256 -adagrad -layer_lrs 0.02,0.02,0.2 -epochs 15 -lr_decay 1 -curriculum 1 -dropout 0.2```

(and use the epoch with the lowest validation perplexity).

MIT License.
