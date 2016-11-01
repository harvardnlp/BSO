require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'

require 'models.lua'
require 'data.lua'
require 'util.lua'
require 'model_utils.lua'
require 'margin_beam_searcher.lua'
require 'con_margin_beam_searcher.lua'
require 'state_disq.lua'

require 'SynchedDropout.lua'

stringx = require('pl.stringx')

cmd = torch.CmdLine()

-- file location
cmd:option('-model', 'seq2seq_lstm_attn.t7.', [[Path to model .t7 file]])
cmd:option('-src_file', '',[[Source sequence to decode (one line per sequence)]])
cmd:option('-targ_file', '', [[True target sequence (optional)]])
cmd:option('-output_file', 'pred.txt', [[Path to output the predictions (each line will be the
    decoded sequence]])

cmd:option('-val_data_file','data/demo-val.hdf5',[[Path to validation *.hdf5 file
    from preprocess.py]])
cmd:option('-src_dict', 'data/demo.src.dict', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', 'data/demo.targ.dict', [[Path to target vocabulary (*.targ.dict file)]])

cmd:option('-fixed', false, 'fixed length search')
cmd:option('-con', '', 'empty, or one of "wo" or "sr"')
cmd:option('-clean', false, 'leave in <unk> etc')

-- beam search options
cmd:option('-beam_size', 5,[[Beam size]])
cmd:option('-max_sent_l', 250, [[Maximum sentence length. If any sequences in srcfile are longer
    than this then it will error out]])
cmd:option('-gpuid', -1,[[ID of the GPU to use (-1 = use CPU)]])

opt = cmd:parse(arg)

function sent2wordidx(sent, word2idx)
  local t = {}
  for word in sent:gmatch'([^%s]+)' do
    local idx = word2idx[word] or UNK
    table.insert(t, idx)
  end
  return torch.LongTensor(t)
end

-- skips START by default
function wordidxvec2sent(sent, idx2word, skip_end)
  local t = {}
  local start_i, end_i
  if skip_end then
    end_i = sent:size(1)-1
  else
    end_i = sent:size(1)
  end
  for i = 2, end_i do -- skip START and END
    table.insert(t, idx2word[sent[i]])
  end
  return table.concat(t, ' ')
end

function clean_sent(sent)
  local s = stringx.replace(sent, UNK_WORD, '')
  s = stringx.replace(s, START_WORD, '')
  s = stringx.replace(s, END_WORD, '')
  return s
end

function fwd_prop_enc(source, source_l, batch_l, train)
  local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
  local context = context_proto[{{1, batch_l}, {1, source_l}}]
  for t = 1, source_l do
    if train then
      encoder_clones[t]:training()
    else
      encoder_clones[t]:evaluate()
    end
    local encoder_input = {source[t], unpack(rnn_state_enc[t-1])}
    local out = encoder_clones[t]:forward(encoder_input)
    rnn_state_enc[t] = out
    context[{{},t}]:copy(out[#out]) -- copy final layer for t'th timestep (for entire batch)
  end
  return rnn_state_enc, context
end

function main()
  -- some globals
  PAD = 1; UNK = 2; START = 3; END = 4
  PAD_WORD = '<blank>'; UNK_WORD = '<unk>'; START_WORD = '<s>'; END_WORD = '</s>'
  assert(path.exists(opt.src_file), 'src_file does not exist')
  assert(path.exists(opt.model), 'model does not exist')

  -- parse input params
  if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
  end

  print('loading ' .. opt.model .. '...')
  checkpoint = torch.load(opt.model)
  print('done!')

  -- load model and word2idx/idx2word dictionaries
  model, model_opt = checkpoint[1], checkpoint[2]
  idx2word_src = idx2key(opt.src_dict)
  word2idx_src = flip_table(idx2word_src)
  idx2word_targ = idx2key(opt.targ_dict)
  word2idx_targ = flip_table(idx2word_targ)

  valid_data = data.new(model_opt, opt.val_data_file) -- just to init model and whatnot

  -- recreate model
  -- since using dropout v2, fine to just turn off dropout for eval
  model_opt.dropout = 0
  encoder = make_lstm(valid_data, model_opt, 'enc', 0)
  decoder = make_lstm2(valid_data, model_opt, 'dec', 0)
  generator, _ = make_generator(valid_data, model_opt)
  -- get rid of softmax
  generator.modules[2] = nil
  collectgarbage()

  layers = {encoder, decoder, generator}

  if opt.gpuid >= 0 then
    for i = 1, #layers do
      layers[i]:cuda()
    end
  end

  encoder:apply(get_layer)
  decoder:apply(get_layer)

  -- load model's params etc
  for j = 1, #layers do
    local p, gp = layers[j]:getParameters()
    local savedP, savedGp = model[j]:getParameters()
    p:copy(savedP)
  end

  MAX_SENT_L = opt.max_sent_l
  --print("Setting MAX_SENT_L:", MAX_SENT_L)

  context_proto = torch.zeros(1, MAX_SENT_L, model_opt.rnn_size)
  beam_context_proto = torch.zeros(opt.beam_size, MAX_SENT_L, model_opt.rnn_size)
  pred_pfx_proto = torch.LongTensor(MAX_SENT_L, 1):fill(1)
  source_proto = torch.LongTensor(model_opt.max_sent_l)
  inp_proto = torch.LongTensor(opt.beam_size):fill(1)
  local mask_proto = torch.Tensor(opt.beam_size*valid_data.target_size)
  local disq_idx_proto = torch.LongTensor(model_opt.max_sent_l)

  encoder_clones = clone_many_times(encoder, model_opt.max_sent_l)

  local h_init_enc = torch.zeros(1, model_opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init_enc = h_init_enc:cuda()
    context_proto = context_proto:cuda()
    beam_context_proto = beam_context_proto:cuda()
    pred_pfx_proto = pred_pfx_proto:cuda()
    inp_proto = inp_proto:cuda()
    source_proto = source_proto:cuda()
    mask_proto = mask_proto:cuda()
    disq_idx_proto = disq_idx_proto:cuda()
  end

  init_fwd_enc = {}
  for L = 1, model_opt.num_layers do
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_enc, h_init_enc:clone())
  end

  local sent_id = 0
  local file = io.open(opt.src_file, "r")
  local out_file = io.open(opt.output_file,'w')
  local mbbs
  if opt.con == '' then
    mbbs = MarginBatchBeamSearcher(model_opt.rnn_size, 2*model_opt.num_layers+1, opt.gpuid >= 0)
  else
    mbbs = ConMarginBatchBeamSearcher(model_opt.rnn_size, 2*model_opt.num_layers+1, opt.gpuid >= 0)
  end
  for line in file:lines() do
    sent_id = sent_id + 1
    if opt.clean then
      line = clean_sent(line)
    end
    print('SENT ' .. sent_id .. ': ' ..line)
    local source = sent2wordidx(line, word2idx_src)
    if opt.gpuid >= 0 then
      source = source:cuda():view(source:size(1),1)
    end
    local source_l = source:size(1)
    local disqs = {}
    if opt.con == "wo" then
      -- N.B. passing in source only works for WO b/c src and target vocabs are the same
      disqs[1] = WOStateDisqualifier(source:squeeze(2), opt.beam_size, valid_data.target_size,
        mask_proto, disq_idx_proto)
    elseif opt.con == "sr" then
      if not label_idxs then
        label_idxs = torch.CudaTensor(79)
        local ii = 1
        for w, b in pairs(dep_labels) do
          label_idxs[ii] = word2idx_targ[w]
          ii = ii + 1
        end
        assert(ii == 80)
      end
      disqs[1] = LabeledSRStateDisqualifier(source:squeeze(2), opt.beam_size, valid_data.target_size,
            mask_proto, disq_idx_proto, label_idxs, idx2word_src, word2idx_targ, idx2word_targ)
    elseif opt.con ~= "" then
      assert(false)
    end
    local rnn_state_enc, context = fwd_prop_enc(source, source_l, 1, false) -- train=false
    local beam_dec = decoder
    beam_dec:evaluate()
    local beam_scorer = generator
    beam_scorer:evaluate()
    mbbs:initSearch(opt.beam_size, 1, rnn_state_enc[source_l], source, source_l, model_opt.num_layers)
    local beam_ctx = beam_context_proto[{{1, opt.beam_size}, {1, source_l}}]
    local inps = inp_proto:sub(1, opt.beam_size)
    torch.repeatTensor(beam_ctx, context, opt.beam_size, 1, 1)
    inps[1] = START
    local search_len
    if opt.con == "sr" then
      search_len = 2*source_l
    elseif opt.con == "wo" then
      search_len = source_l + 1
    else
      search_len = MAX_SENT_L
    end
    local batch_preds
    if opt.con ~= "" then
      batch_preds = mbbs:evalFixedLengthSearch(inps, beam_ctx, beam_dec, beam_scorer, search_len-1, disqs)
    else
      batch_preds = mbbs:evalSearch(inps, beam_ctx, beam_dec, beam_scorer, search_len-1, END, disqs)
    end
    local pred = batch_preds[1]
    local predTensor = pred_pfx_proto:sub(1, search_len, 1, 1)
    local ptr = search_len
    while true do
      predTensor[ptr][1] = pred.val
      ptr = ptr - 1
      pred = pred.prev
      if pred == nil then
        break
      end
    end
    if opt.con ~= "" then assert(ptr == 0) end
    local actualPred = predTensor:sub(ptr+1, search_len)
    local pred_sent
    if opt.con ~= "" then -- don't skip end b/c we never predicted it
      pred_sent = wordidxvec2sent(actualPred:squeeze(2), idx2word_targ, false)
    else
      pred_sent = wordidxvec2sent(actualPred:squeeze(2), idx2word_targ, true)
    end
    out_file:write(pred_sent .. '\n')
    print('PRED ' .. sent_id .. ': ' .. pred_sent)
    print('')
  end
  out_file:close()
end

main()
