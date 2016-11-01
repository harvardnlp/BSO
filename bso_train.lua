require 'nn'
require 'nngraph'
require 'hdf5'

require 'data.lua'
require 'util.lua'
require 'models.lua'
require 'model_utils.lua'
require 'state_disq.lua'
require 'bleu.lua'
require 'margin_beam_searcher.lua'
require 'con_margin_beam_searcher.lua'

cmd = torch.CmdLine()

-- data files
cmd:text("")
cmd:text("**Data options**")
cmd:text("")
cmd:option('-data_file','data/demo-train.hdf5',[[Path to the training *.hdf5 file
                                               from preprocess.py]])
cmd:option('-val_data_file','data/demo-val.hdf5',[[Path to validation *.hdf5 file
                                                 from preprocess.py]])
cmd:option('-savefile', 'seq2seq_lstm_attn', [[Savefile name (model will be saved as
                         savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is
                         the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the
                                pretrained model.]])

-- rnn model specs
cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-num_layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 500, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 500, [[Word embedding sizes]])
cmd:option('-reverse_src', 0, [[If 1, reverse the source sequence. The original
                              sequence-to-sequence paper found that this was crucial to
                              achieving good performance, but with attention models this
                              does not seem necessary. Recommend leaving it to 0]])
cmd:option('-start_symbol', 0, [[Use special start-of-sentence and end-of-sentence tokens
                       on the source side. We've found this to make minimal difference]])
cmd:option('-init_dec', 1, [[Initialize the hidden/cell state of the decoder at time
                           0 to be the last hidden/cell state of the encoder. If 0,
                           the initial states of the decoder are set to zero vectors]])
cmd:option('-beam_size', 2, 'initial beam size')
cmd:option('-max_beam_size', 11, 'max beam size to train with')
cmd:option('-epochs_per_beam_size', 2,
    'increase beam by 1 after this many epochs')
cmd:option('-con', '', 'empty, or one of "wo" or "sr"')
cmd:option('-predcon', '', 'empty, or one of "wo" or "sr"')
cmd:option('-ignore_eos', false, 'ignore eos token for target')
cmd:option('-src_dict', '', [[Path to source vocabulary (*.src.dict file)]])
cmd:option('-targ_dict', '', [[Path to target vocabulary (*.targ.dict file)]])
cmd:option('-extra_room', 5, 'extra rheum')
cmd:option('-mt_delt_multiple', 0, 'scale 1-bleu by this')


cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

-- optimization
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoing, the epoch from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support
                               (-param_init, param_init)]])
cmd:option('-learning_rate', 0.01, [[Starting learning rate]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this, renormalize it
                                to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability.
                            Dropout is applied between vertical LSTM stacks.]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
                sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])
cmd:option('-adagrad', false, 'use adagrad')
cmd:option('-clip', false, 'clip rather than renorm')
cmd:option('-layer_etas', "", 'comma separated learning rates')

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])

-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-save_after', 15, 'save starting at this epoch')

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

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


function fwd_prop_dec(rnn_state_enc, source_l, batch_l, context, target, num_steps, pred, train)
  local rnn_state_dec
  if pred then
    rnn_state_dec = reset_state(init_fwd_dec_p, batch_l, 0)
  else
    rnn_state_dec = reset_state(init_fwd_dec_g, batch_l, 0)
  end
  -- initialize with last state of encoder
  if rnn_state_enc then
    for L = 1, opt.num_layers do -- leave last layer zeros (corresponds to prev attn)
      rnn_state_dec[0][L*2-1]:copy(rnn_state_enc[source_l][L*2-1])
      rnn_state_dec[0][L*2]:copy(rnn_state_enc[source_l][L*2])
    end
  end

  -- now push thru assuming using target_n as input
  local decoder_input
  local dclones, sclones
  if pred then
    dclones = decoder_clones_p
    sclones = generator_clones_p
  else
    dclones = decoder_clones_g
    sclones = generator_clones_g
  end
  for t = 1, num_steps do
    if train then
      dclones[t]:training()
    else
      dclones[t]:evaluate()
    end
    local decoder_input = {target[t], context, unpack(rnn_state_dec[t-1])}
    rnn_state_dec[t] = dclones[t]:forward(decoder_input)
    if train then
      sclones[t]:training()
    else
      sclones[t]:evaluate()
    end
    sclones[t]:forward(rnn_state_dec[t][#rnn_state_dec[t]])
  end
  return rnn_state_dec
end

function fwd_prop_spliced_dec(gold_rnn_state_dec, source_l, batch_l, context, preds,
    target_w, delt_vals, num_steps)
  -- initialize the same as gold decoder (since everyone sees same encoder etc)
  -- and also same 1st token, which is start token
  local dec_state_size = #gold_rnn_state_dec[0]
  local rnn_state_dec = reset_state(init_fwd_dec_p, batch_l, 0)

  for j = 1, dec_state_size do
    rnn_state_dec[0][j]:copy(gold_rnn_state_dec[0][j])
  end

  local dclones, sclones = decoder_clones_p, generator_clones_p

  -- now do remaining time-steps, splicing in when we did a reset
  for t = 1, num_steps do
    dclones[t]:training()
    local decoder_input = {preds[t], context, unpack(rnn_state_dec[t-1])}
    rnn_state_dec[t] = dclones[t]:forward(decoder_input)
    -- replace predicted states w/ gold ones when we reset
    for n = 1, batch_l do
      -- we could really copy for t=1 too...
      if t > 1 and t <= target_w[n] and delt_vals[t-1][n] > 0 then -- we reset @ prev step so this step is gold
        for j = 1, dec_state_size do
          rnn_state_dec[t][j][n]:copy(gold_rnn_state_dec[t][j][n])
        end
      end
    end
    sclones[t]:training()
    sclones[t]:forward(rnn_state_dec[t][#rnn_state_dec[t]])
  end
  return rnn_state_dec
end

function merge_backprop(ctx, target, batch_l, target_l, target_w, gold_rnn_state_dec, preds,
    pred_rnn_state_dec, step_delts, wrongs, rights, delt_vals, drnn_state_enc, encoder_grads)
  local drnn_g = reset_state(init_bwd_dec_g, batch_l, 1) -- gold rnn gradOuts
  local drnn_p = reset_state(init_bwd_dec_p, batch_l, 1) -- pred rnn gradOuts

  -- step_delts is batch_l x V, and we'll update it sparsely with flat_delts
  local flat_delts = step_delts:view(-1)

  -- N.B. delt_vals[t] are assumed to be 0 when no mistake made at step t...
  for t = target_l-1, 1, -1 do
    -- get gradOuts of final layer pred predictions
    --assert(torch.all(wrongs[t]:le(flat_delts:size(1))))
    flat_delts:indexCopy(1, wrongs[t], delt_vals[t])
    local dl_dpredt = generator_clones_p[t]:backward(pred_rnn_state_dec[t][#pred_rnn_state_dec[t]], step_delts)
    drnn_p[#drnn_p]:add(dl_dpredt) -- add in gradOut from prediction
    flat_delts:indexFill(1, wrongs[t], 0) -- reset to 0

    -- now get gradOuts of final layer gold predictions
    delt_vals[t]:mul(-1)
    --assert(torch.all(rights[t]:le(flat_delts:size(1))))
    flat_delts:indexCopy(1, rights[t], delt_vals[t])
    local dl_dgoldt = generator_clones_g[t]:backward(gold_rnn_state_dec[t][#gold_rnn_state_dec[t]], step_delts)
    drnn_g[#drnn_g]:add(dl_dgoldt) -- add in gradout from gold
    flat_delts:indexFill(1, rights[t], 0) -- reset

    for n = 1, batch_l do
      -- if we reset at previous time-step then current state is gold so add all gradOuts to gold
      -- (note we can treat t=0 as a reset)
      if t == 1 or (t <= target_w[n] and delt_vals[t-1][n] > 0) then -- tricky; delt_vals[t-1] still unmodified
        for j = 1, #drnn_g do
          drnn_g[j][n]:add(drnn_p[j][n])
          drnn_p[j][n]:zero()
        end
        -- assert(target[t][n] == preds[t][n])
      end
    end

    local gradIn_p = decoder_clones_p[t]:backward({preds[t], ctx, unpack(pred_rnn_state_dec[t-1])}, drnn_p)
    encoder_grads:add(gradIn_p[2])
    local gradIn_g = decoder_clones_g[t]:backward({target[t], ctx, unpack(gold_rnn_state_dec[t-1])}, drnn_g)
    encoder_grads:add(gradIn_g[2])

    -- update previous gradOuts with current gradIns
    for j = 3, #gradIn_p do
      drnn_p[j-2]:copy(gradIn_p[j])
      drnn_g[j-2]:copy(gradIn_g[j])
    end
  end -- end for t = target_l-1

  -- now we need to add gradients wrt first decoding timestep to encoder grads
  for L = 1, opt.num_layers do
    drnn_state_enc[L*2-1]:add(drnn_g[L*2-1])
    drnn_state_enc[L*2]:add(drnn_g[L*2])
    -- drnn_p should be all zero since we merged...
  end
end

function train(train_data, valid_data)
  local timer = torch.Timer()
  local num_params = 0
  params, grad_params = {}, {}
  opt.train_perf = {}
  opt.val_perf = {}

  for i = 1, #layers do
    local p, gp = layers[i]:getParameters()
    if opt.train_from:len() == 0 then
	    p:uniform(-opt.param_init, opt.param_init)
    end
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
  end

  if opt.train_from:len() > 0 then
    assert(path.exists(opt.train_from), 'checkpoint path invalid')
    print('loading ' .. opt.train_from .. '...')
    local checkpoint = torch.load(opt.train_from)
    local model, model_opt = checkpoint[1], checkpoint[2]
    opt.num_layers = model_opt.num_layers
    opt.rnn_size = model_opt.rnn_size
    local cpencoder = model[1]
    local cpdecoder = model[2]
    local cpgenerator = model[3]
    if #cpgenerator.modules > 1 then -- remove LogSoftMax (tho not really nec)
      assert(#cpgenerator.modules == 2)
      cpgenerator.modules[2] = nil
      collectgarbage()
    end
    local penc, genc = cpencoder:getParameters()
    params[1]:copy(penc)
    local pdec, gdec = cpdecoder:getParameters()
    params[2]:copy(pdec)
    local pgen, ggen = cpgenerator:getParameters()
    params[3]:copy(pgen)
  end
  collectgarbage()

  local optStates = {}
  for i = 1, #params do
    optStates[i] = {}
  end

  print("Number of parameters: " .. num_params)

  word_vecs_enc.weight[1]:zero()
  word_vecs_dec.weight[1]:zero()


  -- prototypes for gradients so there is no need to clone
  local encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  context_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  beam_context_proto = torch.zeros(opt.max_batch_l*opt.max_beam_size, opt.max_sent_l, opt.rnn_size)

  extra_room = 5
  pred_pfx_proto = torch.LongTensor(opt.max_sent_l+extra_room, opt.max_batch_l):fill(1)
  inp_proto = torch.LongTensor(opt.max_batch_l*opt.max_beam_size):fill(1)

  local wrong_proto = torch.LongTensor(opt.max_sent_l*opt.max_batch_l):fill(1)
  local right_proto = torch.LongTensor(opt.max_sent_l*opt.max_batch_l):fill(1)
  local delt_vals_proto = torch.Tensor(opt.max_sent_l*opt.max_batch_l)
  local last_resets_proto = torch.Tensor(opt.max_batch_l)
  local step_delt_proto = torch.Tensor(valid_data.target_size*opt.max_batch_l)
  local batch_delts_proto = torch.zeros(opt.max_batch_l)
  local batch_losses_proto = torch.zeros(opt.max_batch_l)
  local dropout_proto = torch.Tensor(opt.max_batch_l*opt.max_sent_l*opt.num_layers*opt.rnn_size)
  if opt.con ~= "" or opt.predcon ~= "" then
    mask_proto = torch.Tensor(opt.max_batch_l*opt.max_beam_size*valid_data.target_size)
    disq_idx_proto = torch.LongTensor(opt.max_batch_l*opt.max_sent_l)
  end

  -- clone encoder/decoder up to max source/target length
  decoder_clones_g = clone_many_times(decoder, opt.max_sent_l+1)
  decoder_clones_p = clone_many_times(decoder, opt.max_sent_l)
  encoder_clones = clone_many_times(encoder, opt.max_sent_l)
  for i = 1, opt.max_sent_l do
    decoder_clones_p[i]:apply(get_layer)
    decoder_clones_g[i]:apply(get_layer)
      --if decoder_clones_p[i].apply then
	  --  decoder_clones_p[i]:apply(function(m) m:setReuse() end)
      --end
      --if decoder_clones_g[i].apply then
	  --  decoder_clones_g[i]:apply(function(m) m:setReuse() end)
      --end
    if encoder_clones[i].apply then
	    encoder_clones[i]:apply(function(m) m:setReuse() end)
    end
  end
  -- get last one
  decoder_clones_g[opt.max_sent_l+1]:apply(get_layer)
   --if decoder_clones_g[opt.max_sent_l+1].apply then
   --  decoder_clones_g[opt.max_sent_l+1]:apply(function(m) m:setReuse() end)
   --end

  generator_clones_g = clone_many_times(generator, opt.max_sent_l+1)
  generator_clones_p = clone_many_times(generator, opt.max_sent_l)

  local h_init_dec = torch.zeros(opt.max_batch_l, opt.rnn_size)
  local h_init_enc = torch.zeros(opt.max_batch_l, opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init_enc = h_init_enc:cuda()
    h_init_dec = h_init_dec:cuda()
    context_proto = context_proto:cuda()
    encoder_grad_proto = encoder_grad_proto:cuda()
    pred_pfx_proto = pred_pfx_proto:cuda()
    inp_proto = inp_proto:cuda()
    wrong_proto = wrong_proto:cuda()
    right_proto = right_proto:cuda()
    delt_vals_proto = delt_vals_proto:cuda()
    last_resets_proto = last_resets_proto:cuda()
    step_delt_proto = step_delt_proto:cuda()
    batch_delts_proto = batch_delts_proto:cuda()
    batch_losses_proto = batch_losses_proto:cuda()
    beam_context_proto = beam_context_proto:cuda()
    dropout_proto = dropout_proto:cuda()
    if opt.con ~= "" or opt.predcon ~= "" then
      mask_proto = mask_proto:cuda()
      disq_idx_proto = disq_idx_proto:cuda()
    end
  end

  init_fwd_enc = {}
  init_bwd_enc = {}

  init_fwd_dec_p = {h_init_dec:clone()} -- initial context
  init_bwd_dec_p = {h_init_dec:clone()} -- just need one copy of this

  init_fwd_dec_g = {h_init_dec:clone()} -- initial context
  init_bwd_dec_g = {h_init_dec:clone()} -- just need one copy of this

  for L = 1, opt.num_layers do
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_enc, h_init_enc:clone())

    table.insert(init_bwd_enc, h_init_enc:clone())
    table.insert(init_bwd_enc, h_init_enc:clone())

    table.insert(init_fwd_dec_p, h_init_dec:clone()) -- memory cell
    table.insert(init_fwd_dec_p, h_init_dec:clone()) -- hidden state

    table.insert(init_fwd_dec_g, h_init_dec:clone()) -- memory cell
    table.insert(init_fwd_dec_g, h_init_dec:clone()) -- hidden state

    table.insert(init_bwd_dec_p, h_init_dec:clone())
    table.insert(init_bwd_dec_p, h_init_dec:clone())

    table.insert(init_bwd_dec_g, h_init_dec:clone())
    table.insert(init_bwd_dec_g, h_init_dec:clone())
  end

  -- set everyone up with global dropout
  local global_noise = dropout_proto:view(opt.num_layers*opt.max_sent_l,
                          opt.max_batch_l, opt.rnn_size)
  local synch_count = 0
  for ii = 1, opt.max_sent_l do
    assert(#decoder_clones_p[ii].modules == #decoder_clones_g[ii].modules)
    for jj = 1, #decoder_clones_p[ii].modules do
      for lay = 1, opt.num_layers do
        if decoder_clones_p[ii].modules[jj].name == "synched-dropout-" .. tostring(lay) then
          assert(decoder_clones_g[ii].modules[jj].name == "synched-dropout-" .. tostring(lay))
          decoder_clones_p[ii].modules[jj].global_noise = global_noise
          decoder_clones_g[ii].modules[jj].global_noise = global_noise
          decoder_clones_p[ii].modules[jj].noise_idx = (ii-1)*opt.num_layers + lay
          decoder_clones_g[ii].modules[jj].noise_idx = (ii-1)*opt.num_layers + lay
          synch_count = synch_count + 1
        end
      end
    end
  end
  assert(synch_count == opt.max_sent_l*opt.num_layers)

  -- set up the beam decoder to have consecutive idxs; global noise will be set elsewhere
  synch_count = 0
  for jj = 1, #decoder_clones_g[#decoder_clones_g].modules do
    for lay = 1, opt.num_layers do
      if decoder_clones_g[#decoder_clones_g].modules[jj].name == "synched-dropout-" .. tostring(lay) then
        decoder_clones_g[#decoder_clones_g].modules[jj].noise_idx = lay
        synch_count = synch_count + 1
      end
    end
  end
  assert(synch_count == opt.num_layers)

  function train_batch(data, epoch)
    local train_nonzeros = 0
    local train_loss = 0
    local batch_order = torch.randperm(data.length) -- shuffle mini batch order
    local start_time = timer:time().real
    local num_words_target = 0
    local num_words_source = 0

    local half_epoch = 0
    local mbbs
    if opt.con ~= "" then
      mbbs = ConMarginBatchBeamSearcher(opt.rnn_size, 2*opt.num_layers+1, opt.gpuid >= 0)
    else
      mbbs = MarginBatchBeamSearcher(opt.rnn_size, 2*opt.num_layers+1, opt.gpuid >= 0)
    end

    --set up beam decoder to index mbbs's noise
    local synch_count = 0
    for jj = 1, #decoder_clones_g[#decoder_clones_g].modules do
      if torch.type(decoder_clones_g[#decoder_clones_g].modules[jj]) == 'nn.SynchedDropout' then
        decoder_clones_g[#decoder_clones_g].modules[jj].global_noise = mbbs.expanded_global_noise
        synch_count = synch_count + 1
      end
    end
    assert(synch_count == opt.num_layers)

    for i = 1, data:size() do
      zero_table(grad_params, opt)
      local d
      if epoch <= opt.curriculum then
        d = data[i]
      else
        d = data[batch_order[i]]
      end
      --cutorch.setDevice(opt.gpuid)
      local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
      local batch_l, target_l, source_l, target_w = d[5], d[6], d[7], d[8]

      if opt.ignore_eos then
        target_l = target_l - 1
        target_w:add(-1)
      end

      -- generate this mini-batch's mask
      dropout_proto:sub(1,opt.num_layers*target_l*opt.max_batch_l*opt.rnn_size):bernoulli(
        1-opt.dropout):div(1-opt.dropout)

      local encoder_grads = encoder_grad_proto[{{1, batch_l}, {1, source_l}}]
      encoder_grads:zero()

      -- encode entire batch
      local rnn_state_enc, context = fwd_prop_enc(source, source_l, batch_l, true) -- train=true
      local gold_rnn_state_dec = fwd_prop_dec(rnn_state_enc, source_l, batch_l, context, target,
        target_l-1, false, true) -- pred = false, train = true

      local beam_dec = decoder_clones_g[#decoder_clones_g]
      beam_dec:training()
      -- need to get this guy's dropout
      local beam_scorer = generator_clones_g[#generator_clones_g]
      beam_scorer:training()

      mbbs:initSearch(opt.beam_size, batch_l, gold_rnn_state_dec[0], target, target_l, opt.num_layers)
      local beam_ctx = beam_context_proto[{{1,opt.beam_size*batch_l}, {1, source_l}}]
      local inps = inp_proto:sub(1, batch_l*opt.beam_size)
      local disqs
      if opt.con ~= "" then
        disqs = {}
      end
      for n = 1, batch_l do
        torch.repeatTensor(beam_ctx:sub((n-1)*opt.beam_size+1,n*opt.beam_size),
          context:sub(n,n), opt.beam_size, 1, 1)
        inps[(n-1)*opt.beam_size+1] = target[1][n]
        if opt.con == "sr" then
          disqs[n] = LabeledSRStateDisqualifier(source:select(2,n), opt.beam_size, valid_data.target_size,
            mask_proto:sub((n-1)*opt.beam_size*valid_data.target_size+1,
            n*opt.beam_size*valid_data.target_size),disq_idx_proto:sub((n-1)*source_l+1,n*source_l),
            dep_label_idxs, idx2word_src, word2idx_targ, idx2word_targ)
        elseif opt.con == "wo" then
          disqs[n] = WOStateDisqualifier(source:select(2,n), opt.beam_size, valid_data.target_size,
            mask_proto:sub((n-1)*opt.beam_size*valid_data.target_size+1,
              n*opt.beam_size*valid_data.target_size),
              disq_idx_proto:sub((n-1)*source_l+1,n*source_l))
        end
      end

      local drnn_state_enc = reset_state(init_bwd_enc, batch_l, 1) -- gradients wrt enc timesteps
      local step_delts = step_delt_proto:sub(1, batch_l*valid_data.target_size):view(batch_l, valid_data.target_size)
      local preds = pred_pfx_proto:sub(1, target_l-1, 1, batch_l)
      local wrongs = wrong_proto:sub(1, (target_l-1)*batch_l):view(target_l-1, batch_l)
      wrongs:fill(1)
      local rights = right_proto:sub(1, (target_l-1)*batch_l):view(target_l-1, batch_l)
      rights:fill(1)
      local delt_vals = delt_vals_proto:sub(1, (target_l-1)*batch_l):view(target_l-1, batch_l)
      delt_vals:zero()
      local last_resets = last_resets_proto:sub(1, batch_l)
      last_resets:zero()
      local batch_losses = batch_losses_proto:sub(1, batch_l)
      local batch_delts = batch_delts_proto:sub(1, batch_l) -- only gonna use these for penult

      local loss = 0
      for t = 1, target_l-2 do
        local bpreds = mbbs:nextSearchStep(t, inps, beam_ctx, beam_dec, beam_scorer,
          generator_clones_g[t].output, target, target_w, gold_rnn_state_dec,
          delt_vals[t], batch_losses, global_noise, disqs)
        for n = 1, batch_l do
          if t <= target_w[n]-1 then -- go up to actual_l-2
            loss = loss + batch_losses[n]
            if delt_vals[t][n] > 0 then -- otherwise assumed to be 0
              local pred = bpreds[n]
              local last_bad_pred = pred.val
              wrongs[t][n] = (n-1)*valid_data.target_size + pred.val
              rights[t][n] = (n-1)*valid_data.target_size + target[t+1][n]
              pred = pred.prev
              for t2 = t, last_resets[n]+1, -1 do
                preds[t2][n] = pred.val
                pred = pred.prev
              end
              if opt.mt_delt_multiple > 0 then
                -- reset delt_vals[t][n]
                local tempPredTbl = preds:sub(last_resets[n]+1, t, n, n):squeeze(2):totable()
                table.insert(tempPredTbl, last_bad_pred)
                delt_vals[t][n] = opt.mt_delt_multiple*(1- get_bleu(tempPredTbl,
                                  target:sub(last_resets[n]+1, t+1, n, n):squeeze(2)))
              end
              mbbs:resetBeam(n, gold_rnn_state_dec, inps, target, t)
              last_resets[n] = t
            end
          end -- end if t <= target_w[n]-1
        end -- end for n = 1
      end -- end for t = 1

      -- do final time-step
      local bpreds = mbbs:finalStep(inps, beam_ctx, beam_dec, beam_scorer,
        generator_clones_g, target, target_w, batch_delts, batch_losses, global_noise, disqs)
      for n = 1, batch_l do
        local penult = target_w[n]
        loss = loss + batch_losses[n]
        if batch_delts[n] > 0 then
          local pred = bpreds[n]
          local last_bad_pred = pred.val
          delt_vals[penult][n] = batch_delts[n]
          wrongs[penult][n] = (n-1)*valid_data.target_size + pred.val
          rights[penult][n] = (n-1)*valid_data.target_size + target[penult+1][n]
          pred = pred.prev
          for t2 = penult, last_resets[n]+1, -1 do
            preds[t2][n] = pred.val
            pred = pred.prev
          end
          if opt.mt_delt_multiple > 0 then
            local tempPredTbl = preds:sub(last_resets[n]+1, penult, n, n):squeeze(2):totable()
            table.insert(tempPredTbl, last_bad_pred)
            delt_vals[penult][n] = opt.mt_delt_multiple*(1-get_bleu(tempPredTbl,
                                   target:sub(last_resets[n]+1, penult, n, n):squeeze(2)))
          end
        else
          delt_vals[penult][n] = 0
        end
      end

      -- recreate fprop of wrong preds
      local pred_rnn_state_dec = fwd_prop_spliced_dec(gold_rnn_state_dec, source_l, batch_l, context, preds,
        target_w, delt_vals, target_l-1)

      -- backprop through all wrong preds
      merge_backprop(context, target, batch_l, target_l, target_w, gold_rnn_state_dec, preds,
        pred_rnn_state_dec, step_delts, wrongs, rights, delt_vals, drnn_state_enc, encoder_grads)

      word_vecs_dec.gradWeight[1]:zero()
      local grad_norm = 0
      grad_norm = grad_norm + grad_params[2]:norm()^2 + grad_params[3]:norm()^2

      -- after accumulating all encoder grads, we backprop enc in batch
      for t = source_l, 1, -1 do
        local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
        drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
        local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
        for j = 1, #drnn_state_enc do
          drnn_state_enc[j]:copy(dlst[j+1])
        end
      end

      word_vecs_enc.gradWeight[1]:zero()
      grad_norm = (grad_norm + grad_params[1]:norm()^2)^0.5

      -- Shrink norm and update params
      local param_norm = 0
      local shrinkage = opt.max_grad_norm / grad_norm
      for j = 1, #grad_params do
        if shrinkage < 1 then
          grad_params[j]:mul(shrinkage)
        end
        if opt.adagrad then
          adagradStep(params[j], grad_params[j], layer_etas[j], optStates[j])
        else
          params[j]:add(-opt.learning_rate, grad_params[j])
        end
        param_norm = param_norm + params[j]:norm()^2
      end
      param_norm = param_norm^0.5

      -- Bookkeeping
      num_words_target = num_words_target + batch_l*target_l
      num_words_source = num_words_source + batch_l*source_l
      train_nonzeros = train_nonzeros + nonzeros
      train_loss = train_loss + loss*batch_l
      local time_taken = timer:time().real - start_time
      if i % opt.print_every == 0 then
        local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
          epoch, i, data:size(), batch_l, opt.learning_rate)
        stats = stats .. string.format('Loss: %.4f, |Param|: %.2f, |GParam|: %.2f, ',
          train_loss/train_nonzeros, param_norm, grad_norm)
        stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
          (num_words_target+num_words_source) / time_taken,
          num_words_source / time_taken,
          num_words_target / time_taken)
        print(stats)
      end
      if i % 200 == 0 then
        collectgarbage()
      end
    end -- end for i = 1
    return train_loss, train_nonzeros
  end

  local epTimer = torch.Timer()
  local total_loss, total_nonzeros, batch_loss, batch_nonzeros

  --print("beam", opt.beam_size, "mb", opt.max_beam_size)
  for epoch = opt.start_epoch, opt.epochs do
    generator_clones_g[1]:training()
    local epStart = epTimer:time().real
    total_loss, total_nonzeros = train_batch(train_data, epoch)
    print("epoch time:", epTimer:time().real - epStart)
    local train_score = total_loss/total_nonzeros
    print('Train', train_score)
    opt.train_perf[#opt.train_perf + 1] = train_score
    local score
    if opt.predcon == "" then
      score = unconstrained_eval(valid_data)
    else
      score = eval(valid_data)
    end
    opt.val_perf[#opt.val_perf + 1] = -score
    -- clean and save models
    local savefile = string.format('%s_epoch%d_%.2f.t7', opt.savefile, epoch, score)
    if epoch >= opt.save_after and epoch % opt.save_every == 0 then
      print('saving checkpoint to ' .. savefile)
      clean_layer(encoder); clean_layer(decoder); clean_layer(generator)
      torch.save(savefile, {{encoder, decoder, generator}, opt})
    end

    if epoch % opt.epochs_per_beam_size == 0 then
      if opt.beam_size < opt.max_beam_size then
        opt.beam_size = opt.beam_size+1
        for j = 1, #optStates do
          if optStates[j].var then
            optStates[j].var:zero()
            optStates[j].std:zero()
          end
        end
        print("beam size now", opt.beam_size)
      end
    end
   end -- end for epoch
end

function eval(data)
  print("eval'ing with beam size", opt.beam_size)
  local START, END = 3, 4
  local ngram_crct = torch.zeros(4)
  local ngram_total = torch.zeros(4)
  local total_gold_l, total_pred_l = 0, 0
  local mbbs
  if opt.predcon ~= "" then
    mbbs = ConMarginBatchBeamSearcher(opt.rnn_size, 2*opt.num_layers+1, opt.gpuid >= 0)
  else
    mbbs = MarginBatchBeamSearcher(opt.rnn_size, 2*opt.num_layers+1, opt.gpuid >= 0)
  end
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l, target_w = d[5], d[6], d[7], d[8]
    if opt.ignore_eos then
      target_l = target_l - 1
      target_w:add(-1)
    end

    local max_len
    if opt.sr ~= "" then
      max_len = opt.predcon == "sr" and 2*source_l or source_l + 1
    else
      max_len = opt.max_sent_l + opt.extra_room
    end
    --cutorch.setDevice(opt.gpuid)
    -- fwd prop encoder
    local rnn_state_enc, context = fwd_prop_enc(source, source_l, batch_l, false) -- train=false

    -- now beam search
    local beam_dec = decoder_clones_g[#decoder_clones_g]
    beam_dec:evaluate()
    local beam_scorer = generator_clones_g[#generator_clones_g]
    beam_scorer:evaluate()

    mbbs:initSearch(opt.beam_size, batch_l, rnn_state_enc[source_l], target, target_l, opt.num_layers)
    local beam_ctx = beam_context_proto[{{1, opt.beam_size*batch_l}, {1, source_l}}]
    local inps = inp_proto:sub(1, opt.beam_size*batch_l)
    local disqs
    if opt.predcon ~= "" then
      disqs = {}
    end

    for n = 1, batch_l do
      torch.repeatTensor(beam_ctx:sub((n-1)*opt.beam_size+1, n*opt.beam_size), context:sub(n,n),
        opt.beam_size, 1, 1)
      inps[(n-1)*opt.beam_size+1] = target[1][n]
      if opt.predcon == "sr" then
        disqs[n] = LabeledSRStateDisqualifier(source:select(2,n), opt.beam_size, valid_data.target_size,
          mask_proto:sub((n-1)*opt.beam_size*valid_data.target_size+1,
          n*opt.beam_size*valid_data.target_size),disq_idx_proto:sub((n-1)*source_l+1,n*source_l),
          dep_label_idxs, idx2word_src, word2idx_targ, idx2word_targ)
      elseif opt.predcon == "wo" then
        disqs[n] = WOStateDisqualifier(source:select(2,n), opt.beam_size, valid_data.target_size,
          mask_proto:sub((n-1)*opt.beam_size*valid_data.target_size+1,
            n*opt.beam_size*valid_data.target_size),
            disq_idx_proto:sub((n-1)*source_l+1,n*source_l))
      end
    end
    local batch_preds
    if opt.predcon ~= "" then
      batch_preds = mbbs:evalFixedLengthSearch(inps, beam_ctx, beam_dec, beam_scorer,
          max_len-1, disqs)
    else
      batch_preds = mbbs:evalSearch(inps, beam_ctx, beam_dec, beam_scorer,
          max_len-1, END)
    end

    for n = 1, batch_l do
      local actual_len_with_end = target_w[n]
      local pred = batch_preds[n]
      local predTensor = pred_pfx_proto:sub(1, max_len, 1, 1)
      local ptr = max_len
      while true do
        predTensor[ptr][1] = pred.val
        ptr = ptr - 1
        pred = pred.prev
        if pred == nil then
          break
        end
      end
      local actualPred = predTensor:sub(ptr+1, max_len)
      assert(opt.predcon == "" or ptr == 0)
      assert(actualPred[1][1] == START)
      -- but don't subtract last thing in case we don't find and EOS
      local pred_sent = actualPred:sub(2, actualPred:size(1)):squeeze(2):totable()
      local gold_sent = target:sub(2, 1+actual_len_with_end, n, n):squeeze(2):totable()
      local prec = get_ngram_prec(pred_sent, gold_sent, 4)
      for ii = 1, 4 do
        ngram_crct[ii] = ngram_crct[ii] + prec[ii][2]
        ngram_total[ii] = ngram_total[ii] + prec[ii][1]
      end
      -- we'll ignore start but not end token in determining length
      total_gold_l = total_gold_l + actual_len_with_end
      total_pred_l = total_pred_l + actualPred:size(1)-1
    end -- end for n = 1
  end -- end for i = 1
  ngram_crct:cdiv(ngram_total)
  print("Accs", ngram_crct[1], ngram_crct[2], ngram_crct[3], ngram_crct[4])
  ngram_crct:log()
  local bp = math.exp(1 - math.max(1, total_gold_l/total_pred_l)) -- brevity penalty
  print("bp", bp)
  local bleu = bp*math.exp(ngram_crct:sum()/4)
  print("bleu", bleu)
  return bleu
end

function unconstrained_eval(data)
  print("eval'ing with beam size", opt.beam_size)
  local START, END = 3, 4
  local max_len = opt.max_sent_l + opt.extra_room
  local ngram_crct = torch.zeros(4)
  local ngram_total = torch.zeros(4)
  local total_gold_l, total_pred_l = 0, 0
  local mbbs = MarginBatchBeamSearcher(opt.rnn_size, 2*opt.num_layers+1, opt.gpuid >= 0)
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l, target_w = d[5], d[6], d[7], d[8]
    --cutorch.setDevice(opt.gpuid)
    -- fwd prop encoder
    local rnn_state_enc, context = fwd_prop_enc(source, source_l, batch_l, false) -- train=false

    -- now beam search
    local beam_dec = decoder_clones_g[#decoder_clones_g]
    beam_dec:evaluate()
    local beam_scorer = generator_clones_g[#generator_clones_g]
    beam_scorer:evaluate()

    mbbs:initSearch(opt.beam_size, batch_l, rnn_state_enc[source_l], target, target_l, opt.num_layers)
    local beam_ctx = beam_context_proto[{{1, opt.beam_size*batch_l}, {1, source_l}}]
    local inps = inp_proto:sub(1, opt.beam_size*batch_l)
    for n = 1, batch_l do
      torch.repeatTensor(beam_ctx:sub((n-1)*opt.beam_size+1, n*opt.beam_size), context:sub(n,n),
        opt.beam_size, 1, 1)
      inps[(n-1)*opt.beam_size+1] = target[1][n]
    end
    local batch_preds = mbbs:evalSearch(inps, beam_ctx, beam_dec, beam_scorer,
      max_len-1, END)

    for n = 1, batch_l do
      local actual_len_with_end = target_w[n]
      local pred = batch_preds[n]
      local predTensor = pred_pfx_proto:sub(1, max_len, 1, 1)
      local ptr = max_len
      while true do
        predTensor[ptr][1] = pred.val
        ptr = ptr - 1
        pred = pred.prev
        if pred == nil then
          break
        end
      end
      local actualPred = predTensor:sub(ptr+1, max_len)
      assert(actualPred[1][1] == START)
      local pred_sent = actualPred:sub(2, actualPred:size(1)):squeeze(2):totable()
      local gold_sent = target:sub(2, 1+actual_len_with_end, n, n):squeeze(2):totable()
      local prec = get_ngram_prec(pred_sent, gold_sent, 4)
      for ii = 1, 4 do
        ngram_crct[ii] = ngram_crct[ii] + prec[ii][2]
        ngram_total[ii] = ngram_total[ii] + prec[ii][1]
      end
      -- we'll ignore start but not end token in determining length
      total_gold_l = total_gold_l + actual_len_with_end
      total_pred_l = total_pred_l + actualPred:size(1)-1
    end -- end for n = 1
  end -- end for i = 1
  ngram_crct:cdiv(ngram_total)
  print("Accs", ngram_crct[1], ngram_crct[2], ngram_crct[3], ngram_crct[4])
  ngram_crct:log()
  local bp = math.exp(1 - math.max(1, total_gold_l/total_pred_l)) -- brevity penalty
  print("bp", bp)
  local bleu = bp*math.exp(ngram_crct:sum()/4)
  print("bleu", bleu)
  return bleu
end

function main()
  -- parse input params
  opt = cmd:parse(arg)
  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid)
    cutorch.manualSeed(opt.seed)
  end

  -- Create the data loader class.
  print('loading data...')
  train_data = data.new(opt, opt.data_file)
  valid_data = data.new(opt, opt.val_data_file)
  print('done!')
  print(string.format('Source vocab size: %d, Target vocab size: %d',
      valid_data.source_size, valid_data.target_size))
  opt.max_sent_l = math.max(valid_data.source:size(2), valid_data.target:size(2))
  opt.max_batch_l = math.max(valid_data.batch_l:max(), train_data.batch_l:max())
  print(string.format('Source max sent len: %d, Target max sent len: %d',
      valid_data.source:size(2), valid_data.target:size(2)))

  if opt.src_dict:len() > 0 then
    assert(opt.targ_dict:len() > 0)
    idx2word_src = idx2key(opt.src_dict)
    word2idx_src = flip_table(idx2word_src)
    idx2word_targ = idx2key(opt.targ_dict)
    word2idx_targ = flip_table(idx2word_targ)
  end

  if opt.con == "sr" or opt.predcon == "sr" then
    dep_label_idxs = get_dep_label_idxs(word2idx_targ, opt.gpuid >= 0)
  end

  -- Build model
  encoder = make_lstm(valid_data, opt, 'enc')
  --decoder = make_lstm2(valid_data, opt, 'dec', opt.use_chars_dec)
  decoder = make_sd_lstm_dec(valid_data, opt, 'dec')
  generator, criterion = make_generator(valid_data, opt)
  -- get rid of softmax
  generator.modules[2] = nil
  collectgarbage()

  layers = {encoder, decoder, generator}

  if opt.gpuid >= 0 then
    for i = 1, #layers do
      layers[i]:cuda()
    end
    criterion:cuda()
  end

  encoder:apply(get_layer)
  decoder:apply(get_layer)

  layer_etas = {}
  local stringx, layer_eta_strs

  if opt.layer_etas ~= "" then
    stringx = require('pl.stringx')
    layer_eta_strs = stringx.split(opt.layer_etas, ",")
    assert(#layer_eta_strs == 3)
  end
  for j = 1, #layers do
    if layer_eta_strs then
      layer_etas[j] = tonumber(layer_eta_strs[j])
    else
      layer_etas[j] = opt.learning_rate
    end
  end

  print("layer_etas", layer_etas[1], layer_etas[2], layer_etas[3])
  train(train_data, valid_data)
end

main()
