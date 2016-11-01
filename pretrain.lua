require 'nn'
require 'nngraph'
require 'hdf5'

require 'data.lua'
require 'util.lua'
require 'models.lua'
require 'model_utils.lua'
require 'bleu.lua'

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
cmd:option('-init_dec', 1, [[Initialize the hidden/cell state of the decoder at time
  0 to be the last hidden/cell state of the encoder. If 0,
  the initial states of the decoder are set to zero vectors]])

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
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
  pretrained word embeddings (hdf5 file) on the encoder side.
  See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
  pretrained word embeddings (hdf5 file) on the decoder side.
  See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', 0, [[If = 1, fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', 0, [[If = 1, fix word embeddings on the decoder side]])
cmd:option('-adagrad', false, 'use adagrad')
cmd:option('-layer_etas', "", 'comma separated learning rates')

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

cmd:option('-start_symbol', 0, [[Use special start-of-sentence and end-of-sentence tokens
  on the source side. We've found this to make minimal difference]])
-- GPU
cmd:option('-gpuid', -1, [[Which gpu to use. -1 = use CPU]])
-- bookkeeping
cmd:option('-save_every', 1, [[Save every this many epochs]])
cmd:option('-print_every', 50, [[Print stats after this many batches]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-save_after', 1, 'save starting at this epoch')

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

function train(train_data, valid_data)
  local timer = torch.Timer()
  local num_params = 0
  params, grad_params = {}, {}
  opt.train_perf = {}
  opt.val_perf = {}

  for i = 1, #layers do
    local p, gp = layers[i]:getParameters()
    p:uniform(-opt.param_init, opt.param_init)
    num_params = num_params + p:size(1)
    params[i] = p
    grad_params[i] = gp
  end

  collectgarbage()

  local optStates = {}
  for i = 1, #params do
    optStates[i] = {}
  end

  if opt.pre_word_vecs_enc:len() > 0 then
    print("loading enc vecs...")
    local f = hdf5.open(opt.pre_word_vecs_enc)
    local pre_word_vecs = f:read('w2vLT'):all()
    for i = 5, pre_word_vecs:size(1) do -- skip special symbol embeddings
      word_vecs_enc.weight[i]:copy(pre_word_vecs[i])
    end
  end

  if opt.pre_word_vecs_dec:len() > 0 then
    print("loading dec vecs...")
    local f = hdf5.open(opt.pre_word_vecs_dec)
    local pre_word_vecs = f:read('w2vLT'):all()
    for i = 5, pre_word_vecs:size(1) do
      word_vecs_dec.weight[i]:copy(pre_word_vecs[i])
    end
  end

  print("Number of parameters: " .. num_params)

  word_vecs_enc.weight[1]:zero()
  word_vecs_dec.weight[1]:zero()

  -- prototypes for gradients so there is no need to clone
  local encoder_grad_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)
  context_proto = torch.zeros(opt.max_batch_l, opt.max_sent_l, opt.rnn_size)

  -- clone encoder/decoder up to max source/target length
  decoder_clones_g = clone_many_times(decoder, opt.max_sent_l)
  encoder_clones = clone_many_times(encoder, opt.max_sent_l)
  for i = 1, opt.max_sent_l do
    decoder_clones_g[i]:apply(get_layer)
    if encoder_clones[i].apply then
      encoder_clones[i]:apply(function(m) m:setReuse() end)
    end
  end

  local h_init_dec = torch.zeros(opt.max_batch_l, opt.rnn_size)
  local h_init_enc = torch.zeros(opt.max_batch_l, opt.rnn_size)
  if opt.gpuid >= 0 then
    h_init_enc = h_init_enc:cuda()
    h_init_dec = h_init_dec:cuda()
    context_proto = context_proto:cuda()
    encoder_grad_proto = encoder_grad_proto:cuda()
  end

  init_fwd_enc = {}
  init_bwd_enc = {}

  init_fwd_dec_g = {h_init_dec:clone()} -- initial context
  init_bwd_dec_g = {h_init_dec:clone()} -- just need one copy of this

  for L = 1, opt.num_layers do
    table.insert(init_fwd_enc, h_init_enc:clone())
    table.insert(init_fwd_enc, h_init_enc:clone())

    table.insert(init_bwd_enc, h_init_enc:clone())
    table.insert(init_bwd_enc, h_init_enc:clone())

    table.insert(init_fwd_dec_g, h_init_dec:clone()) -- memory cell
    table.insert(init_fwd_dec_g, h_init_dec:clone()) -- hidden state

    table.insert(init_bwd_dec_g, h_init_dec:clone())
    table.insert(init_bwd_dec_g, h_init_dec:clone())
  end

  function pretrain_batch(data, epoch)
    local train_nonzeros = 0
    local train_loss = 0
    local batch_order = torch.randperm(data.length) -- shuffle mini batch order
    local start_time = timer:time().real
    local num_words_target = 0
    local num_words_source = 0
    local init_fwd_dec = init_fwd_dec_g
    local init_bwd_dec = init_bwd_dec_g
    local ptgen = generator
    local decoder_clones = decoder_clones_g

    for i = 1, data:size() do
      zero_table(grad_params, opt)
      local d
      if epoch <= opt.curriculum then
        d = data[i]
      else
        d = data[batch_order[i]]
      end
      local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
      local batch_l, target_l, source_l = d[5], d[6], d[7]

      local encoder_grads = encoder_grad_proto[{{1, batch_l}, {1, source_l}}]

      local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 0)
      local context = context_proto[{{1, batch_l}, {1, source_l}}]
      if opt.gpuid >= 0 then
        cutorch.setDevice(opt.gpuid)
      end

      -- forward prop encoder
      for t = 1, source_l do
        encoder_clones[t]:training()
        local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
        local out = encoder_clones[t]:forward(encoder_input)
        rnn_state_enc[t] = out
        context[{{},t}]:copy(out[#out])
      end

      -- forward prop decoder
      local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 0)
      if opt.init_dec == 1 then
        for L = 1, opt.num_layers do
          rnn_state_dec[0][L*2-1]:copy(rnn_state_enc[source_l][L*2-1])
          rnn_state_dec[0][L*2]:copy(rnn_state_enc[source_l][L*2])
        end
      end

      for t = 1, target_l do
        decoder_clones[t]:training()
        local decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
        rnn_state_dec[t] = decoder_clones[t]:forward(decoder_input)
      end

      -- backward prop decoder
      encoder_grads:zero()
      local drnn_state_dec = reset_state(init_bwd_dec, batch_l, 1)
      local loss = 0
      for t = target_l, 1, -1 do
        local pred = ptgen:forward(rnn_state_dec[t][#rnn_state_dec[t]])
        loss = loss + criterion:forward(pred, target_out[t])/batch_l
        local dl_dpred = criterion:backward(pred, target_out[t])
        dl_dpred:div(batch_l)
        local dl_dtarget = ptgen:backward(rnn_state_dec[t][#rnn_state_dec[t]], dl_dpred)
        drnn_state_dec[#drnn_state_dec]:add(dl_dtarget)
        local decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
        local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
        -- accumulate encoder/decoder grads
        encoder_grads:add(dlst[2])
        --drnn_state_dec[#drnn_state_dec]:copy(dlst[3])
        for j = 3, #dlst do
          drnn_state_dec[j-2]:copy(dlst[j])
        end
      end
      word_vecs_dec.gradWeight[1]:zero()
      if opt.fix_word_vecs_dec == 1 then
        word_vecs_dec.gradWeight:zero()
      end

      local grad_norm = 0
      grad_norm = grad_norm + grad_params[2]:norm()^2 + grad_params[3]:norm()^2

      -- backward prop encoder
      local drnn_state_enc = reset_state(init_bwd_enc, batch_l, 1)
      if opt.init_dec == 1 then
        for L = 1, opt.num_layers do
          drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
          drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
        end
      end

      for t = source_l, 1, -1 do
        local encoder_input = {source[t], table.unpack(rnn_state_enc[t-1])}
        drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
        local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
        for j = 1, #drnn_state_enc do
          drnn_state_enc[j]:copy(dlst[j+1])
        end
      end

      word_vecs_enc.gradWeight[1]:zero()
      if opt.fix_word_vecs_enc == 1 then
        word_vecs_enc.gradWeight:zero()
      end

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
        --stats = stats .. string.format('PPL: %.3f, |Param|: %.2f, |GParam|: %.2f, ',
        --  math.exp(train_loss/train_nonzeros), param_norm, grad_norm)
        stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
          (num_words_target+num_words_source) / time_taken,
          num_words_source / time_taken,
          num_words_target / time_taken)
        print(stats)
      end
      if i % 200 == 0 then
        collectgarbage()
      end
    end
    return train_loss, train_nonzeros
  end

  local epTimer = torch.Timer()
  local total_loss, total_nonzeros, batch_loss, batch_nonzeros

  for epoch = opt.start_epoch, opt.epochs do
    generator:training()
    local epStart = epTimer:time().real
    total_loss, total_nonzeros = pretrain_batch(train_data, epoch)
    print("epoch time:", epTimer:time().real - epStart)
    local train_score = total_loss/total_nonzeros
    print('Train', math.exp(train_score))
    opt.train_perf[#opt.train_perf + 1] = train_score
    local score = pteval(valid_data)
    opt.val_perf[#opt.val_perf + 1] = score
    -- clean and save models
    local savefile = string.format('%s_epoch%d_%.2f.t7', opt.savefile, epoch, score)
    if epoch >= opt.save_after and epoch % opt.save_every == 0 then
      print('saving checkpoint to ' .. savefile)
      clean_layer(encoder); clean_layer(decoder); clean_layer(generator)
      torch.save(savefile, {{encoder, decoder, generator}, opt})
    end
  end
end

function pteval(data)
  local init_fwd_dec = init_fwd_dec_g
  local ptgen = generator
  local decoder_clones = decoder_clones_g
  encoder_clones[1]:evaluate()
  decoder_clones[1]:evaluate() -- just need one clone
  ptgen:evaluate()
  local ngram_crct = torch.zeros(4)
  local ngram_total = torch.zeros(4)
  local nll = 0
  local total = 0
  for i = 1, data:size() do
    local d = data[i]
    local target, target_out, nonzeros, source = d[1], d[2], d[3], d[4]
    local batch_l, target_l, source_l = d[5], d[6], d[7]
    local rnn_state_enc = reset_state(init_fwd_enc, batch_l, 1)
    local context = context_proto[{{1, batch_l}, {1, source_l}}]
    -- forward prop encoder
    for t = 1, source_l do
      local encoder_input = {source[t], table.unpack(rnn_state_enc)}
      local out = encoder_clones[1]:forward(encoder_input)
      rnn_state_enc = out
      context[{{},t}]:copy(out[#out])
    end

    local rnn_state_dec = reset_state(init_fwd_dec, batch_l, 1)
    if opt.init_dec == 1 then
      for L = 1, opt.num_layers do
        rnn_state_dec[L*2-1]:copy(rnn_state_enc[L*2-1])
        rnn_state_dec[L*2]:copy(rnn_state_enc[L*2])
      end
    end

    local loss = 0
    for t = 1, target_l do
      local decoder_input = {target[t], context, table.unpack(rnn_state_dec)}
      local out = decoder_clones[1]:forward(decoder_input)
      rnn_state_dec = out
      local pred = ptgen:forward(out[#out])
      loss = loss + criterion:forward(pred, target_out[t])
    end
    nll = nll + loss
    total = total + nonzeros
  end
  local validppl = math.exp(nll/total)
  print("Valid Perp", validppl)
  return validppl
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

  -- Build model
  encoder = make_lstm(valid_data, opt, 'enc')
  decoder = make_lstm2(valid_data, opt, 'dec')
  generator, criterion = make_generator(valid_data, opt)

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
  local stringx, layer_eta_strs, pt_layer_eta_strs

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
