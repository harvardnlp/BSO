function nn.Module:reuseMem()
  self.reuse = true
  return self
end

function nn.Module:setReuse()
  if self.reuse then
    self.gradInput = self.output
  end
end

function make_lstm(data, opt, model)
  assert(model == 'enc' or model == 'dec')
  local name = '_' .. model
  local dropout = opt.dropout or 0
  local n = opt.num_layers
  local rnn_size = opt.rnn_size
  local input_size

  input_size = opt.word_vec_size

  local offset = 0
  -- there will be 2*n+3 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
  if model == 'dec' then
    table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
    table.insert(inputs, nn.Identity()()) -- prev context_attn (batch_size x rnn_size)
    offset = offset + 2
  end
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
      local word_vecs
      if model == 'enc' then
        word_vecs = nn.LookupTable(data.source_size, input_size)
      else
        word_vecs = nn.LookupTable(data.target_size, input_size)
      end
      word_vecs.name = 'word_vecs' .. name
      x = word_vecs(inputs[1]) -- batch_size x word_vec_size
      input_size_L = input_size
      if model == 'dec' then
        x = nn.JoinTable(2)({x, inputs[1+offset]}) -- batch_size x (word_vec_size + rnn_size)
        input_size_L = input_size + rnn_size
      end
    else
      x = outputs[(L-1)*2]
      input_size_L = rnn_size
      if dropout > 0 then
        x = nn.Dropout(dropout, nil, false)(x)
      end
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if model == 'dec' then
    local top_h = outputs[#outputs]
    local decoder_attn = make_decoder_attn(data, opt)
    decoder_attn.name = 'decoder_attn'
    local attn_out = decoder_attn({top_h, inputs[offset]})
    if dropout > 0 then --and not opt.ignore_attn_do then
      attn_out = nn.Dropout(dropout, nil, false)(attn_out)
    end
    table.insert(outputs, attn_out)
  end
  return nn.gModule(inputs, outputs)
end

-- just changes order of inputs so don't have to move stuff around during training
function make_lstm2(data, opt, model)
  assert(model == 'enc' or model == 'dec')
  local name = '_' .. model
  local dropout = opt.dropout or 0
  local n = opt.num_layers
  local rnn_size = opt.rnn_size
  local input_size
  input_size = opt.word_vec_size
  local offset = 0
  -- there will be 2*n+3 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
  if model == 'dec' then
    offset = 1
    table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
  end
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  if model == 'dec' then
    table.insert(inputs, nn.Identity()()) -- prev context attn (batch_size x rnn_size)
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
      local word_vecs
      if model == 'enc' then
        word_vecs = nn.LookupTable(data.source_size, input_size)
      else
        word_vecs = nn.LookupTable(data.target_size, input_size)
      end
      word_vecs.name = 'word_vecs' .. name
      x = word_vecs(inputs[1]) -- batch_size x word_vec_size
      input_size_L = input_size
      if model == 'dec' then
        x = nn.JoinTable(2)({x, inputs[#inputs]}) -- batch_size x (word_vec_size + rnn_size)
        input_size_L = input_size + rnn_size
      end
    else
      x = outputs[(L-1)*2]
      input_size_L = rnn_size
      if dropout > 0 then
        x = nn.Dropout(dropout, nil, false)(x)
      end
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if model == 'dec' then
    local top_h = outputs[#outputs]
    local decoder_attn = make_decoder_attn(data, opt)
    decoder_attn.name = 'decoder_attn'
    local attn_out = decoder_attn({top_h, inputs[2]})
    if dropout > 0 then --and not opt.ignore_attn_do then
      attn_out = nn.Dropout(dropout, nil, false)(attn_out)
    end
    table.insert(outputs, attn_out)
  end
  return nn.gModule(inputs, outputs)
end

-- just changes order of inputs so don't have to move stuff around during training
function make_sd_lstm_dec(data, opt, model)
  require 'SynchedDropout.lua'
  assert(model == 'dec')
  local name = '_' .. model
  local dropout = opt.dropout or 0
  local n = opt.num_layers
  local rnn_size = opt.rnn_size
  local input_size
  input_size = opt.word_vec_size
  local offset = 0
  -- there will be 2*n+3 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
  if model == 'dec' then
    offset = 1
    table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
  end
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end
  if model == 'dec' then
    table.insert(inputs, nn.Identity()()) -- prev context attn (batch_size x rnn_size)
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
      local word_vecs
      if model == 'enc' then
        word_vecs = nn.LookupTable(data.source_size, input_size)
      else
        word_vecs = nn.LookupTable(data.target_size, input_size)
      end
      word_vecs.name = 'word_vecs' .. name
      x = word_vecs(inputs[1]) -- batch_size x word_vec_size
      input_size_L = input_size
      if model == 'dec' then
        x = nn.JoinTable(2)({x, inputs[#inputs]}) -- batch_size x (word_vec_size + rnn_size)
        input_size_L = input_size + rnn_size
      end
    else
      x = outputs[(L-1)*2]
      input_size_L = rnn_size
      if dropout > 0 then
        local sdo = nn.SynchedDropout(dropout, nil, nil, false, false)
        sdo.name = "synched-dropout-" .. tostring(L-1)
        x = sdo(x)
      end
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size):reuseMem()(x)
    local h2h = nn.LinearNoBias(rnn_size, 4 * rnn_size):reuseMem()(prev_h)
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid():reuseMem()(n1)
    local forget_gate = nn.Sigmoid():reuseMem()(n2)
    local out_gate = nn.Sigmoid():reuseMem()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh():reuseMem()(n4)
    -- perform the LSTM update
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh():reuseMem()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  if model == 'dec' then
    local top_h = outputs[#outputs]
    local decoder_attn = make_decoder_attn(data, opt)
    decoder_attn.name = 'decoder_attn'
    local attn_out = decoder_attn({top_h, inputs[2]})
    if dropout > 0 then --and not opt.ignore_attn_do then
      --if dropout > 0 then
      local sdo2 = nn.SynchedDropout(dropout, nil, nil, false, false)
      sdo2.name = "synched-dropout-" .. tostring(opt.num_layers)
      attn_out = sdo2(attn_out)
    end
    table.insert(outputs, attn_out)
  end
  return nn.gModule(inputs, outputs)
end

function make_decoder_attn(data, opt, simple)
  -- 2D tensor target_t (batch_l x rnn_size) and
  -- 3D tensor for context (batch_l x source_l x rnn_size)

  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local target_t = nn.LinearNoBias(opt.rnn_size, opt.rnn_size)(inputs[1])
  local context = inputs[2]
  simple = simple or 0
  -- get attention

  local attn = nn.MM()({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
  attn = nn.Sum(3)(attn)
  local softmax_attn = nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  attn = softmax_attn(attn)
  attn = nn.Replicate(1,2)(attn) -- batch_l x 1 x source_l

  -- apply attention to context
  local context_combined = nn.MM()({attn, context}) -- batch_l x 1 x rnn_size
  context_combined = nn.Sum(2)(context_combined) -- batch_l x rnn_size
  local context_output
  if simple == 0 then
    context_combined = nn.JoinTable(2)({context_combined, inputs[1]}) -- batch_l x rnn_size*2
    context_output = nn.Tanh()(nn.LinearNoBias(opt.rnn_size*2,
        opt.rnn_size)(context_combined))
  else
    context_output = nn.CAddTable()({context_combined,inputs[1]})
  end
  return nn.gModule(inputs, {context_output})
end

function make_generator(data, opt)
  local model = nn.Sequential()
  model:add(nn.Linear(opt.rnn_size, data.target_size))
  model:add(nn.LogSoftMax())
  local w = torch.ones(data.target_size)
  w[1] = 0
  criterion = nn.ClassNLLCriterion(w)
  criterion.sizeAverage = false
  return model, criterion
end
