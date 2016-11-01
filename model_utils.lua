-- from https://github.com/wojciechz/learning_to_execute/blob/master/utils/utils.lua
-- presumably not actually necessary anymore (can use :clone())
function clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function adagradStep(x, dfdx, eta, state)
   if not state.var then
      state.var  = torch.Tensor():typeAs(x):resizeAs(x):zero()
      state.std = torch.Tensor():typeAs(x):resizeAs(x)
   end

   state.var:addcmul(1, dfdx, dfdx)
   state.std:sqrt(state.var)
   x:addcdiv(-eta, dfdx, state.std:add(1e-10))
end

function zero_table(t, opt)
  for i = 1, #t do
    if opt.gpuid >= 0 and opt.gpuid2 and opt.gpuid2 >= 0 then
      if i == 1 then
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(opt.gpuid2)
      end
    end
    t[i]:zero()
  end
end

function reset_state(state, batch_l, t)
  local u = {[t] = {}}
  for i = 1, #state do
    state[i]:zero()
    table.insert(u[t], state[i][{{1, batch_l}}])
  end
  if t == 0 then
    return u
  else
    return u[t]
  end
end

function clean_layer(layer)
  if opt.gpuid >= 0 then
    layer.output = torch.CudaTensor()
    layer.gradInput = torch.CudaTensor()
  else
    layer.output = torch.DoubleTensor()
    layer.gradInput = torch.DoubleTensor()
  end
  if layer.modules then
    for i, mod in ipairs(layer.modules) do
      clean_layer(mod)
    end
  elseif torch.type(self) == "nn.gModule" then
    layer:apply(clean_layer)
  end
end

function get_layer(layer)
  if layer.name ~= nil then
    if layer.name == 'word_vecs_dec' then
      word_vecs_dec = layer
    elseif layer.name == 'word_vecs_enc' then
      word_vecs_enc = layer
    elseif layer.name == 'decoder_attn' then
      decoder_attn = layer
    end
  end
end

function idx2key(file)
  local f = io.open(file,'r')
  local t = {}
  for line in f:lines() do
    local c = {}
    for w in line:gmatch'([^%s]+)' do
      table.insert(c, w)
    end
    t[tonumber(c[2])] = c[1]
  end
  return t
end

function flip_table(u)
  local t = {}
  for key, value in pairs(u) do
    t[value] = key
  end
  return t
end
