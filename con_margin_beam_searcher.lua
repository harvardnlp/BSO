do
  local ConMarginBatchBeamSearcher = torch.class('ConMarginBatchBeamSearcher')

  function ConMarginBatchBeamSearcher:__init(rnn_size, state_tbl_size, cuda)
    self.rnn_size = rnn_size
    self.state_tbl_size = state_tbl_size -- basically num layers

    self.resval = cuda and torch.Tensor():cuda() or torch.Tensor()
    self.resind = cuda and torch.LongTensor():cuda() or torch.LongTensor()
    self.finalval = cuda and torch.Tensor(1):cuda() or torch.Tensor(1)
    self.finalind = cuda and torch.LongTensor(1):cuda() or torch.LongTensor(1)
    self.rembuff = cuda and torch.Tensor():cuda() or torch.LongTensor()
    self.prev_state = {}
    for j = 1, state_tbl_size do
      self.prev_state[j] = cuda and torch.Tensor():cuda() or torch.Tensor()
    end
    self.expanded_global_noise = cuda and torch.Tensor():cuda() or torch.Tensor()
  end

  function ConMarginBatchBeamSearcher:initSearch(K, batch_size, init_dec_state, target, target_l, num_layers)
    collectgarbage()
    self.K = K
    self.batch_size = batch_size
    self.resval:resize(K)
    self.resind:resize(K)
    self.rembuff:resize(K)

    -- copy state of each example
    for j = 1, self.state_tbl_size do
      self.prev_state[j]:resize(batch_size*K, self.rnn_size)
      self.prev_state[j]:zero() -- just to avoid nans
      if init_dec_state[j] then -- just doing this so can initialize w/ encoder sometimes
        for n = 1, batch_size do
          self.prev_state[j][(n-1)*K+1]:copy(init_dec_state[j][n])
        end
      end
    end

    self.pred_pfxs = {}
    self.gold_pred_holders = {}

    for n = 1, batch_size do
      self.pred_pfxs[n] = {}
      self.gold_pred_holders[n] = {}
      -- we'll use a linked list so we don't have to copy
      self.pred_pfxs[n][1] = {prev = nil, val = target[1][n]}
      -- make a gold linked_list
      self.gold_pred_holders[n][1] = {prev = nil, val = target[1][n]}
      for t = 2, target_l do
        self.gold_pred_holders[n][t] = {prev = self.gold_pred_holders[n][t-1], val = target[t][n]}
      end
    end
    self.num_layers = num_layers
    self.expanded_global_noise:resize(num_layers, batch_size*K, self.rnn_size)
  end

  -- assumes beam_dec SynchedDropouts have indices 1..num_layers
  function ConMarginBatchBeamSearcher:synchDropout(t, global_noise)
    for j = 1, self.num_layers do
      for n = 1, self.batch_size do
        torch.repeatTensor(self.expanded_global_noise[j]:sub((n-1)*self.K+1, n*self.K),
          global_noise[(t-1)*self.num_layers+j]:sub(n,n), self.K, 1)
      end
    end
  end

  -- pred_inp and ctx should be of length batch_l*K along first dim; in particular, ctx should
  -- consist of K stacked copies of the context of each example in the batch.
  -- this fills pred_inp
  function ConMarginBatchBeamSearcher:nextSearchStep(t, batch_pred_inp, batch_ctx, beam_dec, beam_scorer,
      gold_scores, target, target_w, gold_rnn_state_dec, delts, losses, global_noise, disqs)
    local K = self.K
    local resval, resind, rembuff = self.resval, self.resind, self.rembuff
    local finalval, finalind = self.finalval, self.finalind
    self:synchDropout(t, global_noise)
    -- pred_inp should be what was predicted at the last step
    local outs = beam_dec:forward({batch_pred_inp, batch_ctx, unpack(self.prev_state)})
    local all_scores = beam_scorer:forward(outs[#outs]) -- should be (batch_l*K) x V matrix
    local V = all_scores:size(2)
    local mistaken_preds = {}
    for n = 1, self.batch_size do
      delts[n] = 0
      losses[n] = 0
      if t <= target_w[n]-1 then -- only do things if t <= length (incl end token) - 2
        local beam_size = #self.pred_pfxs[n]
        local nstart = (n-1)*K+1
        local nend = n*K
        local scores = all_scores:sub(nstart, nstart+beam_size-1)--:view(-1) -- scores for this example
        disqs[n]:disqualify(scores)
        scores = scores:view(-1)
        -- take top K
        torch.topk(resval, resind, scores, K, 1, true)
        -- see if we violated margin
        torch.min(finalval, finalind, resval, 1) -- resind[finalind[1]] is idx of K'th highest predicted word
        -- checking that true score at least 1 higher than K'th
        losses[n] = math.max(0, 1 - gold_scores[n][target[t+1][n]] + finalval[1])
        -- losses[n] = math.max(0, - gold_scores[n][target[t+1][n]] + finalval[1])
        if losses[n] > 0 then
          local parent_idx = math.ceil(resind[finalind[1]]/V)
          local pred_word = ((resind[finalind[1]]-1)%V) + 1
          mistaken_preds[n] = {prev = self.pred_pfxs[n][parent_idx], val = pred_word}
          delts[n] = 1 -- can change.....
        else
          -- put predicted next words in pred_inp
          rembuff:add(resind, -1) -- set rembuff = resind - 1
          rembuff:div(V)
          --if rembuff.floor then
          rembuff:floor()    -- note rembuff contains floor((resind-1)/V)
          --end
          local pred_inp = batch_pred_inp:sub(nstart, nend)
          pred_inp:add(resind, -V, rembuff) -- pred_inp = (resind-1)%V + 1 = resind - V*floor(resind-1/V)

          -- get parents of top K things
          rembuff:add(1)

          -- update previous state
          for j = 1, #self.prev_state do
            self.prev_state[j]:sub(nstart, nend):index(outs[j]:sub(nstart, nend), 1, rembuff)
          end

          -- update predictions on beam
          local newpp = {}
          for k = 1, K do
            local parent_idx = rembuff[k]
            newpp[k] = {prev = self.pred_pfxs[n][parent_idx], val = pred_inp[k]}
          end
          self.pred_pfxs[n] = newpp

          -- update disqualification stuff
          disqs[n]:updateStates(rembuff, pred_inp, resval)
        end -- end if loss_t > 0
      end -- end if t <= target_w[n]-1 then
    end -- end for n = 1
    return mistaken_preds
  end


  -- assumes beam_dec SynchedDropouts have indices 1..num_layers
  function ConMarginBatchBeamSearcher:synchFinalDropout(target_w, global_noise)
    for j = 1, self.num_layers do
      for n = 1, self.batch_size do
        local penult = target_w[n]
        torch.repeatTensor(self.expanded_global_noise[j]:sub((n-1)*self.K+1, n*self.K),
          global_noise[(penult-1)*self.num_layers+j]:sub(n,n), self.K, 1)
      end
    end
  end

  -- this is for t = actual_l - 1
  function ConMarginBatchBeamSearcher:finalStep(batch_pred_inp, batch_ctx, beam_dec, beam_scorer,
      gold_scorers, target, target_w, delts, losses, global_noise, disqs)
    local K = self.K
    local resval, resind = self.resval:sub(1,2), self.resind:sub(1,2)
    local finalval, finalind = self.finalval, self.finalind
    -- pred_inp should be what was predicted at the last step
    -- too annoying to do different lenght final dropout
    self:synchFinalDropout(target_w, global_noise)
    local outs = beam_dec:forward({batch_pred_inp, batch_ctx, unpack(self.prev_state)})
    local all_scores = beam_scorer:forward(outs[#outs]) -- should be (batch_l*K) x V matrix
    local V = all_scores:size(2)
    local mistaken_preds = {}
    for n = 1, self.batch_size do
      delts[n] = 0
      losses[n] = 0
      local penult = target_w[n]
      local beam_size = #self.pred_pfxs[n]
      local nstart = (n-1)*K+1
      local nend = n*K
      local scores = all_scores:sub(nstart, nstart+beam_size-1)--:view(-1) -- scores for this example
      disqs[n]:disqualify(scores)
      scores = scores:view(-1)
      -- take top K
      torch.topk(resval, resind, scores, 1, 1, true)
      -- see if we violated margin
      torch.max(finalval, finalind, resval, 1) -- resind[finalind[1]] should now hold max thing
      local offending_idx
      local gold_score_t = gold_scorers[penult].output[n][target[penult+1][n]]
      if finalval[1] >= gold_score_t then -- assume highest is offender
        offending_idx = resind[finalind[1]]
      else -- 2nd highest is potential offender
        offending_idx = finalind == 1 and resind[2] or resind[1]
      end
      -- checking that true score at least 1 higher than K'th
      losses[n] = math.max(0, 1 - gold_score_t + scores[offending_idx])
      if losses[n] > 0 then
        local parent_idx = math.ceil(resind[finalind[1]]/V)
        local pred_word = ((resind[finalind[1]]-1)%V) + 1
        mistaken_preds[n] = {prev = self.pred_pfxs[n][parent_idx], val = pred_word}
        delts[n] = 1 -- can change.....
      end
    end
    return mistaken_preds
  end

  function ConMarginBatchBeamSearcher:resetBeam(n, gold_rnn_state_dec, inps, target, t)
    for j = 1, #self.prev_state do
      self.prev_state[j][(n-1)*self.K+1]:copy(gold_rnn_state_dec[t][j][n])
    end
    inps[(n-1)*self.K+1] = target[t+1][n]
    self.pred_pfxs[n] = {}
    self.pred_pfxs[n][1] = self.gold_pred_holders[n][t+1] -- should be correct thru t's prediction
  end

   -- search up to some maximum length and keep track of highest scoring stuff
   -- assumes search already init'd
   -- assumes disqualifiers already init'd
  function ConMarginBatchBeamSearcher:evalSearch(batch_pred_inp, batch_ctx, beam_dec, beam_scorer,
      max_length, end_token, disqs)
    local K = self.K
    local resval, resind, rembuff = self.resval, self.resind, self.rembuff
    local finalval, finalind = self.finalval, self.finalind
    local all_preds, all_done, all_best = {}, {}, {}

    for n = 1, self.batch_size do
      all_preds[n] = {}
      all_best[n] = {pred = nil, score = -math.huge}
      hyp_state[n] = {}
      disqs[n]:allowEnd(end_token) -- assume end is always allowed
    end

    for t = 1, max_length do
      local outs = beam_dec:forward({batch_pred_inp, batch_ctx, unpack(self.prev_state)})
      local all_scores = beam_scorer:forward(outs[#outs]) -- should be (batch_l*K) x V matrix
      local V = all_scores:size(2)
      for n = 1, self.batch_size do
        if not all_done[n] then
          local beam_size = #self.pred_pfxs[n]
          local nstart = (n-1)*K+1
          local nend = n*K
          local scores = all_scores:sub(nstart, nstart+beam_size-1)--:view(-1) -- scores for this example
          -- disqualify shit
          disqs[n]:disqualify(scores)
          scores = scores:view(-1)
          -- take top K
          torch.topk(resval, resind, scores, K, 1, true)
          local pred_inp = batch_pred_inp:sub(nstart, nend)
          -- put predicted next words in pred_inp
          rembuff:add(resind, -1) -- set rembuff = resind - 1
          rembuff:div(V)
          --if rembuff.floor then
          rembuff:floor()    -- note rembuff contains floor((resind-1)/V)
          --end
          pred_inp:add(resind, -V, rembuff) -- pred_inp = (resind-1)%V + 1 = resind - V*floor(resind-1/V)

          -- get parents of top K things
          rembuff:add(1)

          local done, notdone = {}, {}
          for k = 1, K do
            if resval[k] > all_best[n].score then
              local pred = {prev = self.pred_pfxs[n][rembuff[k]], val = pred_inp[k]}
              all_best[n] = {pred = pred, score = resval[k]}
            end
            if pred_inp[k] == end_token then
              local pred = {prev = self.pred_pfxs[n][rembuff[k]], val = end_token}
              table.insert(all_preds[n], {pred = pred, score = resval[k]})
              table.insert(done, k)
            else
              table.insert(notdone, k)
            end
          end

          if #notdone == 0 then -- we're done
            all_done[n] = true
            break
          else -- just replace stuff with first notdone thing; will come out in the wash
            for j, k in ipairs(done) do
              pred_inp[k] = pred_inp[notdone[1]]
              rembuff[k] = rembuff[notdone[1]]
            end
          end

          -- update previous state
          for j = 1, #outs do
            self.prev_state[j]:sub(nstart, nend):index(outs[j]:sub(nstart, nend), 1, rembuff)
          end

          -- update predictions on beam
          local newpp = {}
          for k = 1, K do
            newpp[k] = {prev = self.pred_pfxs[n][rembuff[k]], val = pred_inp[k]}
          end
          self.pred_pfxs[n] = newpp

          -- update disqualification stuff
          disqs[n]:updateStates(rembuff, pred_inp)

        end -- end if not all_done[n]
      end -- end for n = 1
    end -- end for t = 1

    -- finally return the ended prediction w/ highest score (if any)
    local final_preds = {}
    for n = 1, self.batch_size do
      if #all_preds[n] == 0 then
        print("laaaaame, no predictions with an end token")
        final_preds[n] = all_best[n].pred
      else
        local best_score = all_preds[n][1].score
        final_preds[n] = all_preds[n][1].pred
        for i = 2, #all_preds[n] do
          if all_preds[n][i].score > best_score then
            best_score = all_preds[n][i].score
            final_preds[n] = all_preds[n][i].pred
          end
        end
      end
    end
    return final_preds
  end

  -- for when we know how long the target sequence is (like for pos tagging)
  -- max length should presumably be 1 before end token
  function ConMarginBatchBeamSearcher:evalFixedLengthSearch(batch_pred_inp, batch_ctx, beam_dec, beam_scorer,
      max_length, disqs)
    local K = self.K
    local resval, resind, rembuff = self.resval, self.resind, self.rembuff
    local finalval, finalind = self.finalval, self.finalind
    local all_best, all_best_scores = {}, {}
    for n = 1, self.batch_size do
      all_best[n] = nil
      all_best_scores[n] = -math.huge
    end

    for t = 1, max_length do -- max_length should be 1 less than the fixed length
      local outs = beam_dec:forward({batch_pred_inp, batch_ctx, unpack(self.prev_state)})
      local all_scores = beam_scorer:forward(outs[#outs]) -- should be (batch_l*K) x V matrix
      local V = all_scores:size(2)
      for n = 1, self.batch_size do
        local beam_size = #self.pred_pfxs[n]
        local nstart = (n-1)*K+1
        local nend = n*K
        local scores = all_scores:sub(nstart, nstart+beam_size-1)--:view(-1) -- scores for this example
        -- disqualify bad things
        disqs[n]:disqualify(scores)
        scores = scores:view(-1)
        -- take top K
        torch.topk(resval, resind, scores, K, 1, true)
        local pred_inp = batch_pred_inp:sub(nstart, nend)
        -- put predicted next words in pred_inp
        rembuff:add(resind, -1) -- set rembuff = resind - 1
        rembuff:div(V)
        --if rembuff.floor then
        rembuff:floor()    -- note rembuff contains floor((resind-1)/V)
        --end
        pred_inp:add(resind, -V, rembuff) -- pred_inp = (resind-1)%V + 1 = resind - V*floor(resind-1/V)

        -- get parents of top K things
        rembuff:add(1)

        -- update previous state
        for j = 1, #outs do
          self.prev_state[j]:sub(nstart, nend):index(outs[j]:sub(nstart, nend), 1, rembuff)
        end

        -- update predictions on beam
        local newpp = {}
        for k = 1, K do
          local parent_idx = rembuff[k]
          newpp[k] = {prev = self.pred_pfxs[n][parent_idx], val = pred_inp[k]}
          if t == max_length and resval[k] > all_best_scores[n] then
            all_best_scores[n] = resval[k]
            all_best[n] = newpp[k]
          end
        end
        self.pred_pfxs[n] = newpp

        -- update disqualification stuff
        disqs[n]:updateStates(rembuff, pred_inp, resval)
      end -- end for n = 1
    end -- end for t = 1
    return all_best
  end

end
