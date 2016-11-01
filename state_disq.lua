-- stolen from penlight
local function copy_tbl (t)
  local res = {}
  for k,v in pairs(t) do
    res[k] = v
  end
  return res
end

do
  local WOStateDisqualifier = torch.class('WOStateDisqualifier')

  -- i assume source is a single sequence w/o start or end tokens.
  -- this will by default disqualify end-token
  function WOStateDisqualifier:__init(source, B, V, maskmem, idxmem)
    self.B = B
    self.hyp_states = {[1]={}}
    local word_bag = {}
    local word_type_count = 0
    -- keep track of remaining words we can use
    for t = 1, source:size(1) do
      if self.hyp_states[1][source[t]] then
        self.hyp_states[1][source[t]] = self.hyp_states[1][source[t]] + 1
      else
        self.hyp_states[1][source[t]] = 1
        word_bag[source[t]] = true
        word_type_count = word_type_count + 1
      end
    end

    -- return an initial additive mask we can always use
    self.mask = maskmem:view(B, V)
    self.mask:fill(-math.huge)

    -- zero out stuff actually in the source
    self.src_idxs = idxmem:sub(1, word_type_count)
    local i = 1
    for w, b in pairs(word_bag) do
      self.src_idxs[i] = w
      i = i + 1
    end

    self.mask[1]:indexFill(1, self.src_idxs, 0)
    self.word_bag = word_bag
  end

  -- update before next search
  function WOStateDisqualifier:updateStates(parents, preds, scores) -- rembuff, pred_inp, resval
    local new_hyps = {}
    local removal_cands = {}
    for k = 1, self.B do
      if scores[k] > -math.huge then
        new_hyps[k] = copy_tbl(self.hyp_states[parents[k]])
        new_hyps[k][preds[k]] = new_hyps[k][preds[k]] - 1
        if new_hyps[k][preds[k]] == 0 then
          new_hyps[k][preds[k]] = nil -- remove key
        end
        -- update the mask
        for w, b in pairs(self.word_bag) do
          if new_hyps[k][w] then -- still an option going fwd
            self.mask[k][w] = 0
          else
            self.mask[k][w] = -math.huge
            if removal_cands[w] then
              removal_cands[w] = removal_cands[w] + 1
            else
              removal_cands[w] = 1
            end
            if k == self.B and removal_cands[w] == self.B then -- everybody removed, so we're done
              self.word_bag[w] = nil
              self.mask:select(2, w):fill(-math.huge)
            end
          end
        end
      else -- we got -inf, which can happen if beam is longer than sentence, for instance
        new_hyps[k] = self.hyp_states[parents[k]]
        -- just to be sure, no next prediction is valid
        for w, b in pairs(self.word_bag) do
          self.mask[k][w] = -math.huge
        end
      end
    end -- end for k
    self.hyp_states = new_hyps
  end

  function WOStateDisqualifier:disqualify(scores) -- scores assumed to be B x V
    scores:add(self.mask:sub(1, scores:size(1))) -- just handles beginning when one thing on beam
  end

  function WOStateDisqualifier:allowEnd(end_token)
    self.mask:select(2,end_token):zero()
  end

end

-- stanford dependency labels
dep_labels = {['@L_PRT']=true, ['@R_PRT']=true, ['@L_PARATAXIS']=true, ['@R_IOBJ']=true,
    ['@R_POSSESSIVE']=true, ['@R_NUMBER']=true, ['@R_APPOS']=true, ['@L_PUNCT']=true,
    ['@L_CONJ']=true, ['@L_CCOMP']=true, ['@L_CC']=true, ['@L_POSS']=true, ['@R_COP']=true,
    ['@R_NN']=true, ['@L_PREP']=true, ['@R_NSUBJPASS']=true, ['@L_PREDET']=true, ['@R_TMOD']=true,
    ['@R_MWE']=true, ['@R_QUANTMOD']=true, ['@L_NSUBJ']=true, ['@R_PARATAXIS']=true, ['@R_PUNCT']=true,
    ['@R_EXPL']=true, ['@L_APPOS']=true, ['@R_ACOMP']=true, ['@R_NPADVMOD']=true, ['@L_MARK']=true,
    ['@L_PRECONJ']=true, ['@R_DOBJ']=true, ['@L_ADVCL']=true, ['@R_DISCOURSE']=true, ['@L_POBJ']=true,
    ['@L_NUMBER']=true, ['@R_CONJ']=true, ['@R_ADVMOD']=true, ['@L_DOBJ']=true, ['@R_ADVCL']=true,
    ['@R_XCOMP']=true, ['@L_AMOD']=true, ['@L_ADVMOD']=true, ['@L_POSSESSIVE']=true,
    ['@L_DISCOURSE']=true, ['@R_INFMOD']=true, ['@R_NUM']=true, ['@L_DEP']=true, ['@L_NSUBJPASS']=true,
    ['@R_PCOMP']=true, ['@L_NPADVMOD']=true, ['@L_DET']=true, ['@L_COP']=true, ['@L_PARTMOD']=true,
    ['@R_AMOD']=true, ['@R_AUXPASS']=true, ['@L_AUXPASS']=true, ['@R_PREP']=true, ['@R_PARTMOD']=true,
    ['@L_CSUBJ']=true, ['@L_EXPL']=true, ['@L_AUX']=true, ['@R_NEG']=true, ['@L_MWE']=true,
    ['@R_RCMOD']=true, ['@R_CCOMP']=true, ['@R_CC']=true, ['@R_AUX']=true, ['@L_XCOMP']=true,
    ['@R_NSUBJ']=true, ['@L_NUM']=true, ['@R_POBJ']=true, ['@L_NN']=true, ['@L_NEG']=true,
    ['@L_TMOD']=true, ['@L_CSUBJPASS']=true, ['@R_DET']=true, ['@R_POSS']=true, ['@R_DEP']=true,
    ['@L_QUANTMOD']=true, ['@L_ACOMP']=true}


function get_dep_label_idxs(word2idx_targ, cuda)
  local label_idxs = cuda and torch.CudaTensor() or torch.Tensor()
  label_idxs:resize(79)
  local ii = 1
  for w, b in pairs(dep_labels) do
    label_idxs[ii] = word2idx_targ[w]
    ii = ii + 1
  end
  assert(ii == 80)
  return label_idxs
end

do
  local LabeledSRStateDisqualifier = torch.class('LabeledSRStateDisqualifier')

  -- i assume source is a single sequence w/o start or end tokens.
  -- this will by default disqualify end-token.
  function LabeledSRStateDisqualifier:__init(source, B, V, maskmem,
      idxmem, label_idxs, idx2word_src, word2idx_targ, idx2word_targ)
    self.B = B
    self.word2idx = word2idx_targ
    self.idx2word = idx2word_targ
    self.source = idxmem:sub(1, source:size(1))
    -- copy target vocab version of source
    for i = 1, source:size(1) do
      self.source[i] = word2idx_targ[idx2word_src[source[i]]]
    end
    self.hyp_states = {}
    self.hyp_states[1] = {next_word_idx = 1, stack_size = 0}

    -- return an initial additive mask we can always use
    self.mask = maskmem:view(B, V)
    self.mask:fill(-math.huge)

    -- the only kosher initial prediction is the first word
    self.mask[1][self.source[1]] = 0
    self.lb = 1 -- earliest token any hyp is still up to
    self.ub = 1 -- latest token any hyp is still up to

    self.label_idxs = label_idxs
  end

  -- update before next search
  function LabeledSRStateDisqualifier:updateStates(parents, preds, scores) -- rembuff, pred_inp, resval
    local new_hyps = {}
    local beam_min = math.huge
    for k = 1, self.B do
      if scores[k] > -math.huge then
        new_hyps[k] = copy_tbl(self.hyp_states[parents[k]])
        if dep_labels[self.idx2word[preds[k]]] then
        --if preds[k] == self.lred_idx or preds[k] == self.rred_idx then
          new_hyps[k].stack_size = new_hyps[k].stack_size - 1
        else
          new_hyps[k].stack_size = new_hyps[k].stack_size + 1
          new_hyps[k].next_word_idx = new_hyps[k].next_word_idx + 1
          if new_hyps[k].next_word_idx > self.ub and new_hyps[k].next_word_idx <= self.source:size(1) then
            self.ub = new_hyps[k].next_word_idx
          end
        end

        -- keep track of lowest index still in play
        if new_hyps[k].next_word_idx < beam_min then
          beam_min = new_hyps[k].next_word_idx
        end

        -- update mask for next search step
        for i = self.lb, self.ub do
          self.mask[k][self.source[i]] = -math.huge
        end

        -- allow only after setting to -inf, in case there are repeats
        if new_hyps[k].next_word_idx <= self.source:size(1) then
          self.mask[k][self.source[new_hyps[k].next_word_idx]] = 0
        end

        if new_hyps[k].stack_size >= 2 then
          --self.mask[k][self.lred_idx] = 0
          --self.mask[k][self.rred_idx] = 0
          self.mask[k]:indexFill(1, self.label_idxs, 0)
        else
          --self.mask[k][self.lred_idx] = -math.huge
          --self.mask[k][self.rred_idx] = -math.huge
          self.mask[k]:indexFill(1, self.label_idxs, -math.huge)
        end
      else -- we got -inf, which can happen if beam is longer than sentence, for instance
        for i = self.lb, self.ub do
          self.mask[k][self.source[i]] = -math.huge
        end
        --self.mask[k][self.lred_idx] = -math.huge
        --self.mask[k][self.rred_idx] = -math.huge
        self.mask[k]:indexFill(1, self.label_idxs, -math.huge)
      end
    end -- end for k
    -- we may be able to move up lb
    if beam_min > self.lb then -- everything on beam is above lb so we can forget about lower indices
      self.lb = beam_min
    end
    self.hyp_states = new_hyps
  end

  function LabeledSRStateDisqualifier:disqualify(scores) -- scores assumed to be B x V
    scores:add(self.mask:sub(1, scores:size(1))) -- just handles beginning when one thing on beam
  end

  function LabeledSRStateDisqualifier:allowEnd(end_token)
    self.mask:select(2,end_token):zero()
  end

end
