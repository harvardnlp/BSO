local SynchedDropout, Parent = torch.class('nn.SynchedDropout', 'nn.Module')

function SynchedDropout:__init(p,global_noise,noise_idx,v1,inplace)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   self.inplace = inplace
   -- version 2 scales output during training instead of evaluation
   self.v2 = not v1
   if self.p >= 1 or self.p < 0 then
      error('<SynchedDropout> illegal percentage, must be 0 <= p < 1')
   end
   --self.noise = torch.Tensor()
   self.global_noise = global_noise
   self.noise_idx = noise_idx
end

function SynchedDropout:updateOutput(input)
   if self.inplace then
      self.output:set(input)
   else
      self.output:resizeAs(input):copy(input)
   end
   if self.p > 0 then
      if self.train then
         self.noise = self.global_noise[self.noise_idx]:sub(1, input:size(1))
         self.output:cmul(self.noise)
      elseif not self.v2 then
         self.output:mul(1-self.p)
      end
   end
   return self.output
end

function SynchedDropout:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
   if self.train then
      if self.p > 0 then
         self.gradInput:cmul(self.noise) -- simply mask the gradients with the noise vector
      end
   else
      if not self.v2 and self.p > 0 then
         self.gradInput:mul(1-self.p)
      end
   end
   return self.gradInput
end

function SynchedDropout:setp(p)
   self.p = p
end

function SynchedDropout:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end


function SynchedDropout:clearState()
   if self.noise then
      self.noise:set()
   end
   return Parent.clearState(self)
end
