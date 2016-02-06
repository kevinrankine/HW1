-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-alpha', 'alpha parameter for NB')
cmd:option('-epochs', 'number of sweeps over the training data')
cmd:option('-learn', 'the learning rate')
cmd:option('-lambda', 'regularization term')
cmd:option('-batchsize', 'size of the minibatches to use in training')

function main()
   opt = cmd:parse(arg)
   
   local datafile = opt.datafile
   local classifier = opt.classifier
   
   if (classifier == 'nb') then
      local alpha = opt.alpha or 1
      NB(datafile, alpha)
   elseif (classifier == 'log') then
      logistic(datafile, tonumber(opt.epochs), tonumber(opt.learn), tonumber(opt.lambda), tonumber(opt.batchsize))
   else
      -- hinge loss svm
   end
end

function logistic(datafile, epochs, learn, lambda, batchsize)
   local f = hdf5.open(datafile, 'r')
   local nclasses = f:read('nclasses'):all():long()[1]
   local nfeatures = f:read('nfeatures'):all():long()[1]
   
   local train_input = f:read('train_input'):all()
   local train_output = f:read('train_output'):all()
   local valid_input = f:read('valid_input'):all()
   local valid_output = f:read('valid_output'):all()

   W = torch.DoubleTensor(nclasses, nfeatures):zero()
   b = torch.DoubleTensor(nclasses):zero()

   logistic_train(nfeatures, nclasses, W, b, train_input, train_output, epochs, learn, lambda, batchsize)

   logistic_validate(nfeatures, nclasses, W, b, valid_input, valid_output)

end

function softmax(nfeatures, nclasses, W, b, x, y)
   -- x is not a sparse vector, it is the vector of feature indicies f_k with 1 as the padding dummy feature
   -- returns class prediction probabilities and the gradient
   
   local z = W:index(2, x):sum(2):add(b) -- Wx + b
   local partition = z:exp():sum(1)
   local y = z:div(partition[1][1])

   return y
end

function logexpsum(z)
   local z = z
   M = z:max()
   return (torch.log(torch.sum(torch.exp(torch.add(z, -M))))) + M
end



function strp1s(x) -- x is feature indicies vector (0, 1)
   -- removes the dummy feature indicies and returns a new vector
   local D = x:size(1)
   local count = 0
   for i = 1, D do
      if x[i] ~= 1 then
	 count = count + 1
      end
   end

   if count == 0 then
      return nil
   end
   
   new_x = torch.LongTensor(count)
   local j = 1
   for i = 1, count do
      while x[j] == 1 do
	 j = j + 1
      end
      new_x[i] = x[j]
      j = j + 1
   end
   return new_x
end

function L_ce(nfeatures, nclasses, W, b, x, y)
   -- cross-entropy loss for a single example
   -- in this case y would be the class, not one hot coded
   
   if (x == nil) then
      return nil, nil
   end
   local z = W:index(2, x):sum(2):add(b) -- 
   local loss = -(z[y] - logexpsum(z)) -- -log(p(c|x))
   local y_hat = torch.exp(z:add(-logexpsum(z)))

   return loss, y_hat
end


function dL(nfeatures, nclasses, W, b, x, y)
   -- finite difference gradients dL/dW and dL/db (for checking)
   
   local epsilon_W = torch.DoubleTensor(nclasses, nfeatures):zero()
   local epsilon_b = torch.DoubleTensor(nclasses):zero()

   local grad_W = torch.DoubleTensor(nclasses, nfeatures):zero()
   local grad_b = torch.DoubleTensor(nclasses):zero()

   for class = 1, nclasses do
      for j = 1, x:size(1) do
	 epsilon_W[class][x[j]] = 1e-3
	 local derivative = (L_ce(nfeatures, nclasses, W + epsilon_W, b, x, y)[1] -
				L_ce(nfeatures, nclasses, W - epsilon_W, b, x, y)[1]) / (2 * 1e-3)
	 grad_W[class][x[j]] = derivative

	 epsilon_W:zero()
      end
      epsilon_b[class] = 1e-3
      
      local derivative = (L_ce(nfeatures, nclasses, W, b + epsilon_b, x, y)[1] -
			     L_ce(nfeatures, nclasses, W, b - epsilon_b, x, y)[1]) / (2 * 1e-3)

      grad_b[class] = derivative
      epsilon_b:zero()
   end

   return grad_W, grad_b
end

function log_grad(nfeatures, nclasses, x, y_hat, y, grad_W, grad_b, batch_size)
   -- ***UPDATES*** the gradient passed in by a factor of 1/batch_size
   -- you must zero the gradient at the start
   for i = 1, nclasses do
      if y == i then
	 local vals = torch.DoubleTensor(x:size(1)):fill(-(1 - y_hat[i][1]) / batch_size)
	 grad_W[i]:indexAdd(1, x, vals)
	 grad_b[i] = grad_b[i] -((1 - y_hat[i][1]) / batch_size)
      else
	 local vals = torch.DoubleTensor(x:size(1)):fill(y_hat[i][1]/ batch_size)
	 
	 grad_W[i]:indexAdd(1, x, vals)
	 grad_b[i] = grad_b[i] + (y_hat[i][1] / batch_size)
      end
   end
end

function logistic_train(nfeatures, nclasses, W, b, X, Y, epochs, rate, lambda, batchsize)
   local epochs = epochs or 1
   local rate = rate or 1
   local batch_size = batchsize or 100
   
   local N = X:size(1)
   local n = N
   local D = X:size(2)
   local grad_W = torch.DoubleTensor(nclasses, nfeatures):zero()
   local grad_b = torch.DoubleTensor(nclasses):zero()

   local shuffle_indicies = torch.randperm(N):type('torch.LongTensor'):narrow(1, 1, n)
   
   X = X:index(1, shuffle_indicies)
   Y = Y:index(1, shuffle_indicies)
   
   local i = 1
   local total_loss = 0.0
   local pct_correct = 0.0
   for epoch = 1, epochs do
      while (i < n) do
	 for j = 1, batch_size do
	    local x = strp1s(X[i]:type('torch.LongTensor'))
	    local y = Y[i]
	    local loss, y_hat = L_ce(nfeatures,
				     nclasses,
				     W,
				     b,
				     x,
				     y)
	    if (loss ~= nil) then
	       total_loss = total_loss  + loss[1]
	       -- should add a bit of a gradient
	       log_grad(nfeatures, nclasses, x, y_hat, y, grad_W, grad_b, batch_size)
	       
	       --local dW, db = dL(nfeatures, nclasses, W, b, x, y)
	       --print (torch.sum(torch.abs( dW - grad_W)))
	       --print (torch.sum(torch.abs( db - grad_b)))
	       if (y_hat[y][1] == torch.max(y_hat) ) then
		  pct_correct = pct_correct + 1 / n
	       end
	    end
	    
	    i = i + 1

	    if (i > N) then
	       break
	    end
	 end
	 W:add(grad_W:mul(-rate))
	 b:add(grad_b:mul(-rate))

	 grad_W:zero()
	 grad_b:zero()
      end
      -- print ("Training Loss: ", total_loss)
      print ("Percent correct on training: ", 100 * pct_correct)
      total_loss = 0
      pct_correct = 0
      i = 1
   end
   return W, b
   -- W:index(2, x):add(grad_W:index(2, x):mul(-rate / batch_size))
   -- b:add(grad_b:mul(-rate / batch_size))
end

function logistic_validate(nfeatures, nclasses, W, b, X, Y)
   local N = X:size(1)
   local pct_correct = 0.0
   for i = 1, N do
      local x = strp1s(X[i]:type('torch.LongTensor'))
      local y = Y[i]
      local loss, y_hat = L_ce(nfeatures,
			       nclasses,
			       W,
			       b,
			       x,
			       y)

      if (y_hat[y][1] == torch.max(y_hat) ) then
	 pct_correct = pct_correct + 1 / N
      end
   end
   print ("Percent correct on validation: ", pct_correct * 100)
end

function NB(datafile, alpha) 
   -- Parse input params

   
   print ("-- Reading input...", "\n")
   
   local f = hdf5.open(datafile, 'r')
   local alpha = 1 or alpha
   local nclasses = f:read('nclasses'):all():long()[1]
   local nfeatures = f:read('nfeatures'):all():long()[1]
   
   local train_input = f:read('train_input'):all()
   local train_output = f:read('train_output'):all()
   local valid_input = f:read('valid_input'):all()
   local valid_output = f:read('valid_output'):all()

   print ("-- Training", "\n")
   local priors, likelihood = NB_train(nfeatures, nclasses, train_input, train_output, alpha)
   
   print ("-- Validating", "\n")
   local nb_output = NB_predict(nclasses, nfeatures, valid_input, priors, likelihood)

   percent_correct = 0
   
   for i = 1, valid_output:size(1) do
      if valid_output[i] == nb_output[i] then
	 percent_correct = percent_correct + 1 / (valid_output:size(1))
      end
   end

   print (percent_correct)
end

function NB_train(nfeatures, nclasses, train_input, train_output, alpha)
   -- define parameters related
   
   local N = train_input:size(1) -- number of examples
   local D = train_input:size(2) -- maximum sentence length
   local X = train_input -- input matrix (row i is feature indexes in xi
   local Y = train_output -- output matrix (row i is class of xi)
   
   local priors = torch.FloatTensor(nclasses):zero()
   local F = torch.LongTensor(nfeatures, nclasses):zero():add(alpha)
   local F_class = torch.LongTensor(nclasses):zero()
   local likelihood = torch.FloatTensor(nfeatures, nclasses):zero()
   
   -- TODO: populate priors
   
   for i = 1, N do
      priors[Y[i]] = priors[Y[i]] + 1 / N
   end

   assert(torch.abs(priors:sum() - 1) < 1e-3)
   -- TODO: construct F matrix and F_class (cumulative num features in a class)

   for row = 1, N do
      for col = 1, D do
	 local feat = X[row][col]
	 if feat ~= 1 then
	    local class = Y[row]
	    F[feat][class] = F[feat][class] + 1
	 end
      end
   end

   for c = 1, nclasses do
      for f = 2, nfeatures do
	 F_class[c] = F_class[c] + F[f][c]
      end
   end

   -- TODO: populate likehihood

   for f = 2, nfeatures do
      for c = 1, nclasses do
	 likelihood[f][c] = F[f][c] / F_class[c]
      end
   end

   return priors, likelihood
end

function NB_predict(nclasses, nfeatures, X, priors, likelihood)
   N = X:size(1)
   D = X:size(2)
   Y = torch.FloatTensor(N)
   
   for i = 1, N do
      local mprob = 0
      local choice = 1
      for class = 1, nclasses do
	 local cprob = priors[class]
	 for j = 1, D do
	    feat = X[i][j]
	    if feat ~= 1 then
	       cprob = cprob * likelihood[feat][class]
	    end
	 end
	 if cprob > mprob then
	    mprob = cprob
	    choice = class
	 end
      end
      Y[i] = choice
   end

   return Y
end

main()
