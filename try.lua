function strp1s(x) -- x is feature indicies vector (0, 1)
   local D = x:size(1)
   local count = 0
   for i = 1, D do
      if x[i] ~= 1 then
	 count = count + 1
      end
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

function feat_on(x, f)
   local D = x:size(1)
   for i = 1, D do
      if x[i] == f then
	 return true
      end
   end
   return false
end

function main()
   local x = torch.LongTensor({1, 2, 54, 1, 1, 1, 2,1, 3, 1, 2, 3,1, 3})
   print (feat_on(x, 6))
end



main()
