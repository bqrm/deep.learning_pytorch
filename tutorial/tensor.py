# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/9

import torch


x = torch.empty(5, 3)

print('empty matrix\n', x)
print('uniform random\n', torch.rand(5, 3))
print('normalized random\n', torch.randn(5, 3))
print('tensor\n', torch.tensor([5.5, 3]))

x = x.new_ones(5, 3, dtype=torch.double)
print(x)
print('new tensor from exited\n', torch.randn_like(x, dtype=torch.float))

print('x.view(15)=', x.view(15))
print('x.view(-1, 5)=', x.view(-1, 5))

# numpy transformation
print(x.numpy())
x.add_(1)
print(x)
y = torch.from_numpy(x.numpy())
print(y)




















