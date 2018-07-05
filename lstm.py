import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim

import numpy as np


class net(nn.Module):
	def __init__(self,input_size,batch_size,hidden_size,n_layers=1,n_directions=1):
		super(net,self).__init__()

		#参数初始化
		self.n_layers = n_layers
		self.n_directions = n_directions
		self.input_size = input_size
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		#self.sentence_size = sentence_size

		#定义lstm部分
		self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.n_layers,batch_first=True)

		#线性部分
		self.fc1 = nn.Linear(hidden_size,128)
		self.fc2 = nn.Linear(128,2)

	def forward(self,x):
		#初始化lstm中的各个参数(感觉不包括各个gate的w和b)
		h0 = torch.zeros(self.n_layers*self.n_directions,self.batch_size,self.hidden_size).cuda()
		c0 = torch.zeros(self.n_layers*self.n_directions,self.batch_size,self.hidden_size).cuda()

		#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		#h0,c0 = h0.to(device),c0.to(device)

		x = x.view(self.batch_size,1,-1).cuda()
		#x = x.to(device)

		x,hn = self.lstm(x,(h0,c0))

		x = x.view(-1,256)
		#x = x.to(device)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		return x

#读取.npy文件
#把.npy文件得到的向量扔到DataLoader里进行预处理
#输出DataLoader
def get_data(filename,batch_size):
	data_array = np.load(filename)
	data_tensor = torch.from_numpy(data_array)
	data_loader = torch.utils.data.DataLoader(data_tensor,batch_size=batch_size,\
											  shuffle=False,num_workers=8)
	return data_loader

#定义损失函数
#定义optimizer
#forward计算output
#计算损失以及相应的偏导数(梯度) backward
#用optimizer进行参数更新 step
def train(net,input_loader,label_loader,n_epochs=10):
	i = 0
	for epoch in range(n_epochs):
		total_loss = 0.0
		for data in zip(input_loader,label_loader):
			inputs,labels = data[0],data[1]
			inputs = inputs.float()
			#print(inputs,labels)
			#采用GPU进行运算
			#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
			#inputs,labels = inputs.to(device),labels.to(device)
			inputs_cuda,labels_cuda = inputs.cuda(),labels.cuda()
			#print(type(inputs[0][0][0].item()),type(labels[0].item()))

			#定义优化器
			optimizer = optim.SGD(net.parameters(),lr=0.004)
			optimizer.zero_grad()

			#定义损失函数
			cri = nn.CrossEntropyLoss()

			#forward计算损失和梯度
			outputs = net(inputs_cuda)
			loss = cri(outputs,labels_cuda)
			loss.backward(retain_graph=True)

			#bp优化
			optimizer.step()

			total_loss += loss
			#显示进度
			if i % 4000 == 3999:
				print('[epoch:%d,trained_data:%5d] average loss is : %.3f ' %(epoch+1,i+1,total_loss/4000))
				total_loss = 0.0
			i += 1
	print('--------------Finished--------------')


#做一个简单的测试，不做cv
#拿出labeled数据的前k个测试一下准确性
def test(net,input_filename,label_filename,batch_size,k=10000):
	input_loader = get_data(input_filename,batch_size)
	label_loader = get_data(label_filename,batch_size)
	i,n_right = 0,0
	for data in zip(input_loader,label_loader):
		if i >= k :
			print('accuracy of test_set k=',k,'is',100*n_right/k,'%')
			break
		inputs,labels = data
		inputs = inputs.float()

		out = net(inputs)
		labels = labels.cuda()
		out = F.sigmoid(out)
		outputs = torch.max(out,1)[1]
		mid = outputs==labels
		n_right += torch.sum(mid).item()

		i += batch_size

'''
	#载入保存好的npy文件
	input_numpy = np.load(input_filename)[:k]
	label_numpy = np.load(label_filename)[:k]

	#numpy→tensor转换，input从double_float转到float
	inputs = torch.from_numpy(input_numpy)
	inputs = inputs.float()
	labels = torch.from_numpy(label_numpy)

	#使用train好的模型进行预测，得出准确率
	out = net(inputs)
	labels = labels.cuda()   #因为放在gpu里跑出来的模型，所以out是tensor.cuda.float,因此labels也要保持相同type
	out = F.sigmoid(out)
	outputs = torch.max(out,1)[1]
	mid = outputs==labels
	n_right = torch.sum(mid)
'''

	




def main():
	label_loader = get_data('labels_200000.npy',2)
	input_loader = get_data('train_50dim.npy',2)
	ls = net(50,2,256)
	ls = ls.cuda()
	train(ls,input_loader,label_loader)
	return ls

