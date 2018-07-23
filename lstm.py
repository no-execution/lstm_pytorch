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

		#dropout层
		self.dropout = nn.Dropout(p=0.5)

		#self.sentence_size = sentence_size
		self.embedding = nn.Embedding(125084+1,self.input_size,padding_idx=0)

		#定义lstm部分
		self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.n_layers,batch_first=True)

		#线性部分
		self.fc1 = nn.Linear(hidden_size,64)
		self.fc2 = nn.Linear(64,1)

	def forward(self,x):
		#对采用数字表示的sentences进行embedding
		x = self.embedding(x)

		#初始化lstm中的各个参数(感觉不包括各个gate的w和b)
		h0 = (torch.ones(self.n_layers*self.n_directions,self.batch_size,self.hidden_size)*0.5).cuda()
		c0 = (torch.ones(self.n_layers*self.n_directions,self.batch_size,self.hidden_size)*0.5).cuda()

		#每个batch里面有120个tensor
		x = x.view(self.batch_size,-1,self.input_size)

		x,_ = self.lstm(x,(h0,c0))
		
		x = x[:,-1,:]
		return x
		'''
		x = self.dropout(x)

		#x = F.relu(self.fc1(x))
		#x = self.fc2(x)
		x = F.relu(self.fc1(x))
		#x = self.dropout(x)
		x = F.sigmoid(self.fc2(x))

		return x
        '''
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
			inputs,labels = data
			inputs = inputs.long()
			labels = labels.float()
			#print(inputs,labels)
			#采用GPU进行运算
			#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
			#inputs,labels = inputs.to(device),labels.to(device)
			inputs,labels = inputs.cuda(),labels.cuda()
			#print(type(inputs[0][0][0].item()),type(labels[0].item()))

			#定义优化器
			optimizer = optim.SGD(net.parameters(),lr=0.004)
			optimizer.zero_grad()

			#定义损失函数
			#cri = nn.CrossEntropyLoss()
			cri = nn.BCELoss()

			#forward计算损失和梯度
			outputs = net(inputs)
			#print(outputs.size(),outputs,labels_cuda.size())
			#loss = cri(outputs,labels)
			loss = cri(outputs.view(net.batch_size),labels)
			loss.backward(retain_graph=True)

			#bp优化
			optimizer.step()

			total_loss += loss
			#显示进度
			if i % 2000 == 1999:
				print('[epoch:%d,trained_data:%5d] average loss is : %.3f ' %(epoch+1,i+1,total_loss/2000))
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
		inputs = inputs.long()
		inputs = inputs.cuda()

		out = net(inputs)
		labels = labels.cuda()
		#out = F.sigmoid(out)
		#outputs = torch.max(out,1)[1]  #对于输出为2维
		outputs = torch.ge(out,0.51)
		outputs = outputs.long()
		outputs = outputs.view(batch_size)
		#print(outputs,outputs.size(),labels,labels.size())
		mid = outputs==labels
		n_right += torch.sum(mid).item()

		i += batch_size


def main():
	label_loader = get_data('labels_200000.npy',4)
	input_loader = get_data('train_pad_20dim.npy',4)
	ls = net(300,4,128)
	ls = ls.cuda()
	train(ls,input_loader,label_loader)
	return ls


def predict(net,input_filename,label_filename,batch_size,k=2000000):
	input_loader = get_data(input_filename,batch_size)
	label_loader = get_data(label_filename,batch_size)
	i,n_right = 0,0
	res = []
	for data in zip(input_loader,label_loader):
		if i >= k :
			print('accuracy of test_set k=',k,'is',100*n_right/k,'%')
			break
		inputs,labels = data
		inputs = inputs.long()
		inputs = inputs.cuda()

		out = net(inputs)
		#labels = labels.cuda()
		#out = F.sigmoid(out)
		#outputs = torch.max(out,1)[1]  #对于输出为2维
		outputs = torch.ge(out,0.5)
		outputs = outputs.long()
		outputs = outputs.view(batch_size)
		res.extend([x.item() for x in outputs])
		#print(outputs,outputs.size(),labels,labels.size())
		
		#n_right += torch.sum(mid).item()

		i += batch_size
		if i % 2000 == 0:
			print('doing',i)
	return res


#该函数的作用是，拿到lstm层的输出，用以进行semi-supervised learning或者拟合等操作
def get_res_of_lstm(net,input_filename,batch_size,k=2000000):
	input_loader = get_data(input_filename,batch_size)
	for i,data in enumerate(input_loader):
		inputs = data
		inputs = inputs.long()
		inputs = inputs.cuda()

		out = net(inputs)
		if i == 0:
			res = out.cpu().detach().numpy()
			continue
		res = np.concatenate((res,out.cpu().detach().numpy()),axis=0)
		if i % 1000 == 999:
			print('finished',i+1)
	return res


#net.load_state_dict(torch.load('net_1_out_300_4_128_20dim_10epoch.pkl'))
#test