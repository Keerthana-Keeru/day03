import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# Transforms
transforms=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#load the data
train_data=datasets.CIFAR100(root='./dir',train=True,download=True,transform=transforms)
test_data=datasets.CIFAR100(root='./dir',train=False,download=True,transform=transforms)
#data loader
train_loader=DataLoader(train_data,batch_size=128,shuffle=True)
test_loader=DataLoader(test_data,batch_size=128,shuffle=False)
#Architecture
class mlp(nn.Module):
    def __init__(self):
        super(mlp,self).__init__()
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(32*32*3,1024)
        self.fc2=nn.Linear(1024,512)
        self.fc3=nn.Linear(512,256)
        self.fc4=nn.Linear(256,128)
        self.fc5=nn.Linear(128,100)
    def forward(self,x):
        x=self.flatten(x)
        #x=torch.relu(self.fc1(x))  or
        x=self.fc1(x)
        x=torch.relu(x)
        x=torch.relu(self.fc2(x))
        x=torch.relu(self.fc3(x))
        x=torch.relu(self.fc4(x))
        x=self.fc5(x)
        return x
model=mlp() # object for class
criterian =nn.CrossEntropyLoss()#loss
optimizer=optim.Adam(model.parameters(),lr=0.001)

# train ,# send to architecture,prediction,find losstrain,loss,optimizer(send the data,loss,back propogation,optmizer)
epoch_loss=0.0
for epoch in range(3):
    
    
    for image,label in train_loader:
        output=model(image)#send the data
        loss=criterian(output,label)# loss
        loss.backward()#back propogation
        optimizer.step() #step of operation done
        optimizer.zero_grad # initialize gradient to zero for next batch
        epoch_loss+=loss.item()
    print(f"epoch_loss:{epoch_loss}")
model.eval()
total=0
correct=0
with torch.no_grad():
    for images,label in test_loader:
        output=model(images)
        _,predicted=torch.max(output,1) # 1 refer to it must consider 1 row/column
        total+=label.size(0)
        correct+=(predicted==label).sum().item()
print(f"accuracy:{(correct/total)*100}")


# optimizers and depth, pytorch