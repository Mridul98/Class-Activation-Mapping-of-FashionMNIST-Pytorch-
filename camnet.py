from torchvision import datasets , transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim



transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
testset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 600, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle = True)

class camnet(nn.Module):
    def __init__(self):
        super(camnet,self).__init__()
        
        self.convlayers = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=(3,3)),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True), #26x26
            nn.MaxPool2d(kernel_size=(2,2),stride=2), #13x13
            nn.Conv2d(6,16,kernel_size=(2,2)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),#12x12
            nn.MaxPool2d(kernel_size=(2,2),stride=2),#6x6
            nn.Conv2d(16,120,kernel_size=(2,2)),
            nn.BatchNorm2d(120),
            nn.ReLU(inplace=True),#5x5
            nn.Conv2d(120,240,kernel_size=(2,2)), #4x4
            nn.BatchNorm2d(240))
        
        self.last_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(4,4))) # pooling average value from each channel

        self.gap_layer = nn.Sequential(
            nn.Linear(240,10),
            nn.Softmax(dim=-1))
    
    
    def forward(self,x):
        output = self.convlayers(x)
        output = self.last_layer(output)
        output = torch.squeeze(output)
        output = self.gap_layer(output)
        
        return output

loss_per_epoch = []
def train(epoch):
    cam.train()
    
    for i in range(epoch):
        running_loss = 0
        avg_loss = 0
        for j , batch in enumerate(trainloader):
            
            
            image,label = batch
            
            image = image.cuda()
            label = label.cuda()
            
            
            output =cam(image)
            
            cam_optimizer.zero_grad()
            loss = cam_criterion(output,label)
            
            
            
            loss.backward()
            
            cam_optimizer.step()
            
            running_loss += loss.detach()
            
            avg_loss = running_loss/100
        
        print('average loss '+str(avg_loss.item())+' on '+str(i)+' epoch') 
        loss_per_epoch.append(avg_loss.item())

def create_cam(image, state_dict , index , returned_shape):
    
    tensor = 0  #.................240x2x2
    gap_tensor = 0 #..............10x240
    
    for k , v in state_dict.items():
    
        if k == 'convlayers.11.weight':
            tensor = v 
            tensor = torch.sum(tensor,1)

            
            #print ('convlayers matrix',tensor.shape)
        if k == 'gap_layer.0.weight':
            gap_tensor = v
            #print('gap-tensor matrix',gap_tensor.shape)
    flattened_tensor = tensor.view(-1,4)
    picked_vector = gap_tensor[index,:]
    picked_vector = torch.unsqueeze(picked_vector,0)
    print(picked_vector.shape)
    resulted = torch.mul(flattened_tensor,picked_vector.T)
    resulted = torch.reshape(resulted,(240,2,2))
    resulted = torch.sum(resulted,0)
    resulted = torch.unsqueeze(resulted,0)
    resulted = torch.unsqueeze(resulted,0)
    print('resulted', resulted.shape)
    return upsample(resulted)
    
    
def test():
    cam.eval()
    correct,total = 0,0
 
    for data in testloader:
        images,labels = data
           
        images = images.cuda()
        print('shape of image', images.shape)
        labels = labels.cuda()
        outputs = cam(images)
        outputs = torch.squeeze(outputs)
        
        _,predicted = torch.max(outputs.data,0)
        total += 1
        corrected = predicted==labels
        
        if corrected:
            
            cam_image = create_cam(images,cam.state_dict(),predicted,(28,28)).cpu().numpy()
            cam_image = np.squeeze(cam_image)
            plt.imshow(cam_image)
            plt.pause(0.1)
            plt.imshow(np.squeeze(images.cpu().numpy()))
            plt.pause(0.1)
            print(predicted)
            #break
           
      
            
        print(labels,predicted,corrected)  
        correct += (predicted==labels).sum().item()
    accuracy = (correct/total)*100
    print('correct on test set '+ str(correct))
    print('accuracy on test set '+ str(accuracy))  
    

if __name__ == '__main__':

    cam = camnet()
    cam.cuda()
    cam_criterion = nn.CrossEntropyLoss()
    cam_optimizer = optim.SGD(cam.parameters(),lr = 0.1 , momentum=0.9)
    train(10)
    torch.save(cam.state_dict(),'camnet.pth') #the whole parameter model is saved
    upsample = nn.Upsample(size=(28,28),mode='bilinear') #upsampling method
    test()