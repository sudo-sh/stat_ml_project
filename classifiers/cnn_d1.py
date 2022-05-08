# PyTorch libraries and modules
import torch
from torch.autograd import Variable
#from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD


# importing the libraries
#import pandas as pd
import numpy as np

# for reading and displaying images
#from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),                      
            nn.Dropout(p=0.1),
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                

            nn.BatchNorm2d(num_features=32),

            nn.Dropout(p=0.1),
        )
       
        #self.conv3= nn.Sequential(         
        #    nn.Conv2d(32, 64, 5, 1, 2),     
        #    nn.ReLU(),                      
        #    nn.MaxPool2d(2),                
        #)
        #self.conv4= nn.Sequential(         
        #    nn.Conv2d(64, 128, 5, 1, 2),     
        #    nn.ReLU(),                      
        #    nn.MaxPool2d(2),                
        #)
             
        # fully connected layer, output 2 classes
        self.out = nn.Linear(32* 32 * 32, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #x=self.conv3(x)
        #x=self.conv4(x)
        # flatten the output of conv2 to (batch_size, 132 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization


def train(num_epochs, cnn, loaders,loss_func,optimizer):
    
    loss_ar=[]
    cnn.train()
        
    # Train the model
    total_step = len(loaders['train'])
        
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            
            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y

            output = cnn(b_x)[0]               
            #print(output)
            #print(b_y)
            loss = loss_func(output, b_y)
            
            # clear gradients for this training step   
            optimizer.zero_grad()           
            
            # backpropagation, compute gradients 
            loss.backward()    
            # apply gradients             
            optimizer.step()                
            
            #if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            print(loss.item())
                #pass
            val=loss.item()
            loss_ar.append(val)
            #exit()


            pass
    
    
        pass


    np.save("loss_arr.npy",np.array(loss_ar))


def test(cnn,loaders,ty):
    # Test the model
    cnn.eval()
    avg_time_DT = 0.0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders[ty]:
            start_time = time.time()
            test_output, last_layer = cnn(images)
            #print(test_output[:10])
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            pred_y=np.array(pred_y)
            #print(pred_y[:10])
            #print(parse_label(labels)[:10])
            #print((pred_y == parse_label(labels)))

            accuracy = np.count_nonzero(pred_y == parse_label(labels)) / float(labels.size(0))
            avg_time_DT += (time.time() - start_time)

            tn,fp,fn,tp=confusion_matrix(parse_label(labels),pred_y).ravel()
            pass
    num_runs=pred_y.shape[0]
    avg_time_DT /= num_runs
    print(ty+' Accuracy of the model on the images: %.2f' % accuracy)
    print(ty+" running time per inference {}".format(avg_time_DT))
    specificity_DT = tn / (tn+fp)
    sensitivity_DT = tp / (tp+fn)
    print(ty+" Specificity {}".format(specificity_DT))
    print(ty+" Sensitivity {}".format(sensitivity_DT))


    pass


'''
This Function parses the labels


'''

def parse_label(labels):

    mod_labels=np.zeros(labels.shape[0])
    for i in range(0,len(labels)):
        if(labels[i][0]==1):
            mod_labels[i]=0
        else:
            mod_labels[i]=1

    return mod_labels


def main():

    train_data=np.load("../dim_red/train_test_val/train_data.npy")
    train_label=np.load("../dim_red/train_test_val/train_label.npy")

    val_data=np.load("../dim_red/train_test_val/val_data.npy")
    val_label=np.load("../dim_red/train_test_val/val_label.npy")

    test_data=np.load("../dim_red/train_test_val/test_data.npy")
    test_label=np.load("../dim_red/train_test_val/test_label.npy")


    train_data=train_data.reshape(train_data.shape[0],train_data.shape[3],train_data.shape[1],train_data.shape[2])
    val_data=val_data.reshape(val_data.shape[0],val_data.shape[3],val_data.shape[1],val_data.shape[2])
    test_data=test_data.reshape(test_data.shape[0],test_data.shape[3],test_data.shape[1],test_data.shape[2])

    tensor_x=torch.Tensor(train_data)
    tensor_y=torch.Tensor(train_label)

    tensor_val_x=torch.Tensor(val_data)
    tensor_val_y=torch.Tensor(val_label)

    tensor_test_x=torch.Tensor(test_data)
    tensor_test_y=torch.Tensor(test_label)


    print(tensor_x.size())
    #print(tensor_y.size())
    #print(tensor_test_x.size())
    #print(tensor_test_y.size())

    train_dataset=TensorDataset(tensor_x,tensor_y)
    test_dataset=TensorDataset(tensor_test_x,tensor_test_y)
    val_dataset=TensorDataset(tensor_val_x,tensor_val_y)
   
    #print(train_dataset)
    #print(test_dataset)
    #Plot one training image
    #plt.imshow(tensor_x.data[0], cmap='gray')
    #plt.title('%i' % tensor_y[0])
    #plt.show()

    #Plot some training data and label
    #figure = plt.figure(figsize=(10, 8))
    #cols, rows = 5, 5
    #for i in range(1, cols * rows + 1):
    #    sample_idx = torch.randint(len(tensor_x), size=(1,)).item()
    #    img, label = tensor_x[sample_idx],tensor_y.data[sample_idx]
    #    figure.add_subplot(rows, cols, i)
    #    plt.title(label)
    #    plt.axis("off")
    #    plt.imshow(img.squeeze(), cmap="gray")
    #plt.show()
   
    loaders = {
        'train' : torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=100, 
                                              shuffle=True, 
                                              num_workers=1),
        
        'test'  : torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=test_data.shape[0], 
                                              shuffle=True, 
                                              num_workers=1),
         'val'  : torch.utils.data.DataLoader(val_dataset, 
                                              batch_size=val_data.shape[0], 
                                              shuffle=True, 
                                              num_workers=1),

    }




    cnn=CNN()

    print(cnn)

    loss_func=nn.CrossEntropyLoss()


    optimizer = Adam(cnn.parameters(),lr=0.01)

    num_epochs=50


    train(num_epochs, cnn, loaders,loss_func,optimizer)

    test(cnn,loaders,ty="test")
    test(cnn,loaders,ty="val")
    #test(cnn,loaders,ty="train")

if(__name__=="__main__"):
    main()
