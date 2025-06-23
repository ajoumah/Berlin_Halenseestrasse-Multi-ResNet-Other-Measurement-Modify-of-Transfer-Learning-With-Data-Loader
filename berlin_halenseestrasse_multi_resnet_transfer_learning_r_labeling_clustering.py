"""Berlin_Halenseestrasse Multi ResNet andUpdate DTW With Comparison Other Measurement Modify of Transfer Learning With Data Loader For St.Lucia Fining With AlexNet365 and ResNet50 And With Labeling and Clustering.ipynb.ipynb

"""

from _collections import defaultdict
#import caffe
import pickle
import numpy as np
from skimage.measure import regionprops,label
import itertools
import time
import joblib
from joblib import load
#from keras.utils import plot_model
from keras import models
from sklearn.metrics import precision_recall_curve,auc,average_precision_score
import os
import matplotlib.pyplot as plt
from pylab import imread,subplot,imshow,show
import matplotlib.gridspec as gridspec
import math
import argparse
import imutils
import os, codecs
import tensorflow as tf
import scipy.spatial.distance as dist
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import metrics
import cv2
from skimage.feature import hog
from skimage import data, exposure
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

def get_Image_HOG(testI):
    img = imread(testI)
    resized_img = resize(img, (128,64))

    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)
    #img = imread(testI) #as normal image
    #image = cv2.imread(testI,0) # as bayer Image
    #convertedImage = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2RGB) #convert bayer to BGR as OpenCv
    #im_rgb = cv2.cvtColor(convertedImage, cv2.COLOR_BGR2RGB) #convert BGR to RGB
    #resized_img = resize(im_rgb, (128,64))

    #fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, multichannel=True)

    #print("shape of Feature of HOG is")
    #print(fd.shape)

    return fd

def Calculate_Distance(matrix):
    labels=np.zeros(matrix.shape[0], dtype=bool)
    #clusters=[]
    #for i in range(0,matrix.shape[0]) :
    #    distance=[]
    #    for j in range(0,matrix.shape[0]):
    #        dist=dist.cityblock(matrix[i,:], matrix[j,:])
    #        distance.append(dis)
    #    sorted_index=np.argsort(distance)
    #    time=0
    #    for q in range(0,matrix.shape[0]):
    #        if labels[sorted_index[q]]
    cluster=[]
    added=range(0,matrix.shape[0])
    classified=s=np.ones(matrix.shape[0], dtype=bool)

    classified=np.array(classified)
    classified=classified*(-1)
    classifiedAndLabel=np.ones(matrix.shape[0], dtype=bool)
    classifiedAndLabel=classifiedAndLabel*(-1)
    #k=-1
    for i in added:
        distance=[]

        if(not (labels[i])):
            distance.append(i)
            labels[i]=1
            classified[i]=i
            #k=k+1
            #classifiedAndLabel[i]=k
            added= np.delete(added, 0)
        #print("cluster number is ")
        #print(i)
        canSearch=True
        for j in added:
            if canSearch:
                #r=dist.cityblock([matrix[i,:]], [matrix[j,:]])
                r=cosine_similarity([matrix[i,:]], [matrix[j,:]])
                #a = np.array([matrix[i,:]])
                #b = np.array([matrix[j,:]])
                #r = np.linalg.norm(a-b)
                print("distance detween %d and %d is %f"%(i,j,r))
                if (r>=0.65) and (not (labels[j])):
                    dis=j
                    distance.append(dis)
                    labels[j]=1
                    classified[j]=i
                    #classifiedAndLabel[j]=k
                    added= np.delete(added, 0)
                else:
                    canSearch=False
        cluster.append(distance)
    for ind,x in enumerate(cluster):
        if (len(x)>0) :
            print('cluster number is %d '%(ind))
            print(x)
            print('-------------------------')
    print("classification is ")
    #print(classified)
    print("length of classified")
    print(len(classified))
    for i in range(0,len(classified)):
        print("i is %d  and cell is  %d"%(i,classified[i]))
    print('***********************')
    print(len(classifiedAndLabel))
    previuosState=classified[0]
    k=0
    classifiedAndLabel[0]=0
    for i in range(1,len(classified)):
        if previuosState !=classified[i]:
            k=k+1
            previuosState=classified[i]
        classifiedAndLabel[i]=k
    for i in range(0,len(classifiedAndLabel)):

        print("i is %d  and cell is  %d"%(i,classifiedAndLabel[i]))
    print("number of class is ")
    #A=A.flatten()
    c= list(set(classified))
    #c=A.unique().tolist()
    print("unique is ")
    print(len(c))
    data2 = pd.DataFrame({"X Value": range(0,len(classified)), "Y Value": classified, "Category": classified})
    groups2 = data2.groupby("Category")
    for name, group in groups2:
        plt.plot(group["X Value"], group["Y Value"], marker="o", linestyle="", label=name)
    plt.xlabel('Image index ')
    plt.ylabel('Cluster index');
    plt.title("HOG Features ")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.show()
    return classifiedAndLabel

def Convert2List(string):
    list1=[]
    list1[:0]=string
    return list1

def compareString(s1,s2):
    l1=len(s1)
    l2=len(s2)
    for i in range(1,l2+1):
        s1[-i]=s2[-i]
    return s1
def GetStringPathConversion(c2):

    ###c1='000'
    c1='00000'
    c1=Convert2List(c1)
    #c2='1027'
    c2=Convert2List(c2)
    x=compareString(c1,c2)
    #print(x)
    str1 = ""
    f=str1.join(x)
    #print(f)
    return f

dir = "/content/drive/MyDrive/ASLI/berlin_halenseestrasse/berlin_halenseestrasse_1"
    #datasetIndex = 3
img_for=".jpg"
    #imgTag= "images-"
imgTag= "Image"


files_list=[]
for file in os.listdir(dir):
    if file.endswith("jpg")and file.startswith("HH_sCmavjtovcNk5NX8vEg-640-"):

        files_list.append(file)
print(len(files_list))

print(files_list)

file_order1=[]
for i in range(400,467):
    name="HH_sCmavjtovcNk5NX8vEg-640-"+GetStringPathConversion(str(i))+".jpg"
    file_order1.append(name)
print(len(file_order1))

file_order2=[]
for i in range(136,203):
    name="573YUQmuV6nbZK9DvrqT_w-640-"+GetStringPathConversion(str(i))+".jpg"
    file_order2.append(name)
print(len(file_order2))

print(file_order2)

def Read_Images_HOG(source_dir,files_list_oder):
    #dir = "/content/drive/MyDrive/Lucia2/3/"
    dir=source_dir
    #datasetIndex = 3
    #img_for=".jpg"
    #imgTag= "images-"
    #imgTag= "Image"


    #files_list=[]
    #for file in os.listdir(dir):
    #    if file.endswith("jpg")and file.startswith("Image"):
    #        files_list.append(file)
    print(len(files_list_oder))

    images_HOG_list=[]

    #for testImgName,file in enumerate(files_list):
    for i in range(0,len(files_list_oder)):
        #refImg = os.path.join(dir,images_list_file[i])
        refImg = os.path.join(dir,files_list_oder[i])
        #print(imgTag+GetStringPathConversion(str(1+testImgName))+img_for)
        #print(refImg)
        image_Feature=get_Image_HOG(refImg)
        #print("shape of image_Feature is")
        #print(image_Feature.shape)
        images_HOG_list.append(image_Feature)

    images_HOG_list=np.array(images_HOG_list)

    #print("HOG shape feature is ")
    #print(images_HOG_list.shape)
    return images_HOG_list

images_st_Lucia=[]
#Read_Images_HOG()
images_st_Lucia=Read_Images_HOG('/content/drive/MyDrive/berlin_halenseestrasse/berlin_halenseestrasse_1',file_order1)
ClassifiedLabel=Calculate_Distance(images_st_Lucia)

# Commented out IPython magic to ensure Python compatibility.
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
import glob
import cv2
import os.path
import shutil
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
# %matplotlib inline

len(ClassifiedLabel)

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
data=ClassifiedLabel
c= list(set(ClassifiedLabel))
    #c=A.unique().tolist()
print("unique is ")
print(len(c))
#data = [3, 5, 1, 2,2,7, 9, 5, 3,3,3, 7, 5,4,4,4,4,6,7,8,9]
_, _, patches = plt.hist(data ,len(c),facecolor='blue', alpha=0.5)

for pp in patches:
   x = (pp._x0 + pp._x1)/2
   y = pp._y1 + 0.05
   if y>22:
       plt.text(x, y, pp._y1)
plt.subplots_adjust(left=0.25)
plt.show()

import numpy as np #pip install numpy

import matplotlib.pyplot as plt #pip install matplotlib

#generate a random numpy array with 1000 elements

#data = np.random.randn(1000)
data=ClassifiedLabel
#plot the data as histogram

plt.hist(data,edgecolor="black", bins =len(c))

#histogram title

plt.title("Histogram for 67 elements")

#histogram x axis label

plt.xlabel("Values")

#histogram y axis label

plt.ylabel("Frequencies")

#display histogram

plt.show()

import plotly.express as px
import numpy as np

#df = px.data.tips()
# create the bins
counts, bins = np.histogram(data , bins=range(0,len(c)))
bins = 0.5 * (bins[:-1] + bins[1:])

fig = px.bar(x=bins, y=counts, labels={'x':'groups', 'y':'count'},title=" 10 groups with similirity>=0.65")
fig.show()
#fig = px.histogram(data, x=range(0,len(c)))
#fig.show()

##Configure mapping between image path and lable of the image

def createMappingForDataSetWithLabels(source_dir1,files_list_oder,ClassifiedLabel):

    #imgs_path = "/content/drive/MyDrive/Lucia2/3/"
    img_for=".jpg"
    #imgTag= "images-"
    imgTag= "Image"
    dir1=source_dir1
    ###dir2=source_dir2
    #file_list = glob.glob(imgs_path + "*")
    #print(file_list)
    data = []
    for i in range(0,len(ClassifiedLabel)):

        ###img_path1=os.path.join(dir1,images_list_file[i])


        img_path1 = os.path.join(dir1,files_list_oder[i])
        ###img_path2=os.path.join(dir2,images_list_file[i])
        z='{0:011b}'.format(ClassifiedLabel[i])
        Classified='1111'+z
        data.append([img_path1, Classified])
        ###data.append([img_path2, Classified])
            #print([img_path, 0])

    return data

dataset=createMappingForDataSetWithLabels('/content/drive/MyDrive/berlin_halenseestrasse/berlin_halenseestrasse_1',file_order1,ClassifiedLabel)

print(len(dataset))

print(dataset[0:67])

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
##rotate_30 = rotate_image(img, random.randint(-45, 45))

import os.path
import shutil
from pathlib import Path
import skimage
from skimage.util import random_noise
import random
def apply_mask(image, size=20, n_squares=1):
    h, w, channels = image.shape
    new_image = image
    for _ in range(n_squares):
        #y = np.random.randint(h)
        y=random.randint(0, 10)
        #x = np.random.randint(w)
        x=random.randint(0, 10)
        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)
        new_image[y1:y2,x1:x2,:] = 0
    return new_image
###mask=apply_mask(img)

import os.path
import shutil
from pathlib import Path
import skimage
from skimage.util import random_noise
import random


def CreateFolderByLabel(listofPathAndLabel,dir):
    train_folder=dir+'train'
    if not (os.path.isdir(train_folder)):
        Path(train_folder).mkdir(parents=True, exist_ok=True)
    else:
        shutil.rmtree(train_folder)
        Path(train_folder).mkdir(parents=True, exist_ok=True)
    classfolder=''
    for img_path, class_name in listofPathAndLabel:
        #print([img_path,class_name])
        #print(train_folder)
        classfolder=train_folder+'/'+str(class_name)
        #print(classfolder)
        if not (os.path.isdir(classfolder)):
            Path(classfolder).mkdir(parents=True, exist_ok=True)

        ######
        img = cv2.imread(img_path)
        img = cv2.resize(img, (227,227))
        #convertedImage = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB) #convert bayer to BGR as OpenCv
        #im_rgb = cv2.cvtColor(convertedImage, cv2.COLOR_BGR2RGB) #convert BGR to RGB
        #im_rgb=convertedImage


        #print(os.listdir(directory))
        filename = img_path.split("/")[-1]
        folderName = img_path.split("/")[-2]
        #if (os.path.exists(classfolder+'/'+filename)):
        ###if(folderName=='Nordland1'):
        ###filename=filename[:-4]+'-r'+'.png'
        ###os.chdir(classfolder)
        ###cv2.imwrite(filename, img)
           #print(filename,'as r')
        ###else:
        os.chdir(classfolder)
        cv2.imwrite(filename, img)
        #########################################################
        ###Save Blur images ###

        img_blur = cv2.blur(img, (5,5), cv2.BORDER_DEFAULT)
        fname=filename[:-4]+'-b'+'.jpg'
        cv2.imwrite(fname, img_blur)
        ###
        ###Salt & pepper###
        noise_img = random_noise(img, mode='s&p',amount=0.01)
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        fname=filename[:-4]+'-s'+'.jpg'
        cv2.imwrite(fname, noise_img)
        ###
        ###Histogram Equaliation###
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        contrast_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        fname=filename[:-4]+'-h'+'.jpg'
        cv2.imwrite(fname, contrast_image)
        ###
        ###Rotation###
        rotate_30 = rotate_image(img, random.randint(-45, 45))
        fname=filename[:-4]+'-t'+'.jpg'
        cv2.imwrite(fname, rotate_30)
        ###
        ###Cut###
        mask=apply_mask(img)
        fname=filename[:-4]+'-c'+'.jpg'
        cv2.imwrite(fname, mask)
        ###
        #########################################################
    return train_folder
        #print(os.listdir(directory))

#dataset = createMappingForDataSet()
#dataset=createMappingForDataSetWithLabels(ClassifiedLabel)
dataset=createMappingForDataSetWithLabels('/content/drive/MyDrive/berlin_halenseestrasse/berlin_halenseestrasse_1',file_order1,ClassifiedLabel)
dir='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/'
TRAIN_DIR=CreateFolderByLabel(dataset,dir)

def Count_Number_Of_Images_In_Folder(Path):
    #dir = "/content/drive/MyDrive/SavedImage/colored/"
    dir=Path



    files_list=[]
    for file in os.listdir(dir):
        if file.endswith("jpg"):
            files_list.append(file)
    return len(files_list)

def Count_Number_Of_Images_In_All_Folders(Path,EndIndex):
    no_of_images_till_last_folder=0
    last_folder_size=0
    if (os.path.isdir(Path)):
        for i in range(0,EndIndex):
            z='{0:011b}'.format(i)
            Classified='1111'+z
            classfolder=Path+'/'+str(Classified)
            if (os.path.isdir(classfolder)):
                no_of_images_till_last_folder=no_of_images_till_last_folder+Count_Number_Of_Images_In_Folder(classfolder)
        z='{0:011b}'.format(EndIndex)
        Classified='1111'+z
        classfolder=Path+'/'+str(Classified)
        if (os.path.isdir(classfolder)):
            last_folder_size=Count_Number_Of_Images_In_Folder(classfolder)
    return no_of_images_till_last_folder,last_folder_size

n,size=Count_Number_Of_Images_In_All_Folders("/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train",10)
print(n)
print(size)

TRAIN_DIR='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train'

train_data = ImageFolder(TRAIN_DIR)

len(train_data)

CLASSES = list(train_data.class_to_idx.keys())
len(CLASSES)

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
    T.ToTensor(),
    T.Normalize(*imagenet_stats,inplace=True)
])

valid_tfms = T.Compose([ T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
])

test_tfms = T.Compose([ T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(*imagenet_stats)
])

train_data = ImageFolder(TRAIN_DIR, train_tfms)

m,l=train_data[401]
print(l)

General_DataSet=train_data

dataset_size = len(train_data)
#print(dataset_size)
dataset_indices = list(range(dataset_size))
#print(dataset_indices)
np.random.shuffle(dataset_indices)
#print(dataset_indices)
val_split_index = int(np.floor(0.1 * dataset_size))
train_idx, val_idx,test_idx = dataset_indices[2*val_split_index:], dataset_indices[:val_split_index],dataset_indices[val_split_index:2*val_split_index]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)
test_sampler = SubsetRandomSampler(test_idx)
train_ld = DataLoader(dataset=train_data, shuffle=False, batch_size=32, sampler=train_sampler)
valid_ld = DataLoader(dataset=train_data, shuffle=False, batch_size=32, sampler=val_sampler)
test_ld = DataLoader(dataset=train_data, shuffle=False, batch_size=32, sampler=test_sampler)

def show_batch(dl, invert=True):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_xticks([]); ax.set_yticks([])
        data = 1-images if invert else images
        ax.imshow(make_grid(data, nrow=8).permute(1, 2, 0))
        break

show_batch(test_ld, invert=False)

train_ds = ImageFolder(TRAIN_DIR, train_tfms)
#val_ds = ImageFolder(VALID_DIR, valid_tfms)
#test_ds = ImageFolder(TEST_DIR, test_tfms)
print(len(train_ds))

General_Train_DataSet=train_ds

General_Train_DataSet=train_ds

img_tensor, label = train_ds[0]
print(img_tensor.shape, label)

len(train_ds.classes)

len(General_Train_DataSet.classes)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # prediction Generation
        loss = F.cross_entropy(out, labels) # Calculation of loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # prediction Generation
        #print(" out shape")
        #print(len(out))
        #print("here is out")
        #print(out)
        #_, preds = torch.max(out, dim=1)
        #print(" out shape")
        #print(preds.shape())
        #print("here is preds")
        #print(preds)
        #print("here is -")
        #print(_)
        loss = F.cross_entropy(out, labels)   # Calculation of loss
        acc = accuracy(out, labels)           # Calculation of accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Merging of losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Merging of  accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

# th architecture to use
arch = 'resnet18'
#arch = 'alexnet'
# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
#model.eval()
#GlobalAlexnet = model
#GlobalAlexnet
GlobalResNet = model
#GlobalResNet

class ClusterAlexnet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use alext net pretrained model
        #self.network = models.alexnet(pretrained=True)
        #self.network = models.resnet34(pretrained=True)
        #self.network = new_model
        self.network = GlobalResNet
        # Chaning of the last layer
        num_ftrs = self.network.fc.in_features
        #num_ftrs = self.network.classifier[6].in_features
        #num_ftrs = 9216
        #num_ftrs=self.network.classifier[6].in_features
        #num_ftrs=self.network.classifier[6].in_features for -1 Children and param in self.network.classifier[6].parameters()
        #num_ftrs=self.network.classifier[6].in_features for -2 Children and param in self.network.classifier[6].parameters()
        #num_ftrs=self.network.classifier[6].in_features for -3 Children and param in self.network.classifier[6].parameters()
        #num_ftrs=self.network.classifier[6].in_features for -4 Children and param in self.network.classifier[6].parameters()
        #features=list(self.network.classifier.children())[:-6]
        #features.extend([nn.Linear(num_ftrs,len(train_ds.classes))])
        #self.network.classifier=nn.Sequential(*features)
        #self.network.classifier[6] = nn.Linear(num_ftrs,len(train_ds.classes))
        #print("number of features in")
        #print(num_ftrs)
        #self.network.classifier[6] = nn.Linear(num_ftrs, len(train_ds.classes))
        #num_ftrs = self.network.fc.in_features
        #num_ftrs = 512
        #num_ftrs = 256
        #num_ftrs = 128
        #num_ftrs=32768   #-2 layer
        #num_ftrs=57600   #-3 layer
        #num_ftrs=107648  #-4 layer
        #num_ftrs=207936  #-5 layer
        #num_ftrs=207936  #-6 layer
        #num_ftrs=831744  #-7 layer
        #num_ftrs=831744 #-8 layer
        #self.network.AvgPool=nn.AdaptiveAvgPool2d((1,1))
        #self.my_new_layers = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),nn.Linear(512 , len(train_ds.classes)))
        #self.network.fc = nn.Linear(num_ftrs, len(train_ds.classes))
        #self.network.classifier_block = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        #self.network.fc = nn.Linear(num_ftrs, len(train_ds.classes))
        #self.network.AvgPool=GloabalAVGPOOL number_of_output)
        self.network.fc = nn.Linear(num_ftrs, len(train_ds.classes))
        #self.network.fc = nn.Linear(num_ftrs, number_of_output)
        print("number of features in")
        print(num_ftrs)
    def forward(self, xb):

        return torch.sigmoid(self.network(xb))

    def freeze(self):
        # To freezing first layers
        for param in self.network.parameters():
            param.require_grad = False
        #for param in self.network.classifier.parameters():
        #    param.require_grad = True
        for param in self.network.fc.parameters():
            param.require_grad = True
    def unfreeze(self):
        # Unfreeze of all of the layers
        for param in self.network.parameters():
            param.require_grad = True

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_ld = DeviceDataLoader(train_ld, device)
#print(len(valid_ld))
valid_ld = DeviceDataLoader(valid_ld, device)
#print(len(valid_ld))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,Saving_Folder_file,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))
    #max_acc=0.9724
    max_acc=0
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)

            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        network_saving_folder=Saving_Folder_file
        if(result['val_acc']> max_acc):
            max_acc= result['val_acc']
            torch.save(model, network_saving_folder)
            print("max accuaracy",max_acc)
        if result['val_acc']==1:
            break

    return history

model = to_device(ClusterAlexnet(), device)

from torchsummary import summary
summary(model, input_size=(3, 227, 227))

model.freeze()

epochs = 150
max_lr = 0.001
grad_clip = 1
weight_decay = 1e-6
opt_func = torch.optim.Adam

# Commented out IPython magic to ensure Python compatibility.
# %%time
# ###model = torch.load('/content/drive/MyDrive/SavedImage/networks/master/checkpoint.pt')
# ###model.eval()
# #model.freeze()
# #for i in range(0,1):
# network_saving_folder='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/networks/master/' +'checkpoint.pt'
# history = fit_one_cycle(epochs, max_lr, model, train_ld,valid_ld,network_saving_folder ,
#                          grad_clip=grad_clip,
#                          weight_decay=weight_decay,
#                          opt_func=opt_func)
# 
#         #network_saving_folder=network_saving_folder
#         #torch.save(model.state_dict(), network_saving_folder)
# #torch.save(model, network_saving_folder)
# #accuracies = [x['val_acc'] for x in history]
# #train_losses = [x.get('train_loss') for x in history]
# #val_losses = [x['val_loss'] for x in history]
# #now = datetime.now()
# 
# 
#

####RGB#############
model_master = torch.load('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/networks/master/checkpoint.pt')
model_master.eval()

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs,Resnet18 With 11 Clusters  ');

plot_accuracies(history)

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs,Resnet18 With 11 Clusters ');

plot_losses(history)

def plot_lrs(history):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.');

plot_lrs(history)

##Configure mapping between image path and lable of the image

def create_Labels_for_every_image_in_cluster(cluster_dir,files_list_oder,start,size):

    #imgs_path = "/content/drive/MyDrive/Lucia2/3/"
    #dir=imgs_path
    dir=cluster_dir
    #img_for=".bmp"
    #imgTag= "images-"
    #imgTag= "cam0_image0"
    img_for=".jpg"
    #imgTag= "images-"
    imgTag= "Image"
    files_list=[]
    start=int(start/6)
    for file in os.listdir(dir):
        if file.endswith("jpg"):
            files_list.append(file)
    #print("our function")
    #print(len(files_list))
    data = []
    #print("path of folder",dir)
    #print("length of files",int((len(files_list)/2)))
    #print("size of the group",size)
    if(int((len(files_list)/6))==size):
        for i in range(start,start+size):
            ###img_path=os.path.join(dir,images_list_file[i])
            img_path = os.path.join(dir,files_list_oder[i])
            ###fname=images_list_file[i]
            ###fname=fname[:-4]+'-r'+'.png'
            ###img_path1=os.path.join(dir,fname)
            z='{0:016b}'.format(i)
            Classified='1111'+z
            data.append([img_path, Classified])
            ###data.append([img_path1, Classified])
            #print([img_path, 0])

    return data

def Count_Number_Of_Images_In_Folder(Path):
    #dir = "/content/drive/MyDrive/SavedImage/colored/"
    dir=Path



    files_list=[]
    for file in os.listdir(dir):
        if file.endswith("jpg"):
            files_list.append(file)
    return len(files_list)

n=Count_Number_Of_Images_In_Folder('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train/111100000000000')
print(n)

def Count_Number_Of_Images_In_All_Folders(Path,EndIndex):
    no_of_images_till_last_folder=0
    last_folder_size=0
    if (os.path.isdir(Path)):
        for i in range(0,EndIndex):
            z='{0:011b}'.format(i)
            Classified='1111'+z
            classfolder=Path+'/'+str(Classified)
            if (os.path.isdir(classfolder)):
                no_of_images_till_last_folder=no_of_images_till_last_folder+Count_Number_Of_Images_In_Folder(classfolder)
        z='{0:011b}'.format(EndIndex)
        Classified='1111'+z
        classfolder=Path+'/'+str(Classified)
        if (os.path.isdir(classfolder)):
            last_folder_size=Count_Number_Of_Images_In_Folder(classfolder)
    return no_of_images_till_last_folder,last_folder_size

n,size=Count_Number_Of_Images_In_All_Folders("/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train",5)
print(int(n))
print(int(size))

n,size=Count_Number_Of_Images_In_All_Folders("/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train",3)
print(int(n))
print(int(size))

Path='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train'
z='{0:011b}'.format(9)
Classified='1111'+z
classfolder=Path+'/'+str(Classified)
print(classfolder)
n,size=Count_Number_Of_Images_In_All_Folders("/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train",9)
print(n)
print(size)
#Path_and_labels=create_Labels_for_every_image_in_cluster(classfolder,int(n/2),int(size/2))
Path_and_labels=create_Labels_for_every_image_in_cluster(classfolder,file_order1,int(n),int(size/6))
print(Path_and_labels)

Path='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train'
z='{0:011b}'.format(0)
Classified='1111'+z
classfolder=Path+'/'+str(Classified)
print(classfolder)
n,size=Count_Number_Of_Images_In_All_Folders("/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train",0)
print(n/6)
print(size/6)
#Path_and_labels=create_Labels_for_every_image_in_cluster(classfolder,int(n/2),int(size/2))
Path_and_labels=create_Labels_for_every_image_in_cluster(classfolder,file_order1,int(n),int(size/6))
print(Path_and_labels)

import os.path
import shutil
from pathlib import Path

import os.path
import shutil
from pathlib import Path
import os.path
import shutil
from pathlib import Path
import skimage
from skimage.util import random_noise
import random
def CreateFolderByLabel_for_Every_Image_in_cluster(listofPathAndLabel,dir):
    saving_folder=dir+'/'+'trainEveryImage'
    #print(saving_folder)
    if not (os.path.isdir(saving_folder)):
        Path(saving_folder).mkdir(parents=True, exist_ok=True)
    else:
        shutil.rmtree(saving_folder)
        Path(saving_folder).mkdir(parents=True, exist_ok=True)
    classfolder=''
    for img_path, class_name in listofPathAndLabel:
        #print([img_path,class_name])
        #print(train_folder)
        classfolder=saving_folder+'/'+str(class_name)
        #print(classfolder)
        if not (os.path.isdir(classfolder)):
            Path(classfolder).mkdir(parents=True, exist_ok=True)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (227,227))

        os.chdir(classfolder)

        #print(os.listdir(directory))
        #print(filename)
        filename = img_path.split("/")[-1]
        cv2.imwrite(filename, img)
        ##########################################
        #print(os.listdir(directory))
        ###filename = img_path.split("/")[-1]
        ###folderName = img_path.split("/")[-2]
        #if (os.path.exists(classfolder+'/'+filename)):
        ###if(folderName=='Nordland1'):
        ###filename=filename[:-4]+'-r'+'.png'
        ###os.chdir(classfolder)
        ###cv2.imwrite(filename, img)
           #print(filename,'as r')
        ###else:
        ###os.chdir(classfolder)
        ###cv2.imwrite(filename, img)
        #########################################################
        ###Save Blur images ###

        img_blur = cv2.blur(img, (5,5), cv2.BORDER_DEFAULT)
        fname=filename[:-4]+'-b'+'.jpg'
        cv2.imwrite(fname, img_blur)
        ###
        ###Salt & pepper###
        noise_img = random_noise(img, mode='s&p',amount=0.01)
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        fname=filename[:-4]+'-s'+'.jpg'
        cv2.imwrite(fname, noise_img)
        ###
        ###Histogram Equaliation###
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_yuv = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        contrast_image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        fname=filename[:-4]+'-h'+'.jpg'
        cv2.imwrite(fname, contrast_image)
        ###
        ###Rotation###
        rotate_30 = rotate_image(img, random.randint(-45, 45))
        fname=filename[:-4]+'-t'+'.jpg'
        cv2.imwrite(fname, rotate_30)
        ###
        ###Cut###
        mask=apply_mask(img)
        fname=filename[:-4]+'-c'+'.jpg'
        cv2.imwrite(fname, mask)
        ##########################################
    return saving_folder
        #print(os.listdir(directory))

Pathloss='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train'
z='{0:011b}'.format(9)
Classified='1111'+z
clusterfolder=Pathloss+'/'+str(Classified)
#print(classfolder)
n,size=Count_Number_Of_Images_In_All_Folders(Pathloss,9)
print(n)
print(size)
Path_and_labels=create_Labels_for_every_image_in_cluster(clusterfolder,file_order1,int(n),int(size/6))
print(Path_and_labels)
#dataset = createMappingForDataSet()
#Labels_of_every_image=create_Labels_for_every_image()
#dir='/content/drive/MyDrive/SavedImage/'
cluster_path='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/clustered'
clustering_folder=cluster_path+'/'+str(Classified)
TRAIN_DIR_For_Every_Image=CreateFolderByLabel_for_Every_Image_in_cluster(Path_and_labels,clustering_folder)
print(TRAIN_DIR_For_Every_Image)

def CreateFolderByLabel_for_Every_Image_in_all_cluster(sourcePath,files_list_order,clusterPath):

    ###for i in range(0,len(General_Train_DataSet.classes)):
    for i in range(0,10):
        #Pathloss='/content/drive/MyDrive/SavedImage/train'
        z='{0:011b}'.format(i)
        Classified='1111'+z
        clusterfolder=sourcePath+'/'+str(Classified)
        #print(classfolder)
        n,size=Count_Number_Of_Images_In_All_Folders(sourcePath,i)
        #print(n)
        #print(size)
        Path_and_labels=create_Labels_for_every_image_in_cluster(clusterfolder,files_list_order,int(n),int(size/6))
        #print(Path_and_labels)
        #dataset = createMappingForDataSet()
        #Labels_of_every_image=create_Labels_for_every_image()
        #dir='/content/drive/MyDrive/SavedImage/'
        #cluster_path='/content/drive/MyDrive/SavedImage/clustered'
        clustering_folder=clusterPath+'/'+str(Classified)
        TRAIN_DIR_For_Every_Image=CreateFolderByLabel_for_Every_Image_in_cluster(Path_and_labels,clustering_folder)
        #print(TRAIN_DIR_For_Every_Image)
    return clusterPath

cluster_directory=CreateFolderByLabel_for_Every_Image_in_all_cluster('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train',file_order1,'/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/clustered')

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_tfms = T.Compose([
    T.ToTensor(),
    T.Normalize(*imagenet_stats,inplace=True)
])

def create_cluster_dataset_by_cluster_number(clustersource):
    train_data = ImageFolder(clustersource, train_tfms)
    return train_data

td=create_cluster_dataset_by_cluster_number('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/clustered/111100000000000/trainEveryImage')
print(len(td))
print(len(td.classes))

###for i in range(0,len(General_Train_DataSet.classes)):
for i in range(0,10):
        #Pathloss='/content/drive/MyDrive/SavedImage/train'
        z='{0:011b}'.format(i)
        Classified='1111'+z

        #cluster_path='/content/drive/MyDrive/SavedImage/clustered'
        cluster_path='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/clustered'
        clustering_folder=cluster_path+'/'+str(Classified)+'/trainEveryImage'
        td=create_cluster_dataset_by_cluster_number(clustering_folder)
        print(i)
        print('cluster length')
        print(len(td))
        dataset_size = len(td)
        #print(dataset_size)
        dataset_indices = list(range(dataset_size))
        #print(dataset_indices)
        np.random.shuffle(dataset_indices)
        #print(dataset_indices)
        #val_split_index = int(np.floor(0.1 * dataset_size))
        train_idx = dataset_indices[:]
        train_sampler = SubsetRandomSampler(train_idx)
        #val_sampler = SubsetRandomSampler(val_idx)
        #test_sampler = SubsetRandomSampler(test_idx)
        train_ld = DataLoader(dataset=td, shuffle=False, batch_size=32, sampler=train_sampler)
        #print(len(train_ld))
        #valid_ld = DataLoader(dataset=train_data, shuffle=False, batch_size=32, sampler=val_sampler)
        #test_ld = DataLoader(dataset=train_data, shuffle=False, batch_size=32, sampler=test_sampler)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # prediction Generation
        loss = F.cross_entropy(out, labels) # Calculation of loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # prediction Generation
        loss = F.cross_entropy(out, labels)   # Calculation of loss
        acc = accuracy(out, labels)           # Calculation of accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Merging of losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Merging of  accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

def Get_New_Network():
    # th architecture to use
    arch = 'resnet18'
    #arch = 'alexnet'
    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    #Multi_GlobalResNet = model
    return model

class Multi_ClusterAlexnet(ImageClassificationBase):
    def __init__(self,GlobalModel,length_of_dataset):
        super().__init__()

        #self.network = new_model
        self.network = GlobalModel
        # Chaning of the last layer
        num_ftrs = self.network.fc.in_features

        #num_ftrs=57600   #-3 layer

        self.network.fc = nn.Linear(num_ftrs, length_of_dataset)
        #self.network.fc = nn.Linear(num_ftrs, number_of_output)
        print("number of features in")
        print(num_ftrs)
        print("number of length of dataset")
        print(length_of_dataset)
    def forward(self, xb):

        return torch.sigmoid(self.network(xb))

    def freeze(self):
        # To freezing first layers
        for param in self.network.parameters():
            param.require_grad = False

        for param in self.network.fc.parameters():
            param.require_grad = True
    def unfreeze(self):
        # Unfreeze of all of the layers
        for param in self.network.parameters():
            param.require_grad = True

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#train_ld = DeviceDataLoader(train_ld, device)
#print(len(valid_ld))
#valid_ld = DeviceDataLoader(valid_ld, device)
#print(len(valid_ld))

##epochs = 150
#epochs = 200
###epochs = 150
epochs = 150
#max_lr = 0.001
max_lr = 0.001

##max_lr = 0.00003

grad_clip = 1
weight_decay = 1e-6
##weight_decay = 1e-6

opt_func = torch.optim.Adam

network_dir='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/AgmentedNetwork'

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,Saving_Folder_file,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up cutom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    max_acc=0
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)

            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate

            lrs.append(get_lr(optimizer))
            sched.step()
            #print("learninig rate",lrs)

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        #network_saving_folder='/content/drive/MyDrive/SavedImage/networks/111100000000111/checkpoint.pt'
        #network_saving_folder='/content/drive/MyDrive/SavedImage/networks/111100000001000/checkpoint.pt'
        #network_saving_folder='/content/drive/MyDrive/SavedImage/networks/111100000001001/checkpoint.pt'
        network_saving_folder=Saving_Folder_file
        if(result['val_acc']> max_acc):
            max_acc= result['val_acc']
            torch.save(model, network_saving_folder)
            print("max accuaracy",max_acc)
        if result['val_acc']==1:
            break
    return history

History_of_All=[]
#for i in range(0,len(General_Train_DataSet.classes)):
for i in range(0,10):
        #Pathloss='/content/drive/MyDrive/SavedImage/train'
        z='{0:011b}'.format(i)
        Classified='1111'+z

        #cluster_path='/content/drive/MyDrive/SavedImage/clustered'
        #cluster_path='/content/drive/MyDrive/NordlandSavedImages/clustered'
        #cluster_path='/content/drive/MyDrive/NordlandSavedImages/clusteredGrey'
        ###cluster_path='/content/drive/MyDrive/NordlandSavedImages/AgmentedClustered'
        #cluster_path='/content/drive/MyDrive/NordlandSavedImages/WithoutCutAgmentedClustered' ##cluster 7 only
        ###cluster_path ='/content/drive/MyDrive/NordlandSavedImages/WithoutCutRotationAgmnetedClustered' ##cluster 7 only
        cluster_path='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/clustered'
        #cluster_path='/content/drive/MyDrive/NordlandSavedImages/WithoutCutRotationAgmnetedHistogramClustered' ##cluster 7 only
        #cluster_path='/content/drive/MyDrive/NordlandSavedImages/SaltPeperCluster'  ##cluster 7 only

        ###cluster_path2='/content/drive/MyDrive/NordlandSavedImages/clustered'
        cluster_path2='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/clustered'
        clustering_folder=cluster_path+'/'+str(Classified)+'/trainEveryImage'
        td=create_cluster_dataset_by_cluster_number(clustering_folder)
        print(i)
        print('cluster length',len(td))
        #print(len(td))
        print("classes",len(td.classes))
        dataset_size = len(td)

        dataset_indices = list(range(dataset_size))

        np.random.shuffle(dataset_indices)

        train_idx = dataset_indices[:]
        train_sampler = SubsetRandomSampler(train_idx)

        train_ld = DataLoader(dataset=td, shuffle=False, batch_size=32, sampler=train_sampler)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_ld = DeviceDataLoader(train_ld, device)
        ############validation####################################
        ###clustering_folder2=cluster_path2+'/'+str(Classified)+'/trainEveryImage'
        clustering_folder2=clustering_folder
        td2=create_cluster_dataset_by_cluster_number(clustering_folder2)
        print(i)
        print('cluster validation length',len(td2))
        #print(len(td))
        print("classes",len(td2.classes))
        dataset_size2 = len(td2)

        dataset_indices2 = list(range(dataset_size2))

        np.random.shuffle(dataset_indices2)

        train_idx2 = dataset_indices2[:]
        train_sampler2 = SubsetRandomSampler(train_idx2)

        train_ld2 = DataLoader(dataset=td2, shuffle=False, batch_size=32, sampler=train_sampler2)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        validation_ld = DeviceDataLoader(train_ld2, device)
        #################################################
        #new_model = nn.Sequential(*list(Multi_GlobalResNet.children())[:-3],nn.Flatten())
        new_model=Get_New_Network()
        model = to_device(Multi_ClusterAlexnet(new_model,len(td.classes)), device)
        model.freeze()
        #network_saving_folder='/content/drive/MyDrive/SavedImage/networks/111100000000111/checkpoint.pt'
        ###model = torch.load('/content/drive/MyDrive/SavedImage/AgmentedNetwork/111100000010001/checkpoint.pt')

        ###model.eval()
        ###model.freeze()
        ###7th cluster with Acc=95.12%###

        ###model = torch.load('/content/drive/MyDrive/SavedImage/7th/111100000000111/checkpoint.pt')
        ###model.eval()
        ###model.freeze()
        ###
        ##model.unfreeze()

        #history = fit_one_cycle(epochs, max_lr, model, train_ld,train_ld,,grad_clip=grad_clip,weight_decay=weight_decay,opt_func=opt_func)
        #img, label = td[0]
        #plt.imshow(img[0], cmap='gray')
        #print('Label:', label )
        #print('Label:', label, ', Predicted:', predict_image(img, model))
        #print("The state dict keys: \n\n", model.state_dict().keys())
        network_saving_folder=network_dir+'/'+str(Classified)
        print("network saving folder",network_saving_folder)
        if not (os.path.isdir(network_saving_folder)):
            Path(network_saving_folder).mkdir(parents=True, exist_ok=True)
        else:
            shutil.rmtree(network_saving_folder)
            Path(network_saving_folder).mkdir(parents=True, exist_ok=True)
        network_saving_folder=network_saving_folder+'/'+'checkpoint.pt'
        #network_saving_folder=network_saving_folder
        #torch.save(model.state_dict(), network_saving_folder)
        #torch.save(model, network_saving_folder)
        history = fit_one_cycle(epochs, max_lr, model, train_ld,validation_ld,network_saving_folder,grad_clip=grad_clip,weight_decay=weight_decay,opt_func=opt_func)
        History_of_All.append(history)

        #output = open(network_saving_folder, mode="wb")
        #torch.save(model, output)
        #output.close()
        #history = [evaluate(model, train_ld)]
        #print(history)
        #summary(model, input_size=(3, 227, 227))
        #print(len(valid_ld))
        #valid_ld = DeviceDataLoader(valid_ld, device)

##Configure mapping between image path and lable of the image

def create_Labels_for_every_image_as_standard1(source_dir1,file_list_order):

    #imgs_path = "/content/drive/MyDrive/Lucia2/3/"

    dir1=source_dir1
    #####################################

    #img_for=".jpg"
    #imgTag= "images-"
    #imgTag= "Image"

    files_list=[]
    for file in os.listdir(dir1):
        if file.endswith("jpg"):
            files_list.append(file)
    print("our function")
    print(len(files_list))
    #####################################
    ###dir2=source_dir2
    #file_list = glob.glob(imgs_path + "*")
    #print(file_list)
    data = []
    for i in range(0,len(file_list_order)):
        ###img_path1=os.path.join(dir1,images_list_file[i])
        img_path1 = os.path.join(dir1,file_list_order[i])
        ###img_path2=os.path.join(dir2,images_list_file[i])
        #z='{0:011b}'.format(ClassifiedLabel[i])
        #Classified='1111'+z
        z='{0:016b}'.format(i)
        Classified='1111'+z
        data.append([img_path1, Classified])
        ###data.append([img_path2, Classified])
            #print([img_path, 0])

    return data

Labels_of_every_image=create_Labels_for_every_image_as_standard1('/content/drive/MyDrive/berlin_halenseestrasse/berlin_halenseestrasse_2',file_order2)



print(len(file_order2))

print(Labels_of_every_image[66])

def CreateFolderByLabel_for_Every_Image_for_standard_Dataset1(listofPathAndLabel,dir):
    #train_folder=dir+'trainEveryImage'
    train_folder=dir

    classfolder=''
    for img_path, class_name in listofPathAndLabel:
        #print([img_path,class_name])
        #print(train_folder)
        classfolder=train_folder+'/'+str(class_name)
        #print(classfolder)
        if not (os.path.isdir(classfolder)):
            Path(classfolder).mkdir(parents=True, exist_ok=True)
        #img = cv2.imread(img_path)
        #img = cv2.resize(img, (227,227))
        img = cv2.imread(img_path,0)
        img = cv2.resize(img, (227,227))
        #img = cv2.imread(img_path,0)
        #img = cv2.resize(img, (227,227))
        #convertedImage = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB) #convert bayer to BGR as OpenCv
        #im_rgb = cv2.cvtColor(convertedImage, cv2.COLOR_BGR2RGB) #convert BGR to RGB
        #im_rgb=convertedImage
        filename = img_path.split("/")[-1]
        ###folderName = img_path.split("/")[-2]
        #if (os.path.exists(classfolder+'/'+filename)):

        os.chdir(classfolder)
        cv2.imwrite(filename, img)
    return train_folder
        #print(os.listdir(directory))

#dataset = createMappingForDataSet()
#Labels_of_every_image=create_Labels_for_every_image_as_standard()
###Labels_of_every_image=create_Labels_for_every_image_as_standard1('/content/drive/MyDrive/Nordland4','/content/drive/MyDrive/Nordland5',txtlist)
Labels_of_every_image=create_Labels_for_every_image_as_standard1('/content/drive/MyDrive/berlin_halenseestrasse/berlin_halenseestrasse_2',file_order2)
#dir='/content/drive/MyDrive/SavedImage/standard'
#dir='/content/drive/MyDrive/NordlandSavedImages/standard'
###dir='/content/drive/MyDrive/NordlandSavedImages/standardGrey'
dir='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/standard'
TRAIN_DIR_For_Every_Image_as_standard=CreateFolderByLabel_for_Every_Image_for_standard_Dataset1(Labels_of_every_image,dir)

TRAIN_DIR_For_Every_Image_as_standard='/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/standard'

train_data_standard = ImageFolder(TRAIN_DIR_For_Every_Image_as_standard, train_tfms)

print(len(train_data_standard))

print(len(train_data_standard.class_to_idx))

def Calculate_Starting_of_All_Cluster_As_list(no_of_clusters,path_of_clusters):
    list_for_all_clusters=[]
    #z='{0:011b}'.format(0)
    #Classified='1111'+z
    #classfolder="/content/drive/MyDrive/SavedImage/train"+'/'+str(Classified)
    #first_size=Count_Number_Of_Images_In_Folder(classfolder)
    #path_of_colored_images="/content/drive/MyDrive/SavedImage/colored/"
    #path_of_clusters="/content/drive/MyDrive/SavedImage/train"
    for i in range(0,no_of_clusters):
        expected_cluster=i
        n,size=Count_Number_Of_Images_In_All_Folders(path_of_clusters,expected_cluster)
        list_for_all_clusters.append(int(n/6))
    return list_for_all_clusters

Starting_of_All_Clusters=Calculate_Starting_of_All_Cluster_As_list(10,"/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImages/train")
print(Starting_of_All_Clusters[9])

def loading_all_sub_networks(net_dir,Gen_dataset):
    List_of_All_models=[]
    for i in range(0,len(Gen_dataset.classes)):
    #for i in range(20,len(General_Train_DataSet.classes)):
        #Pathloss='/content/drive/MyDrive/SavedImage/train'
        z='{0:011b}'.format(i)
        Classified='1111'+z

        network_saving_folder=net_dir+'/'+str(Classified)+'/'+'checkpoint.pt'
        model1 = torch.load(network_saving_folder)
        model1.eval()
        List_of_All_models.append(model1)
    return List_of_All_models

Cluster_of_all_sub_networks=loading_all_sub_networks(network_dir,General_Train_DataSet)

print(len(Cluster_of_all_sub_networks))

def predict_image_with_two_networks_With_Top9(img, main_model,cluster_of_netwoks,starting_of_clusters):
    xb = img.unsqueeze(0)
    xb = to_device(xb, device)
    yb = main_model(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    copied=yb.detach().numpy()
    #print(copied)
    a = copied.argsort()[0][-7:][::-1]
    #print("top 3 cluster",a)
    expectionList=[]
    #_, preds  = torch.max(yb, dim=1)
    #print('expected cluster:', preds[0].item())
    expected_cluster1=cluster_of_netwoks[a[0]]
    yb = expected_cluster1(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    copied=yb.detach().numpy()
    #print(copied)
    first = copied.argsort()[0][-3:][::-1]
    #_, preds1  = torch.max(yb, dim=1)
    #print('expected rank in cluster:', preds1[0].item())
    #print('start of cluster:', starting_of_clusters[preds[0].item()])
    #print("prediction inside 1cluster",preds1[0].item(),"start of cluster",starting_of_clusters[a[0]])
    #expectation1=preds1[0].item()+starting_of_clusters[a[0]]
    expectionList.append(first[0]+starting_of_clusters[a[0]])
    expectionList.append(first[1]+starting_of_clusters[a[0]])
    expectionList.append(first[2]+starting_of_clusters[a[0]])
    #expectionList.append(first[3]+starting_of_clusters[a[0]])
    #expectionList.append(first[4]+starting_of_clusters[a[0]])
    #expectionList.append(first[5]+starting_of_clusters[a[0]])
    #expectionList.append(first[3]+starting_of_clusters[a[0]])
    ##
    expected_cluster2=cluster_of_netwoks[a[1]]
    yb = expected_cluster2(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    copied=yb.detach().numpy()
    #print(copied)
    second = copied.argsort()[0][-3:][::-1]
    #_, preds2  = torch.max(yb, dim=1)
    #print('expected rank in cluster:', preds1[0].item())
    #print('start of cluster:', starting_of_clusters[preds[0].item()])
    #print("prediction inside 2cluster",preds2[0].item(),"start of cluster",starting_of_clusters[a[1]])
    #expectation2=preds2[0].item()+starting_of_clusters[a[1]]
    expectionList.append(second[0]+starting_of_clusters[a[1]])
    expectionList.append(second[1]+starting_of_clusters[a[1]])
    expectionList.append(second[2]+starting_of_clusters[a[1]])
    #expectionList.append(second[3]+starting_of_clusters[a[1]])
    #expectionList.append(second[4]+starting_of_clusters[a[1]])
    ###
    expected_cluster3=cluster_of_netwoks[a[2]]
    yb = expected_cluster3(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    copied=yb.detach().numpy()
    #print(copied)
    third = copied.argsort()[0][-3:][::-1]
    #_, preds3  = torch.max(yb, dim=1)
    #print('expected rank in cluster:', preds1[0].item())
    #print('start of cluster:', starting_of_clusters[preds[0].item()])
    #print("prediction inside 3cluster",preds3[0].item(),"start of cluster",starting_of_clusters[a[2]])

    expectionList.append(third[0]+starting_of_clusters[a[2]])
    expectionList.append(third[1]+starting_of_clusters[a[2]])
    expectionList.append(third[2]+starting_of_clusters[a[2]])
    #expectionList.append(third[3]+starting_of_clusters[a[2]])
    #expectionList.append(third[4]+starting_of_clusters[a[2]])
    ###
    ###expected_cluster4=cluster_of_netwoks[a[3]]
    ###yb = expected_cluster4(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    ###copied=yb.detach().numpy()
    #print(copied)
    ###fourth = copied.argsort()[0][-3:][::-1]
    #_, preds3  = torch.max(yb, dim=1)
    #print('expected rank in cluster:', preds1[0].item())
    #print('start of cluster:', starting_of_clusters[preds[0].item()])
    #print("prediction inside 3cluster",preds3[0].item(),"start of cluster",starting_of_clusters[a[2]])

    ###expectionList.append(fourth[0]+starting_of_clusters[a[3]])
    ###expectionList.append(fourth[1]+starting_of_clusters[a[3]])
    ###expectionList.append(fourth[2]+starting_of_clusters[a[3]])
    #expectionList.append(fourth[3]+starting_of_clusters[a[3]])
    #expectionList.append(fourth[4]+starting_of_clusters[a[3]])
    ###
    ###expected_cluster5=cluster_of_netwoks[a[4]]
    ###yb = expected_cluster5(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    ###copied=yb.detach().numpy()
    #print(copied)
    ###fifth = copied.argsort()[0][-3:][::-1]
    #_, preds3  = torch.max(yb, dim=1)
    #print('expected rank in cluster:', preds1[0].item())
    #print('start of cluster:', starting_of_clusters[preds[0].item()])
    #print("prediction inside 3cluster",preds3[0].item(),"start of cluster",starting_of_clusters[a[2]])
    #expectation3=preds3[0].item()+starting_of_clusters[a[2]]
    ###expectionList.append(fifth[0]+starting_of_clusters[a[4]])
    ###expectionList.append(fifth[1]+starting_of_clusters[a[4]])
    ###expectionList.append(fifth[2]+starting_of_clusters[a[4]])
    ###
    ###expected_cluster6=cluster_of_netwoks[a[5]]
    ###yb = expected_cluster6(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    ###copied=yb.detach().numpy()
    #print(copied)
    ###sixth = copied.argsort()[0][-3:][::-1]
    #_, preds3  = torch.max(yb, dim=1)
    #print('expected rank in cluster:', preds1[0].item())
    #print('start of cluster:', starting_of_clusters[preds[0].item()])
    #print("prediction inside 3cluster",preds3[0].item(),"start of cluster",starting_of_clusters[a[2]])
    #expectation3=preds3[0].item()+starting_of_clusters[a[2]]
    ###expectionList.append(sixth[0]+starting_of_clusters[a[5]])
    ###expectionList.append(sixth[1]+starting_of_clusters[a[5]])
    ###expectionList.append(sixth[2]+starting_of_clusters[a[5]])
    ###
    ###expected_cluster7=cluster_of_netwoks[a[6]]
    ###yb = expected_cluster7(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    ###copied=yb.detach().numpy()
    #print(copied)
    ###seventh = copied.argsort()[0][-3:][::-1]
    #_, preds3  = torch.max(yb, dim=1)
    #print('expected rank in cluster:', preds1[0].item())
    #print('start of cluster:', starting_of_clusters[preds[0].item()])
    #print("prediction inside 3cluster",preds3[0].item(),"start of cluster",starting_of_clusters[a[2]])
    #expectation3=preds3[0].item()+starting_of_clusters[a[2]]
    ###expectionList.append(seventh[0]+starting_of_clusters[a[6]])
    ###expectionList.append(seventh[1]+starting_of_clusters[a[6]])
    ###expectionList.append(seventh[2]+starting_of_clusters[a[6]])
    return expectionList #[expectation1,expectation2,expectation3]

#img, label = train_ds[820]
#img,label=train_data[1500]
#img,label=Dataset_of_All[784]
#img,lable=train_data_standard[1687]
img,label=train_data_standard[66]
#plt.imshow(img[0], cmap='gray')
#print('Label:', label, ', Predicted:', predict_image(img, Cluster_of_all_sub_networks[25]))
print('Label:', label)
print('--------------------------------------------')
#print('Label:', label, ', Predicted:', predict_image(img, model_master))
#print('Label:', label, ', Predicted:', predict_image_with_Top3_Second(img,model_master))
#print('Label:', label, ', Predicted:',predict_image_with_two_networks(img, model_master,Cluster_of_all_sub_networks,Starting_of_All_Clusters))

print('Label:', label, ', Predicted:',predict_image_with_two_networks_With_Top9(img, model_master,Cluster_of_all_sub_networks,Starting_of_All_Clusters))

def predict_image_with_two_networks_new(img, main_model,cluster_of_netwoks,starting_of_clusters):

    ####################################################
    xb = img.unsqueeze(0)
    xb = to_device(xb, device)
    yb = main_model(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    copied=yb.detach().numpy()
    #print(copied)
    a = copied.argsort()[0][-3:][::-1]
    print("top 3 cluster",a)
    expectionList=[]
    #_, preds  = torch.max(yb, dim=1)
    #print('expected cluster:', preds[0].item())
    expected_cluster1=cluster_of_netwoks[a[0]]
    yb = expected_cluster1(xb)
    #print(yb.shape)
    #z=yb[0][:]
    #z=z.tolist()
    #print(len(z) )
    copied=yb.detach().numpy()
    print("copied",copied[0])
    first = copied.argsort()[0][-3:][::-1]
    print("first",first)
    if(len(copied[0])>3):
       first = copied.argsort()[0][-4:][::-1]
       expectionList.append(first[0]+starting_of_clusters[a[0]])
       expectionList.append(first[1]+starting_of_clusters[a[0]])
       expectionList.append(first[2]+starting_of_clusters[a[0]])
       expectionList.append(first[3]+starting_of_clusters[a[0]])
       print("fourth option")
       print("value1",copied[0][first[0]])
       print("value2",copied[0][first[1]])
       print("value3",copied[0][first[2]])
       print("value3",copied[0][first[3]])
    elif(len(copied[0])==3):
       first = copied.argsort()[0][-3:][::-1]
       expectionList.append(first[0]+starting_of_clusters[a[0]])
       expectionList.append(first[1]+starting_of_clusters[a[0]])
       expectionList.append(first[2]+starting_of_clusters[a[0]])
       print("three option")
       print("value1",copied[0][first[0]])
       print("value2",copied[0][first[1]])
       print("value3",copied[0][first[2]])
       #print("value3",copied[first[3]])
    elif((len(copied[0]))==2):
      first = copied.argsort()[0][-2:][::-1]
      expectionList.append(first[0]+starting_of_clusters[a[0]])
      expectionList.append(first[1]+starting_of_clusters[a[0]])
      print("two option")
      print("value1",copied[0][first[0]])
      print("value2",copied[0][first[1]])

      ##expectionList.append(first[2]+starting_of_clusters[a[0]])
    else:
       ###first = copied.argsort()[0][-3:][::-1]
       _, preds1  = torch.max(yb, dim=1)
       expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
       print("one option")
       print("value1",copied[0][preds1[0].item()])

    ###################################################
    return expectionList

#img, label = train_ds[820]
#img,label=train_data[1500]
#img,label=Dataset_of_All[784]
#img,lable=train_data_standard[1687]
img,label=train_data_standard[66]

print('Label:', label)
print('--------------------------------------------')

print('Label:', label, ', Predicted:',predict_image_with_two_networks_new(img, model_master,Cluster_of_all_sub_networks,Starting_of_All_Clusters))

zeroz=0
ones=0
#for i in range(0,len(train_data_standard)):
for i in range(0,67):
#for i in range(0,10):

    img,label=train_data_standard[i]

    expect=predict_image_with_two_networks_new(img, model_master,Cluster_of_all_sub_networks,Starting_of_All_Clusters)
    #print("label",label,"expected",expect)
    plus=[]

    for x in expect:
        plus.append(x+1)
        plus.append(x-1)
        plus.append(x+2)
        plus.append(x-2)
        plus.append(x+3)
        plus.append(x-3)
        plus.append(x+4)
        plus.append(x-4)
        plus.append(x+5)
        plus.append(x-5)
    if(label in expect)or(label in plus):
        ones=ones+1

    else:
        zeroz=zeroz+1
    #print('Label:', label, ', Predicted:',predict_image_with_two_networks_With_Top9(img, model_master,Cluster_of_all_sub_networks,Starting_of_All_Clusters)+plus)
print(ones)
print(zeroz)

#img, label = train_ds[820]
#img,label=train_data[1500]
#img,label=Dataset_of_All[784]
#img,lable=train_data_standard[1687]
img,label=train_data_standard[66]

#plt.imshow(img[0], cmap='gray')
#print('Label:', label, ', Predicted:', predict_image(img, Cluster_of_all_sub_networks[25]))
print('Label:', label)
print('--------------------------------------------')
#print('Label:', label, ', Predicted:', predict_image(img, model_master))
#print('Label:', label, ', Predicted:', predict_image_with_Top3_Second(img,model_master))
#print('Label:', label, ', Predicted:',predict_image_with_two_networks(img, model_master,Cluster_of_all_sub_networks,Starting_of_All_Clusters))

print('Label:', label, ', Predicted:',predict_image_with_two_networks_new(img, model_master,Cluster_of_all_sub_networks,Starting_of_All_Clusters))
plus=[]
#expect =predict_image_with_two_networks(img, model_master,Cluster_of_all_sub_networks,Starting_of_All_Clusters)
#for x in expect:
#    plus.append(x+1)
#    plus.append(x-1)
#    plus.append(x+2)
#    plus.append(x-2)
#    plus.append(x+3)
#    plus.append(x-3)
#    plus.append(x+4)
#    plus.append(x-4)

import time
from PIL import Image
import numpy as np
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import scipy
from sklearn.metrics.pairwise import cosine_similarity
#batch in val_loader
def Labels_outputs_producing_for_multi(test_loader, main_model,cluster_of_netwoks,starting_of_clusters):
    labelY=[]
    predicts=[]
    PredictedLabel=[]
    Timing=[]
    Find=[]
    Output=[]
    counter=0
    for batch in test_loader:
        images, labels = batch
        counter=counter+1
        if(counter % 50==0):
            print("counter",counter)
        for i in range(0,len(labels)):
            xb = images[i].unsqueeze(0)
            xb = to_device(xb, device)
            ####################################################
            start_time = time.time()
            yb = main_model(xb)


            #_, preds  = torch.max(yb, dim=1)
            ###
            copied=yb.detach().numpy()
            #print(copied)
            a = copied.argsort()[0][-7:][::-1]
            #print("top 3 cluster",a)
            expectionList=[]
            valueList=[]
            ###
            #print('expected cluster:', preds[0].item())
            #expected_cluster=cluster_of_netwoks[preds[0].item()]
            #yb = expected_cluster(xb)
            #z=yb[0][:]
            #z=z.tolist()
            ###
            expected_cluster1=cluster_of_netwoks[a[0]]
            yb = expected_cluster1(xb)
            _, preds1  = torch.max(yb, dim=1)
            #expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
            first0=preds1[0].item()
            copied=yb.detach().numpy()
            #print(copied)
            #######################################
            if(len(copied[0])>2):
               first = copied.argsort()[0][-3:][::-1]
               expectionList.append(first[0]+starting_of_clusters[a[0]])
               expectionList.append(first[1]+starting_of_clusters[a[0]])
               expectionList.append(first[2]+starting_of_clusters[a[0]])
               valueList.append(copied[0][first[0]])
               valueList.append(copied[0][first[1]])
               valueList.append(copied[0][first[2]])
            elif((len(copied[0]))==2):
               first = copied.argsort()[0][-2:][::-1]
               expectionList.append(first[0]+starting_of_clusters[a[0]])
               expectionList.append(first[1]+starting_of_clusters[a[0]])
               valueList.append(copied[0][first[0]])
               valueList.append(copied[0][first[1]])

               ##expectionList.append(first[2]+starting_of_clusters[a[0]])
            else:
                ###first = copied.argsort()[0][-3:][::-1]
                _, preds1  = torch.max(yb, dim=1)
                expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
                valueList.append(copied[0][preds1[0].item()])

            ###########################################
            expected_cluster2=cluster_of_netwoks[a[1]]
            yb = expected_cluster2(xb)
            _, preds1  = torch.max(yb, dim=1)
            #expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
            first1=preds1[0].item()
            copied=yb.detach().numpy()
            #print(copied)
            #######################################
            if(len(copied[0])>2):
               second = copied.argsort()[0][-3:][::-1]
               expectionList.append(second[0]+starting_of_clusters[a[1]])
               expectionList.append(second[1]+starting_of_clusters[a[1]])
               expectionList.append(second[2]+starting_of_clusters[a[1]])
               valueList.append(copied[0][second[0]])
               valueList.append(copied[0][second[1]])
               valueList.append(copied[0][second[2]])
            elif((len(copied[0]))==2):
               second = copied.argsort()[0][-2:][::-1]
               expectionList.append(second[0]+starting_of_clusters[a[1]])
               expectionList.append(second[1]+starting_of_clusters[a[1]])
               valueList.append(copied[0][second[0]])
               valueList.append(copied[0][second[1]])

               ##expectionList.append(first[2]+starting_of_clusters[a[0]])
            else:
                ###first = copied.argsort()[0][-3:][::-1]
                _, preds1  = torch.max(yb, dim=1)
                expectionList.append(preds1[0].item()+starting_of_clusters[a[1]])
                valueList.append(copied[0][preds1[0].item()])

            ###########################################
            expected_cluster3=cluster_of_netwoks[a[2]]
            yb = expected_cluster3(xb)
            _, preds1  = torch.max(yb, dim=1)
            #expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
            first2=preds1[0].item()
            copied=yb.detach().numpy()
            #print(copied)
            #######################################
            if(len(copied[0])>2):
               third = copied.argsort()[0][-3:][::-1]
               expectionList.append(third[0]+starting_of_clusters[a[2]])
               expectionList.append(third[1]+starting_of_clusters[a[2]])
               expectionList.append(third[2]+starting_of_clusters[a[2]])
               valueList.append(copied[0][third[0]])
               valueList.append(copied[0][third[1]])
               valueList.append(copied[0][third[2]])
            elif((len(copied[0]))==2):
               third = copied.argsort()[0][-2:][::-1]
               expectionList.append(third[0]+starting_of_clusters[a[2]])
               expectionList.append(third[1]+starting_of_clusters[a[2]])
               valueList.append(copied[0][third[0]])
               valueList.append(copied[0][third[1]])

               ##expectionList.append(first[2]+starting_of_clusters[a[0]])
            else:
                ###first = copied.argsort()[0][-3:][::-1]
                _, preds1  = torch.max(yb, dim=1)
                expectionList.append(preds1[0].item()+starting_of_clusters[a[2]])
                valueList.append(copied[0][preds1[0].item()])

            ###########################################
            expected_cluster4=cluster_of_netwoks[a[3]]
            yb = expected_cluster4(xb)
            _, preds1  = torch.max(yb, dim=1)
            #expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
            first3=preds1[0].item()
            copied=yb.detach().numpy()
            #print(copied)
            #######################################
            if(len(copied[0])>2):
               fourth = copied.argsort()[0][-3:][::-1]
               expectionList.append(fourth[0]+starting_of_clusters[a[3]])
               expectionList.append(fourth[1]+starting_of_clusters[a[3]])
               expectionList.append(fourth[2]+starting_of_clusters[a[3]])
               valueList.append(copied[0][fourth[0]])
               valueList.append(copied[0][fourth[1]])
               valueList.append(copied[0][fourth[2]])
            elif((len(copied[0]))==2):
               fourth = copied.argsort()[0][-2:][::-1]
               expectionList.append(fourth[0]+starting_of_clusters[a[3]])
               expectionList.append(fourth[1]+starting_of_clusters[a[3]])
               valueList.append(copied[0][fourth[0]])
               valueList.append(copied[0][fourth[1]])

               ##expectionList.append(first[2]+starting_of_clusters[a[0]])
            else:
                ###first = copied.argsort()[0][-3:][::-1]
                _, preds1  = torch.max(yb, dim=1)
                expectionList.append(preds1[0].item()+starting_of_clusters[a[3]])
                valueList.append(copied[0][preds1[0].item()])

            ###########################################
            expected_cluster5=cluster_of_netwoks[a[4]]
            yb = expected_cluster5(xb)
            _, preds1  = torch.max(yb, dim=1)
            #expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
            first4=preds1[0].item()
            copied=yb.detach().numpy()
            #print(copied)
            #######################################
            if(len(copied[0])>2):
               fifth = copied.argsort()[0][-3:][::-1]
               expectionList.append(fifth[0]+starting_of_clusters[a[4]])
               expectionList.append(fifth[1]+starting_of_clusters[a[4]])
               expectionList.append(fifth[2]+starting_of_clusters[a[4]])
               valueList.append(copied[0][fifth[0]])
               valueList.append(copied[0][fifth[1]])
               valueList.append(copied[0][fifth[2]])
            elif((len(copied[0]))==2):
               fifth = copied.argsort()[0][-2:][::-1]
               expectionList.append(fifth[0]+starting_of_clusters[a[4]])
               expectionList.append(fifth[1]+starting_of_clusters[a[4]])
               valueList.append(copied[0][fifth[0]])
               valueList.append(copied[0][fifth[1]])

               ##expectionList.append(first[2]+starting_of_clusters[a[0]])
            else:
                ###first = copied.argsort()[0][-3:][::-1]
                _, preds1  = torch.max(yb, dim=1)
                expectionList.append(preds1[0].item()+starting_of_clusters[a[4]])
                valueList.append(copied[0][preds1[0].item()])

            ###########################################
            expected_cluster6=cluster_of_netwoks[a[5]]
            yb = expected_cluster6(xb)
            _, preds1  = torch.max(yb, dim=1)
            #expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
            first5=preds1[0].item()
            copied=yb.detach().numpy()
            #print(copied)
            #######################################
            if(len(copied[0])>2):
               sixth = copied.argsort()[0][-3:][::-1]
               expectionList.append(sixth[0]+starting_of_clusters[a[5]])
               expectionList.append(sixth[1]+starting_of_clusters[a[5]])
               expectionList.append(sixth[2]+starting_of_clusters[a[5]])
               valueList.append(copied[0][sixth[0]])
               valueList.append(copied[0][sixth[1]])
               valueList.append(copied[0][sixth[2]])
            elif((len(copied[0]))==2):
               sixth = copied.argsort()[0][-2:][::-1]
               expectionList.append(sixth[0]+starting_of_clusters[a[5]])
               expectionList.append(sixth[1]+starting_of_clusters[a[5]])
               valueList.append(copied[0][sixth[0]])
               valueList.append(copied[0][sixth[1]])

               ##expectionList.append(first[2]+starting_of_clusters[a[0]])
            else:
                ###first = copied.argsort()[0][-3:][::-1]
                _, preds1  = torch.max(yb, dim=1)
                expectionList.append(preds1[0].item()+starting_of_clusters[a[5]])
                valueList.append(copied[0][preds1[0].item()])

            ###########################################
            expected_cluster7=cluster_of_netwoks[a[6]]
            yb = expected_cluster7(xb)
            _, preds1  = torch.max(yb, dim=1)
            #expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
            first6=preds1[0].item()
            copied=yb.detach().numpy()
            #print(copied)
            #######################################
            if(len(copied[0])>2):
               seventh = copied.argsort()[0][-3:][::-1]
               expectionList.append(seventh[0]+starting_of_clusters[a[6]])
               expectionList.append(seventh[1]+starting_of_clusters[a[6]])
               expectionList.append(seventh[2]+starting_of_clusters[a[6]])
               valueList.append(copied[0][seventh[0]])
               valueList.append(copied[0][seventh[1]])
               valueList.append(copied[0][seventh[2]])
            elif((len(copied[0]))==2):
               seventh = copied.argsort()[0][-2:][::-1]
               expectionList.append(seventh[0]+starting_of_clusters[a[6]])
               expectionList.append(seventh[1]+starting_of_clusters[a[6]])
               valueList.append(copied[0][seventh[0]])
               valueList.append(copied[0][seventh[1]])

               ##expectionList.append(first[2]+starting_of_clusters[a[0]])
            else:
                ###first = copied.argsort()[0][-3:][::-1]
                _, preds1  = torch.max(yb, dim=1)
                expectionList.append(preds1[0].item()+starting_of_clusters[a[6]])
                valueList.append(copied[0][preds1[0].item()])

            ###########################################
            ###z=yb[0][:]
            ###z=z.tolist()
            ###listofzeros = [0] * len(train_data_standard.classes)
            ###for k in range(0,len(z)):
            ###    listofzeros[k+starting_of_clusters[a[0]]]= z[k]
            ###
            ########Arg Max Method##############
            arg_max=-1
            arg_index=-1
            output_max=0
            for j in range(0,len(expectionList)):
                #img,label=train_data_standard[76]
                img2,label=train_data_standard[expectionList[j]]
                dog_array1 = np.array(images[i])
                dog_array2 = np.array(img2)
                dog_array1 = dog_array1.flatten()


                dog_array2 = dog_array2.flatten()
                dog_array1 = dog_array1/255
                dog_array2 = dog_array2/255


                similarity = 1-spatial.distance.cosine(dog_array1, dog_array2)
                #similarity = dot(dog_array1, dog_array2)/(norm(dog_array1)*norm(dog_array2))
                #similarity=cosine_similarity(dog_array1, dog_array2)

                #similarity=cosine
                if (similarity>arg_max):
                   arg_max=similarity
                   arg_index=expectionList[j]
                   output_max=valueList[j]
            #print(similarity)
            ####################################
            plus=[]
            if (arg_index>-1):
                plus.append(arg_index)
                plus.append(arg_index+1)
                plus.append(arg_index-1)
                plus.append(arg_index+2)
                plus.append(arg_index-2)
                #plus.append(arg_index+3)
                #plus.append(arg_index-3)
                #plus.append(arg_index+4)
                #plus.append(arg_index-4)
                #plus.append(arg_index+5)
                #plus.append(arg_index-5)

            if(labels[i].item() in plus):
                #expectation=labels[i].item()
                expectation=arg_index
                Find.append(1)
            else:
                #expectation=first0+starting_of_clusters[a[0]]
                expectation=arg_index
                Find.append(0)
            ###
            tie=(time.time() - start_time)
            Timing.append(tie)
            labelY.append(labels[i].item())
            #predicts.append(listofzeros)
            PredictedLabel.append(expectation)
            if (arg_max>-1):
                Output.append(output_max)
            ######################################################


    return labelY,PredictedLabel,Timing,Find,Output

#Truelabel1,poba1,PLabel1,WTime1=Labels_outputs_producing(train_ld,model)

"""# ----------------------------____________________________________________"""

import time
from PIL import Image
import numpy as np
from scipy import spatial
from numpy import dot
from numpy.linalg import norm
import scipy
from sklearn.metrics.pairwise import cosine_similarity
#batch in val_loader
def Labels_outputs_producing_for_multi(test_loader, main_model,cluster_of_netwoks,starting_of_clusters):
    labelY=[]
    predicts=[]
    PredictedLabel=[]
    Timing=[]
    Find=[]
    Output=[]
    counter=0
    for batch in test_loader:
        images, labels = batch
        counter=counter+1
        if(counter % 50==0):
            print("counter",counter)
        for i in range(0,len(labels)):
            xb = images[i].unsqueeze(0)
            xb = to_device(xb, device)
            ####################################################
            start_time = time.time()
            yb = main_model(xb)


            #_, preds  = torch.max(yb, dim=1)
            ###
            copied=yb.detach().numpy()
            #print(copied)
            a = copied.argsort()[0][-7:][::-1]
            #print("top 3 cluster",a)
            expectionList=[]
            valueList=[]
            ###
            #print('expected cluster:', preds[0].item())
            #expected_cluster=cluster_of_netwoks[preds[0].item()]
            #yb = expected_cluster(xb)
            #z=yb[0][:]
            #z=z.tolist()
            ###
            expected_cluster1=cluster_of_netwoks[a[0]]
            yb = expected_cluster1(xb)
            _, preds1  = torch.max(yb, dim=1)
            #expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
            first0=preds1[0].item()
            copied=yb.detach().numpy()
            #print(copied)
            #######################################
            if(len(copied[0])>2):
               first = copied.argsort()[0][-3:][::-1]
               expectionList.append(first[0]+starting_of_clusters[a[0]])
               expectionList.append(first[1]+starting_of_clusters[a[0]])
               expectionList.append(first[2]+starting_of_clusters[a[0]])
               valueList.append(copied[0][first[0]])
               valueList.append(copied[0][first[1]])
               valueList.append(copied[0][first[2]])
            elif((len(copied[0]))==2):
               first = copied.argsort()[0][-2:][::-1]
               expectionList.append(first[0]+starting_of_clusters[a[0]])
               expectionList.append(first[1]+starting_of_clusters[a[0]])
               valueList.append(copied[0][first[0]])
               valueList.append(copied[0][first[1]])

               ##expectionList.append(first[2]+starting_of_clusters[a[0]])
            else:
                ###first = copied.argsort()[0][-3:][::-1]
                _, preds1  = torch.max(yb, dim=1)
                expectionList.append(preds1[0].item()+starting_of_clusters[a[0]])
                valueList.append(copied[0][preds1[0].item()])

            ###########################################
            ###z=yb[0][:]
            ###z=z.tolist()
            ###listofzeros = [0] * len(train_data_standard.classes)
            ###for k in range(0,len(z)):
            ###    listofzeros[k+starting_of_clusters[a[0]]]= z[k]
            ###
            ########Arg Max Method##############
            arg_max=-1
            arg_index=-1
            output_max=0
            for j in range(0,len(expectionList)):
                #img,label=train_data_standard[76]
                img2,label=train_data_standard[expectionList[j]]
                dog_array1 = np.array(images[i])
                dog_array2 = np.array(img2)
                dog_array1 = dog_array1.flatten()


                dog_array2 = dog_array2.flatten()
                dog_array1 = dog_array1/255
                dog_array2 = dog_array2/255


                similarity = 1-spatial.distance.cosine(dog_array1, dog_array2)
                #similarity = dot(dog_array1, dog_array2)/(norm(dog_array1)*norm(dog_array2))
                #similarity=cosine_similarity(dog_array1, dog_array2)

                #similarity=cosine
                if (similarity>arg_max):
                   arg_max=similarity
                   arg_index=expectionList[j]
                   output_max=valueList[j]
            #print(similarity)
            ####################################
            plus=[]
            if (arg_index>-1):
                plus.append(arg_index)
                plus.append(arg_index+1)
                plus.append(arg_index-1)
                plus.append(arg_index+2)
                plus.append(arg_index-2)
                #plus.append(arg_index+3)
                #plus.append(arg_index-3)
                #plus.append(arg_index+4)
                #plus.append(arg_index-4)

            if(labels[i].item() in plus):
                #expectation=labels[i].item()
                expectation=arg_index
                Find.append(1)
            else:
                #expectation=first0+starting_of_clusters[a[0]]
                expectation=arg_index
                Find.append(0)
            ###
            tie=(time.time() - start_time)
            Timing.append(tie)
            labelY.append(labels[i].item())
            #predicts.append(listofzeros)
            PredictedLabel.append(expectation)
            if (arg_max>-1):
                Output.append(output_max)
            ######################################################


    return labelY,PredictedLabel,Timing,Find,Output

#Truelabel1,poba1,PLabel1,WTime1=Labels_outputs_producing(train_ld,model)

dataset_size = len(train_data_standard)
#print(dataset_size)
dataset_indices = list(range(dataset_size))
#print(dataset_indices)
#np.random.shuffle(dataset_indices)
#print(dataset_indices)

train_idx= dataset_indices[:]
train_sampler = SubsetRandomSampler(train_idx)

train_ld_multi = DataLoader(dataset=train_data_standard, shuffle=False, batch_size=32, sampler=train_sampler)

Truelabel,PLabel,WTime,found,output=[],[],[],[],[]
Truelabel,PLabel,WTime,found,output=Labels_outputs_producing_for_multi(train_ld_multi ,model_master,Cluster_of_all_sub_networks,Starting_of_All_Clusters)

sum=0
for i in range(0,len(found)):
    sum=sum+found[i]
print(sum)

for i in range(0,len(Truelabel)):
    print("True label",Truelabel[i],"predicted label",PLabel[i],"close enough",found[i],"arg max",output[i])

import statistics
z=statistics.mean(WTime)
s=statistics.stdev(WTime)
#print('Average of Response Time is %f milisecond for  test ' %(meanTime1*1000))
#meanTime2=statistics.mean(WTime1)
#print('Average of Response Time is %f milisecond for  train' %(meanTime2*1000))
#y=[meanTime1,meanTime2]
#z=statistics.mean(y)
print('Average of Response Time is %f milisecond for  Multi Resnet18 With With 10 Clusters With one Cluster Activate  Berlin HaleSeeStrasse ' %(z*1000))
print('Standard Deviation of Response Time is %f milisecond for  Multi Resnet18 With With 10 Clusters With one Cluster Activate Berlin HaleSeeStrasse ' %(s*1000))

from sklearn.metrics import precision_recall_curve,auc,average_precision_score
import os
import matplotlib.pyplot as plt
import pickle
from pylab import imread,subplot,imshow,show
import matplotlib.gridspec as gridspec


def showAUCPR():

    precision, recall, thresholds = precision_recall_curve(found,output)

    plt.step(recall, precision, alpha=0.2,
                 where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
#        PR_values.append(str(auc(recall, precision)))
    tiTle= "H " + " AUC-PR: " + str(auc(recall, precision))
    plt.grid(True)
       # plt.legend(tiTle)
    plt.title(tiTle,fontsize=8)
    #plt.savefig(os.path.join(resultPath,"AUC-PR"+"_"+str(N)+"_"+str(V)+'.jpg'),dpi=200)
    #plt.close()



showAUCPR()

import csv
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_HalenSeeStrasse-1Active-AUC50-Radius2.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(found,output))

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_HalenSeeStrasse-1Active-AUC50-Radius2.csv', header=None)
df.rename(columns={0: 'label', 1: 'score'}, inplace=True)
df.to_csv('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_HalenSeeStrasse-1Active-AUC50-Radius2.csv', index=False) # save to new csv file

import csv
list_of_label=[]
list_of_score=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_HalenSeeStrasse-1Active-AUC50-Radius2.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label.append(row['label'])
        list_of_score.append(row['score'])
list_of_label=list(map(int, list_of_label))
list_of_score=list(map(float, list_of_score))

from sklearn.metrics import precision_recall_curve,auc,average_precision_score
import os
import matplotlib.pyplot as plt
import pickle
from pylab import imread,subplot,imshow,show
import matplotlib.gridspec as gridspec
import csv
def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

datasetIndex = 4
img_for=".jpg"
#imgTag= "images-"
imgTag= ""


def showAUCPR():
    print("i am be 1")
    files_list=[]
    #datasetPath = os.path.join(dir,dataset)
    #testPath = os.path.join(dir,dataset,testTraverse)
    #referencePath = os.path.join(dir,dataset,referenceTraverse)

    #for file in os.listdir('/content/drive/MyDrive/pickefolder/GardenPoint_Walking_200_128 (1).pkl'):
    #    if file.endswith(".pkl"):
    #        files_list.append(file)
    #    else:
    #        continue
    resultPath = '/content/drive/MyDrive/pickefolder/GardenPoint_Walking_200_128 (1).pkl'
    PR_values = list()
    print("i am be 2")

    Results = load_obj('/content/drive/MyDrive/pickefolder/GardenPoint_Walking_200_128 (1).pkl')



    predictionLabel,predictionScore = list(),list()
    for _,result in Results.items():
        predictionLabel.append(result[1])
        predictionScore.append(result[2])


    #with open('/content/drive/MyDrive/pickefolder/Try.csv', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(zip(predictionLabel,predictionScore))

    precision, recall, thresholds = precision_recall_curve(list_of_label,list_of_score)

    plt.step(recall, precision, alpha=0.2,color="Red",where='post')
    #plt.fill_between(recall, precision, step='post', alpha=0.2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
#        PR_values.append(str(auc(recall, precision)))
    tiTle= "Region-VLAD "  + " AUC-PR: " + str(auc(recall, precision))
    plt.grid(True)
       # plt.legend(tiTle)
    plt.title(tiTle,fontsize=8)
    #plt.savefig(os.path.join(resultPath,"AUC-PR"+"_"+str(N)+"_"+str(V)+'.jpg'),dpi=200)
    print("i am here")
        #plt.close()


showAUCPR()

"""_________________________________________________
___________________________________________________________

# ________________________
"""

from sklearn.metrics import precision_recall_curve,auc,average_precision_score
import os
import matplotlib.pyplot as plt
import pickle
from pylab import imread,subplot,imshow,show
import matplotlib.gridspec as gridspec
import csv
def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

datasetIndex = 4
img_for=".jpg"
#imgTag= "images-"
imgTag= ""


def showAUCPR():
    print("i am be 1")
    files_list=[]
    #datasetPath = os.path.join(dir,dataset)
    #testPath = os.path.join(dir,dataset,testTraverse)
    #referencePath = os.path.join(dir,dataset,referenceTraverse)

    #for file in os.listdir('/content/drive/MyDrive/pickefolder/GardenPoint_Walking_200_128 (1).pkl'):
    #    if file.endswith(".pkl"):
    #        files_list.append(file)
    #    else:
    #        continue
    resultPath = '/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Pickle/berlin_halenseestrasse_400_256.pkl'
    PR_values = list()
    print("i am be 2")

    Results = load_obj('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Pickle/berlin_halenseestrasse_400_256.pkl')



    predictionLabel,predictionScore = list(),list()
    for _,result in Results.items():
        predictionLabel.append(result[1])
        predictionScore.append(result[2])
    #result[1]=list(map(int, result[1]))
    #result[2]=list(map(float, result[2]))
    #print(result[1],result[2])

    #with open('/content/drive/MyDrive/BerlinA100SavedImage/berlin_A100_200_128.csv', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(zip(result[1],result[2]))

    precision, recall, thresholds = precision_recall_curve(predictionLabel,predictionScore)
    import csv
    with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_Halenseestrasse_400_256-AUC808.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(predictionLabel,predictionScore))

    plt.step(recall, precision, alpha=0.2,color="Red",where='post')
    #plt.fill_between(recall, precision, step='post', alpha=0.2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
#        PR_values.append(str(auc(recall, precision)))
    tiTle= "Region-VLAD "  + " AUC-PR: " + str(auc(recall, precision))
    plt.grid(True)
       # plt.legend(tiTle)
    plt.title(tiTle,fontsize=8)
    #plt.savefig(os.path.join(resultPath,"AUC-PR"+"_"+str(N)+"_"+str(V)+'.jpg'),dpi=200)
    print("i am here")
        #plt.close()


showAUCPR()

import csv
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_Halenseestrasse_200_128-AUC753.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(predictionLabel,predictionScore))

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_Halenseestrasse_400_256-AUC808.csv', header=None)
df.rename(columns={0: 'label', 1: 'score'}, inplace=True)
df.to_csv('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_Halenseestrasse_400_256-AUC808.csv', index=False) # save to new csv file

"""_________________________________"""

import csv
list_of_label=[]
list_of_score=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/TryHalen.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label.append(row['label'])
        list_of_score.append(row['score'])
list_of_label=list(map(float, list_of_label))
list_of_score=list(map(float, list_of_score))

print(len(list_of_label))

from sklearn.metrics import precision_recall_curve,auc,average_precision_score
import os
import matplotlib.pyplot as plt
import pickle
from pylab import imread,subplot,imshow,show
import matplotlib.gridspec as gridspec
import csv
def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

datasetIndex = 4
img_for=".jpg"
#imgTag= "images-"
imgTag= ""


def showAUCPR():
    print("i am be 1")
    files_list=[]
    #datasetPath = os.path.join(dir,dataset)
    #testPath = os.path.join(dir,dataset,testTraverse)
    #referencePath = os.path.join(dir,dataset,referenceTraverse)

    #for file in os.listdir('/content/drive/MyDrive/pickefolder/GardenPoint_Walking_200_128 (1).pkl'):
    #    if file.endswith(".pkl"):
    #        files_list.append(file)
    #    else:
    #        continue
    #resultPath = '/content/drive/MyDrive/pickefolder/GardenPoint_Walking_200_128 (1).pkl'
    #PR_values = list()
    print("i am be 2")

    #Results = load_obj('/content/drive/MyDrive/pickefolder/GardenPoint_Walking_200_128 (1).pkl')



    #predictionLabel,predictionScore = list(),list()
    #for _,result in Results.items():
    #    predictionLabel.append(result[1])
    #    predictionScore.append(result[2])
    #result[1]=list(map(int, result[1]))
    #result[2]=list(map(float, result[2]))


    #with open('/content/drive/MyDrive/BerlinA100SavedImage/berlin_A100_200_128.csv', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerows(zip(result[1],result[2]))

    precision, recall, thresholds = precision_recall_curve(list_of_label,list_of_score)

    plt.step(recall, precision, alpha=0.2,color="Red",where='post')
    #plt.fill_between(recall, precision, step='post', alpha=0.2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
#        PR_values.append(str(auc(recall, precision)))
    tiTle= "Region-VLAD "  + " AUC-PR: " + str(auc(recall, precision))
    plt.grid(True)
       # plt.legend(tiTle)
    plt.title(tiTle,fontsize=8)
    #plt.savefig(os.path.join(resultPath,"AUC-PR"+"_"+str(N)+"_"+str(V)+'.jpg'),dpi=200)
    print("i am here")
        #plt.close()


showAUCPR()

"""# **Draw Plot for multi lines**"""

import csv
list_of_label_Active1=[]
list_of_score_Active1=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_HalenSeeStrasse.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label_Active1.append(row['label'])
        list_of_score_Active1.append(row['score'])
list_of_label_Active1=list(map(int, list_of_label_Active1))
list_of_score_Active1=list(map(float, list_of_score_Active1))

import csv
list_of_label_Active7=[]
list_of_score_Active7=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_HalenSeeStrasse-7Active-AUC63.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label_Active7.append(row['label'])
        list_of_score_Active7.append(row['score'])
list_of_label_Active7=list(map(int, list_of_label_Active7))
list_of_score_Active7=list(map(float, list_of_score_Active7))

import csv
list_of_label_OXF128=[]
list_of_score_OXF128=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_Halenseestrasse_200_128-AUC753.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label_OXF128.append(row['label'])
        list_of_score_OXF128.append(row['score'])
list_of_label_OXF128=list(map(float, list_of_label_OXF128))
list_of_score_OXF128=list(map(float, list_of_score_OXF128))

import csv
list_of_label_OXF256=[]
list_of_score_OXF256=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Berlin_Halenseestrasse_400_256-AUC808.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label_OXF256.append(row['label'])
        list_of_score_OXF256.append(row['score'])
list_of_label_OXF256=list(map(float, list_of_label_OXF256))
list_of_score_OXF256=list(map(float, list_of_score_OXF256))

import csv
list_of_label_Bow_V26K=[]
list_of_score_Bow_V26K=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Cross-Region-Bow-N200-V26K-AUC418.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label_Bow_V26K.append(row['label'])
        list_of_score_Bow_V26K.append(row['score'])
list_of_label_Bow_V26K=list(map(float, list_of_label_Bow_V26K))
list_of_score_Bow_V26K=list(map(float, list_of_score_Bow_V26K))

import csv
list_of_label_Vlad=[]
list_of_score_Vlad=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Cross-Region-VLAD-N200-V128-AUC2539.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label_Vlad.append(row['label'])
        list_of_score_Vlad.append(row['score'])
list_of_label_Vlad=list(map(float, list_of_label_Vlad))
list_of_score_Vlad=list(map(float, list_of_score_Vlad))

import csv
list_of_label_Alexnet=[]
list_of_score_Alexnet=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Alexnet3635-RMAC-AUC040.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label_Alexnet.append(row['label'])
        list_of_score_Alexnet.append(row['score'])
list_of_label_Alexnet=list(map(float, list_of_label_Alexnet))
list_of_score_Alexnet=list(map(float, list_of_score_Alexnet))

import csv
list_of_label_HybridSPP=[]
list_of_score_HybridSPP=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/HybridNet-SPP-AUC0725.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label_HybridSPP.append(row['label'])
        list_of_score_HybridSPP.append(row['score'])
list_of_label_HybridSPP=list(map(float, list_of_label_HybridSPP))
list_of_score_HybridSPP=list(map(float, list_of_score_HybridSPP))

import csv
list_of_label_Bow_V10K=[]
list_of_score_Bow_V10K=[]
with open('/content/drive/MyDrive/BerlinHalenSeeStrasseSavedImage/Cross-Region-Bow-N200-V10K-AUC528.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['new_num'], row['old_num'])
        list_of_label_Bow_V10K.append(row['label'])
        list_of_score_Bow_V10K.append(row['score'])
list_of_label_Bow_V10K=list(map(float, list_of_label_Bow_V10K))
list_of_score_Bow_V10K=list(map(float, list_of_score_Bow_V10K))

from sklearn.metrics import precision_recall_curve,auc,average_precision_score
import os
import matplotlib.pyplot as plt
import pickle
from pylab import imread,subplot,imshow,show
import matplotlib.gridspec as gridspec
import csv


def showAUCPR():
    print("i am be 1")
    files_list=[]
    #Create a matplotlib figure
    #fig, ax = plt.subplots()
    #fig, axs = plt.subplots(nrows=4, ncols=3, sharex='col', sharey='row',subplot_kw={'projection': })  # <--- Here is the missing piece

    precision, recall, thresholds = precision_recall_curve(list_of_label_Active7,list_of_score_Active7)
    ans1=auc(recall, precision)
    an1=str(round(ans1, 2))
    s1="7-Component triggered (AUC:"+an1+")"

    plt.step(recall, precision, alpha=1,color="Red",where='post')
    #########################
    precision, recall, thresholds = precision_recall_curve(list_of_label_Active1,list_of_score_Active1)
    ans2=auc(recall, precision)
    an2=str(round(ans2, 2))
    s2="1-Component triggered (AUC:"+an2+")"
    plt.step(recall, precision, alpha=1,color="Orange",where='post')
    #########################
    precision, recall, thresholds = precision_recall_curve(list_of_label_OXF128,list_of_score_OXF128)
    ans3=auc(recall, precision)
    an3=str(round(ans3, 2))
    s3="Region-VLAD N:200,V:128 (AUC:"+an3+")"
    plt.step(recall, precision, alpha=1,color="Blue",where='post')
    #########################
    #########################
    precision, recall, thresholds = precision_recall_curve(list_of_label_OXF256,list_of_score_OXF256)
    ans4=auc(recall, precision)
    an4=str(round(ans4, 2))
    s4="Region-VLAD N:400,V:256 (AUC:"+an4+")"
    plt.step(recall, precision, alpha=1,color="Cadetblue",where='post')
    #########################
    #########################
    precision, recall, thresholds = precision_recall_curve(list_of_label_Bow_V26K,list_of_score_Bow_V26K)
    ans5=auc(recall, precision)
    an5=str(round(ans5, 2))
    s5="Cross-Region-BoW N:200,V:2.6K (AUC:"+an5+")"
    plt.step(recall, precision, alpha=1,color="Brown",where='post')
    #########################
    #########################
    precision, recall, thresholds = precision_recall_curve(list_of_label_Vlad,list_of_score_Vlad)
    ans6=auc(recall, precision)
    an6=str(round(ans6, 2))
    s6="Cross-Region-VLAD N:200,V:128 (AUC:"+an6+")"
    plt.step(recall, precision, alpha=1,color="#9C661F",where='post')
    #########################
    #########################
    precision, recall, thresholds = precision_recall_curve(list_of_label_Alexnet,list_of_score_Alexnet)
    ans7=auc(recall, precision)
    an7=str(round(ans7, 2))
    s7="Alexnet365 RMAC (AUC:"+an7+")"
    plt.step(recall, precision, alpha=1,color="#7FFF00",where='post')
    #########################
    #########################
    precision, recall, thresholds = precision_recall_curve(list_of_label_HybridSPP,list_of_score_HybridSPP)
    ans8=auc(recall, precision)
    an8=str(round(ans8, 2))
    s8="HybridNet SPP (AUC:"+an8+")"
    plt.step(recall, precision, alpha=1,color="#808A87",where='post')
    #########################
    #########################
    precision, recall, thresholds = precision_recall_curve(list_of_label_Bow_V10K,list_of_score_Bow_V10K)
    ans9=auc(recall, precision)
    an9=str(round(ans9, 2))
    s9="Cross-Region-BoW N:200,V:10K (AUC:"+an9+")"
    plt.step(recall, precision, alpha=1,color="#68228B",where='post')
    #########################
    #plt.fill_between(recall, precision, step='post', alpha=0.2)
    #tiTle= "Region-VLAD "  + " AUC-PR: " + str(auc(recall, precision))
    #plt.title(tiTle)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #ax.ylim
    #ax.xlabel('Recall')
    #ax.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
#        PR_values.append(str(auc(recall, precision)))

    #plt.legend(loc="lower center","here")
    # Add a legend
    # Add a legend
    # Add a legend
    #pos = ax.get_position()

    #ax.set_position([pos.x1, pos.y2, pos.width, pos.height * 0.85])
    plt.legend([s1,s2,s3,s4,s5,s6,s7,s8,s9],loc='center', bbox_to_anchor=(0.5, -0.5),ncol=1, )


    plt.grid(True)
       # plt.legend(tiTle)
    #ax.title(tiTle,fontsize=8)
    #plt.savefig(os.path.join(resultPath,"AUC-PR"+"_"+str(N)+"_"+str(V)+'.jpg'),dpi=200)
    print("i am here")
        #plt.close()
    plt.show()


showAUCPR()