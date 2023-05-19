import streamlit as st
import numpy as np
import pandas as pd
import pickle5 as pickle

import torchvision
from torchvision import datasets
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

device = torch.device('cpu')
num_classes = 133
with open('dog_classes.pkl', 'rb') as f:
	class_names = pickle.load(f)

def Model_DogClassify():
	model_transfer = models.resnet50(pretrained=True)
	n_inputs = model_transfer.fc.in_features    
	last_layer = nn.Linear(n_inputs, num_classes)
	model_transfer.fc = last_layer
	return model_transfer

def load_image(img_path):
	image = Image.open(img_path).convert('RGB')        
	in_transform = transforms.Compose([
                        transforms.Resize(size = (224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

	# discard the transparent, alpha channel (that's the :3) and add the batch dimension
	image = in_transform(image)[:3,:,:].unsqueeze(0)
	return image


@st.experimental_memo #avoid reloading data again and again
def load_model():
	model = Model_DogClassify()
	model_path = 'model_transfer.pt'
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.eval()
	return model

def dog_classify():
	model = load_model()
	model.to(device)
	model.eval()
	
	st.title('Dog-Breed Classifier')
	with st.sidebar:
		st.write('## Upload a dog image:')
		uploaded_file = st.file_uploader('', type = (["jpg", "jpeg"]))

	if uploaded_file is not None:
		#st.image(img, caption = 'Uploaded image')
		st.sidebar.image(Image.open(uploaded_file))#, caption = 'Uploaded image')
		img = load_image(uploaded_file)
		img = img.to(device)
		
		with torch.no_grad():
			output = model.forward(img)

		ps = torch.exp(output)
		sum_ps = torch.sum(ps)
		top_p, top_class = ps.topk(5)
		top_p = top_p / sum_ps
		probs = top_p.cpu().numpy()[0] * 100
		idx = top_class.cpu().numpy()[0]
		classes = [class_names[x] for x in idx]

		fig = plt.figure(figsize=(5, 10))
		ax1 = fig.add_subplot(2, 1, 1)
		ax1.axis('off')
		ax1.set_title('Input Image', fontsize = 25)
		image = Image.open(uploaded_file)
		ax1.imshow(image)
    
		ax2 = fig.add_subplot(2, 1, 2)
		ax2.set_title('Probability Chart', fontsize = 25)
		ax2.set_xlabel('Probability(%)', fontsize = 20)
		#ax2.set_ylabel('Dog Types', fontsize = 20)
		ax2 = plt.yticks(range(5), classes[::-1], fontsize = 15)
		ax2 = plt.barh(range(5), probs[::-1]) 
		ax2 = plt.xticks(fontsize = 15) 
		st.pyplot(fig)




