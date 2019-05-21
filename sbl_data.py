import os
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img,array_to_img
from skimage import measure
import numpy as np
from keras.utils import to_categorical
import random
import cv2
import re


def generate_txt(train_or_test,data_set):
	subsets = os.listdir(data_set)
	with open('/data2/zyj/wce/2class'+train_or_test+'.txt','w') as file:
		for subset in subsets:
			if subset=='normal':
				continue
			if subset == 'vascularlesions':
				label = 1
			else:
				label = 0
			data_names = os.listdir(os.path.join(data_set,subset))
			for data_name in data_names:
				if '_' in data_name:
					continue
				data_path = os.path.join(os.path.join(data_set,subset),data_name)
				if label == 0:
					file.write(data_path+" ")
					file.write('non-vas,'+str(label)+'\n')
					continue
				if train_or_test=='train':
					mask_name = data_name[:data_name.find('.jpg')]+'_a.jpg'
					mask_path = os.path.join(os.path.join(data_set,subset),mask_name)
					if not os.path.exists(mask_path):
						mask_name = data_name[:data_name.find('.jpg')]+'_a.JPG'
						mask_path = os.path.join(os.path.join(data_set,subset),mask_name)
					mask_img = load_img(mask_path,grayscale=True)
					mask_array = img_to_array(mask_img)
					masks=measure.label(mask_array>150,connectivity=2)
					Props = measure.regionprops(masks)
					for i in range(len(Props)):
						y1,x1,y2,x2 = Props[i]['bbox']
						ymin,xmin,ymax,xmax = int(y1),int(x1),int(y2),int(x2)
					file.write(data_path+" ")
					file.write(str(ymin)+","+str(xmin)+","+str(ymax)+","+str(xmax)+",")
					file.write(str(label))
					file.write('\n')
				else:
					file.write(data_path+" ")
					if label==0:
						file.write('non-vas,'+str(label)+'\n')
					else:
						file.write('vas,'+str(label)+'\n')
	file.close()

def get_random_data(annotation_line):
	image_name = annotation_line[0:annotation_line.find('.jpg')+4]
	image = load_img(image_name)
	image_data = img_to_array(image)
	label = annotation_line[-2:-1]
	label = int(label)
	if label!=0:
		mask_name = image_name[:image_name.find('.jpg')]+'_a.jpg'
		if not os.path.exists(mask_name):
			mask_name = image_name[:image_name.find('.jpg')]+'_a.JPG'
		mask_img = load_img(mask_name,grayscale=True)
		mask_array = img_to_array(mask_img)
	else:
		mask_array = []
	#bbox = annotation_line[annotation_line.find('.jpg')+4:-3]
	return image_data,label,mask_array


def data_generator(annotation_lines,batch_size,stage,name):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    print(name)
    while True:
        image_data = []
        labels = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image_name = annotation_lines[i][0:annotation_lines[i].find('.jpg')+4]
            if not os.path.exists(image_name):
            	b-=1
            	i = (i+1) % n
            	print('b',b)
            	continue
            image,label,mask_array = get_random_data(annotation_lines[i])
            image_resized,new_label = process(image,label,stage,mask_array)
            image_data.append(image_resized)
            labels.append(new_label)
            i = (i+1) % n
        if (0 not in labels) or (1 not in labels):
        	continue
        image_data = np.array(image_data)
        labels = np.array(labels)
        labels_cate = to_categorical(labels)
        yield image_data,labels_cate



def process(image,label,stage,mask):
	if label!=0:
		image_resized,new_label,mask_crop = crop_core(image,label,mask)
	else:
		if stage==1:
			image_resized = cv2.resize(image,(299,299),interpolation=cv2.INTER_CUBIC)
		else:
			k = random.uniform(0,1)
			if k>0.5:
				rand_left = random.randint(0,image.shape[0]-221)
				rand_up = random.randint(0,image.shape[1]-221)
				temp_img_array = image[rand_left:rand_left+220,rand_up:rand_up+220,:] 
				image_resized = cv2.resize(temp_img_array,(299,299),interpolation=cv2.INTER_CUBIC)
			else:
				image_resized = cv2.resize(image,(299,299),interpolation=cv2.INTER_CUBIC)
		new_label = label
	return image_resized,new_label


def crop_core(img,label,mask):
    temp_img_array=np.ndarray((1,299,299,3))
    pos=0
    neg=0
    while (pos==0) and (neg==0):
        rand_left = random.randint(0,img.shape[0]-221)
        rand_up = random.randint(0,img.shape[1]-221)
        temp_img_array = img[rand_left:rand_left+220,rand_up:rand_up+220,:]
        temp_mask_array = mask[rand_left:rand_left+220,rand_up:rand_up+220,:]
        intersection = np.sum(temp_mask_array)
        #print "intersection",intersection
        if intersection !=0 :
            pos+=1
            new_label = label
        else:
        	neg+=1
        	new_label = 0
    temp_img_array = cv2.resize(temp_img_array,(299,299),interpolation=cv2.INTER_CUBIC)
    return temp_img_array,new_label,temp_mask_array

def data_loader(annotation_lines):
    images = []
    labels = []
    masks = []
    for annotation_line in annotation_lines:
        image_name = annotation_line[0:annotation_line.find('.jpg')+4]
        if not os.path.exists(image_name):
            continue
        image = load_img(image_name)
        image_data = img_to_array(image)

        label = annotation_line[-2:-1]
        label = int(label)

        image_array = cv2.resize(image_data,(299,299),interpolation=cv2.INTER_CUBIC)
        images.append(image_array)
        labels.append(label)
        '''
        if label!=0:
			mask_name = image_name[:image_name.find('.jpg')]+'_a.jpg'
			if not os.path.exists(mask_name):
				mask_name = image_name[:image_name.find('.jpg')]+'_a.JPG'
			mask_img = load_img(mask_name,grayscale=True)
			mask_array = img_to_array(mask_img)
			image_resized,new_label,mask_crop = crop_core(image_data,label,mask_array)
			images.append(image_resized)
			labels.append(new_label)
			masks.append(mask_crop)
        else:
			mask_array = []
			image_resized,new_label = process(image_data,label,stage,mask_array)
			images.append(image_resized)
			labels.append(new_label)
			masks.append(mask_array)
		'''
    images = np.array(images)
    labels = np.array(labels)
    print(labels)
    for i in range(10):
    	print(i)
    	temp_img = images[i]
    	temp_label = labels[i]
    	timg = array_to_img(temp_img)
    	'''
    	mask_ = np.array(masks[i])
    	if not mask_.all():
    		t_mask = array_to_img(mask_)
    		t_mask.save('test/'+str(i)+str(temp_label)+'_mask.jpg')
    	'''
    	timg.save('test/'+str(i)+str(temp_label)+'.jpg')
    labels_cate = to_categorical(labels)
    return images,labels_cate
'''
if __name__ == '__main__':
	train = '/data2/zyj/wce/wcetraining'
	test = '/data2/zyj/wce/wcetest'
	generate_txt(train_or_test='train',data_set=train)
	generate_txt(train_or_test='test',data_set=test)
'''
