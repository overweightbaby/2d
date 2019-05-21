#coding:utf-8
import os,random
import xml.etree.ElementTree as etree
from keras.preprocessing.image import img_to_array, load_img,array_to_img
import numpy as np
from scipy.ndimage.interpolation import  zoom
from aug import process
import csv
from whole_image import aug
import cv2
'''

'''


def crop_core(img,xmax,xmin,ymax,ymin):
    area = (xmax-xmin)*(ymax-ymin)
    print 'area',area
    temp_img_array=np.ndarray((1,299,299,3))
    get=0

    while get==0:
        rand_left = random.randint(0,xmin)
        if (xmin+420)>img.shape[0]:
            rand_left = img.shape[0]-421
        rand_up = random.randint(0,ymin)
        if (ymin+420)>img.shape[1]:
            rand_up = img.shape[1]-421
        temp_img_array = img[rand_left:rand_left+420,rand_up:rand_up+420,:]
        y1 = np.maximum(rand_up, ymin)
        x1 = np.maximum(xmin, rand_left)
        y2 = np.minimum(rand_up+420, ymax)
        x2 = np.minimum(rand_left+420, xmax)
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        print "intersection",intersection
        if intersection !=0 :
            get=1
    temp_img_array = cv2.resize(temp_img_array,(299,299),interpolation=cv2.INTER_CUBIC)
    return temp_img_array

def normalization(img):
    img = img.astype("float")
    img = (img-img.min()+1.)/(img.max()-img.min()+1.)
    return img


def extract():
    all_set = os.listdir("/data2/zyj/Xuelang/train_data")
    pictures = []
    i=0
    name_list = ['边扎洞','毛洞','吊经','缺纬','跳花','剪洞','织稀','回边','缺经','扎洞','擦洞']
    for set in all_set:
        set_name = os.path.join("/data2/zyj/Xuelang/train_data", set)
        data_categories = os.listdir(set_name)
        for data_category in data_categories:
            data_path = os.path.join(set_name, data_category)
            print(data_path)
            data_names = os.listdir(data_path)
            if data_category != "正常":
                for data_name in data_names:
                    if data_name.endswith(".jpg"):
                        pic = load_img(os.path.join(data_path, data_name))
                        pic_array = img_to_array(pic)
                        pic_array = np.transpose(pic_array, (1, 0, 2))
                        print("pic",pic_array.max(),pic_array.min())
                        name, ext = os.path.splitext(data_name)
                        xml = os.path.join(data_path, name + ".xml")
                        tree = etree.parse(xml)
                        root = tree.getroot()
                        read_node = root.find("object")
                        if (read_node is None or len(read_node) == 0):
                            continue
                        for read_session in root.findall("object"):
                            box_nodes = read_session.findall('bndbox')
                            for xml_roi in box_nodes:
                                xmin = int(xml_roi.find('xmin').text)
                                xmax = int(xml_roi.find('xmax').text)
                                ymin = int(xml_roi.find('ymin').text)
                                ymax = int(xml_roi.find('ymax').text)
                                #print(xmin, xmax, ymin, ymax), data_category
                                coord = [xmax, xmin, ymax, ymin]
                                loop = 3
                                '''
                                if data_category in name_list:
                                    if data_category=='擦洞':
                                        loop=6
                                '''
                                for l in range(loop):
                                    temp_pos_array = crop_core(pic_array, xmax, xmin, ymax, ymin)
                                    #temp_pos_array = normalization(temp_pos_array)
                                    #temp_pos_array = aug(temp_pos_array)
                                    pictures.append([temp_pos_array,1,data_category,data_name,set_name])
                                #labels.append(coord)

            else:
                for data_name in data_names:
                    if data_name.endswith(".jpg"):
                        pic = load_img(os.path.join(data_path, data_name))
                        pic_array = img_to_array(pic)
                        print("pic",pic_array.max(),pic_array.min())
                        for i in range(8):
                            left = random.randint(0,pic_array.shape[0]-641)
                            up = random.randint(0,pic_array.shape[1]-641)
                            img = np.ndarray((1,299,299,3))
                            img = pic_array[left:left+640,up:up+640,:]
                            img = cv2.resize(img,(299,299),interpolation=cv2.INTER_CUBIC)
                            #img = normalization(img)
                            #img = aug(img)
                            pictures.append([img,0,data_category,data_name,set_name])
                        
                        
    random.shuffle(pictures)
    parray = []
    larray = []
    name_dict = []
    files =[]
    sets = []
    normal = 0
    abnormal = 0
    for p in pictures:

        parray.append(p[0])
        larray.append([p[1]])
        if p[1] == 0:
            normal+=1
        else:
            abnormal+=1
        name_dict.append(p[2])
        files.append(p[3])
        sets.append(p[4])
    #parray = np.concatenate(parray,axis=0)
    #larray = np.concatenate(larray,axis=0)
    parray = np.asarray(parray)
    larray = np.asarray(larray)

    print "shape",parray.shape,larray.shape
    print "prop",normal,abnormal
   
    train_data = parray[0:int(parray.shape[0] * 0.95)]
    train_label = larray[0:int(larray.shape[0] * 0.95)]
    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data-mean)/std

    val_data = parray[int(parray.shape[0] * 0.95):int(parray.shape[0] * 0.98)]
    val_label = larray[int(larray.shape[0] * 0.95):int(parray.shape[0] * 0.98)]
    val_data = (val_data-mean)/std
    val_category = name_dict[int(larray.shape[0] * 0.95):int(parray.shape[0] * 0.98)]
    with open('val_cat.csv','wb') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        i=0
        for name in val_category:
            writer.writerow([name,files[i]])
            print name
 
    test_data = parray[int(parray.shape[0] * 0.98)::]
    test_label = larray[int(larray.shape[0] * 0.98)::]
    test_category = name_dict[int(larray.shape[0] * 0.98)::]
    test_file = files[int(larray.shape[0] * 0.98)::]
    test_set = sets[int(larray.shape[0] * 0.98)::]
    test_data = (test_data-mean)/std
    #print test_label
    with open('test_cat.csv','wb') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        i=0
        for name in test_category:
            writer.writerow([test_set[i],name,test_file[i]])
            i+=1
            print name
    print(mean,std)
    np.save("/data2/zyj/Xuelang/xuelang/train_data.npy", train_data)
    np.save("/data2/zyj/Xuelang/xuelang/train_label.npy",train_label)
    np.save("/data2/zyj/Xuelang/xuelang/val_data.npy", val_data)
    np.save("/data2/zyj/Xuelang/xuelang/val_label.npy",val_label)
    np.save("/data2/zyj/Xuelang/xuelang/test_data.npy", test_data)
    np.save("/data2/zyj/Xuelang/xuelang/test_label.npy",test_label)

if __name__ == "__main__":
    #先统计所有图片数 再想图片要多大
    extract()
    #labels = np.concatenate(labels, axis=0)
    '''
    count = 0
    for i in range(labels.shape[0]):
        if labels[i]==1:
            count += 1
    print(count)
    

    data = np.load("/data2/zyj/Xuelang/xuelang/test_data.npy")
    label = np.load("/data2/zyj/Xuelang/xuelang/test_label.npy")

    for i in range(0, 10):
        temp = data[i]
        temp1 = array_to_img(temp)
        temp1.save("/data2/zyj/Xuelang/xuelang/{0}.jpg".format(i))
    '''
