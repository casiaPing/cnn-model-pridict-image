# -*-coding:utf-8-*-
import numpy
from numpy import float64
import theano
from PIL import Image
from pylab import *
import os,glob
import theano.tensor as T
import random
import pickle



def dataresize(path=r'./image/train'):
    a_lable=[0 for x in range(1) for y in range(10)]
    
    print("a:S",a_lable)
    # test path
    path_t =r"./image/test"
    # train path
    datas = []
    train_x= []
    train_y= []
    valid_x= []
    valid_y= []
    test_x= []
    test_y= []
    #˫��ѭ��û�ã���Ϊ����һ���ļ�������Ϊ�˷��������չ����û��ɾ��
    for filename in os.listdir(path):     
        imgpath =os.path.join(os.path.join(path),filename)
        #print("filename",imgpath)
        img = Image.open(imgpath)
        img =img.convert('L').resize((320,320))
        width,hight=img.size
        img = numpy.asarray(img,dtype='float64')/256.

        tmp = img.reshape(1, hight*width)[0]
        #���濪ʼ��ӱ�ǩ
        for line in open("./image/train_label.txt"): 
             
            imageinfo = line.split(' ')
            imagename = imageinfo[0]
            imagelable = int( imageinfo[1][:-2])
            #print("line",line)
            #print("imagelable",imagelable)         
            if filename == imagename:
                a_lable[10-imagelable] = 1
                imagelable_temp = a_lable
                tmp =hstack((imagelable_temp.reshape(1,10),tmp))  # �ڴ˽���ǩ�������ݵ�ǰ�档
                a_lable[10-imagelable] = 0

        datas.append(tmp)
    # datas.append(img.reshape(1, hight*width)[0])
    #�ڴ˴�ȡ����һ�е����ݷ����ں����ת���Ĺ����л���ֵ��ӵ�������ڳ���ת���ɾ���ʱ������ת���Ĵ���
    #�����ݴ���˳��
    random.shuffle(datas)
    # �����ݺͱ�ǩ���з���
    label=[]
    print("data0", datas[0])
    for num in range(len(datas)):
        label.append((datas[num])[0])
        datas[num] =(datas[num])[1:]
        #�����ݵı�ǩ��ȥ��
    tests = []
        # #��ȡ���Լ�
    for filename in os.listdir(path_t):
        imgpath =os.path.join(os.path.join(path_t),filename)
        #print("filename",imgpath)
        img = Image.open(imgpath)
        img =img.convert('L').resize((320,320))
        width,hight=img.size
        img = numpy.asarray(img,dtype='float64')/256.
        tmp = img.reshape(1, hight*width)[0]
       
        # �ڴ��������ȡ��[0]�Ļ��ں���ᷢ����ʵ����һ����ά�����ݵĵ��ӣ�
        # �ں���ʹ��theano�е�cnn�ڵ���ʱ��������ݵ��쳣��ת�����쳣����
        # �ڴ��Ǹ�ԭʼ��mnist�����ݼ�����ʽ���˱Ƚ��޸Ĳŷ��ֵġ�����
        #���濪ʼ��ӱ�ǩ
        for line in open("./image/test_label.txt"):  
            imageinfo = line.split(' ')
            imagename = imageinfo[0]
            imagelable = int( imageinfo[1][:-2])
            
            if filename == imagename:
                a_lable[10-imagelable] = 1
                imagelable_temp = a_lable
                tmp =hstack((imagelable_temp.reshape(1,10),tmp))  # �ڴ˽���ǩ�������ݵ�ǰ�档
                print("tmp",tmp[0])
                a_lable[10-imagelable] = 0
                
                
        tests.append(tmp)
    #�����ݴ���˳��
    random.shuffle(tests)
    #  �����ݺͱ�ǩ���з���
    label_t=[]
    for num in range(len(tests)):
        #print("test:",test)
        label_t.append((tests[num])[0])
    
        #print("label_t",label_t[0])
        tests[num] =(tests[num])[1:]
        #�����ݵı�ǩ��ȥ��
        '''    �����ݽ��д��ң���ֳ�train test valid    '''
    print("len(label)",len(label))
    for num in range(len(label)):
        train_x.append(datas[num])
        train_y.append(label[num].reshape((1,10)))
    for num in range(len(tests)):
        if num%2==0:
            valid_x.append(tests[num])
            valid_y.append(label_t[num])
        if num%2==1:
            test_x.append(tests[num])
            test_y.append(label_t[num])
    train_x=numpy.asarray(train_x,dtype='float64')
    print("train_x",train_x.shape)
    train_y=numpy.asarray(train_y,dtype='float64')
    print("train_y",train_y.shape)
    valid_x=numpy.array(valid_x,dtype='float64')
    valid_y=numpy.asarray(valid_y,dtype='float64')
    test_x=numpy.asarray(test_x,dtype='float64')
    test_y=numpy.asarray(test_y,dtype='float64')
    print("train_x",train_x.shape)
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset((test_x,test_y))
    valid_set_x, valid_set_y = shared_dataset((valid_x,valid_y))
    train_set_x, train_set_y = shared_dataset((train_x,train_y))
    print("train_set_x",train_set_x)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
         (test_set_x, test_set_y)]
    save_datas_pkl(rval)
        #  return rval
    return train_x,train_y,valid_x,valid_y,test_x,test_y
def save_datas_pkl(file1s,path=r'./image/pkl/datasets.pkl'):
    datas=file1s
    output =open(path,'wb')
    pickle.dump(datas,output)
   
    output.close()
def load_datas(path=r'./image/datasets.pkl'):
    pkl_file =open(path,'rb')
    datas =pickle.load(pkl_file)
    pkl_file.close()
    #return datas

if __name__=='__main__':
    dataresize()