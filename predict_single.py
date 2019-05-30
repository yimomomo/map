import tensorflow as tf
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
from skimage import morphology
from scipy import stats


'''
def get_semantic_labels(img_path):
    img = load_img(img_path)
    img = img_to_array(img)
    img=np.expand_dims(img,axis=0).astype(np.uint8)

    sess = tf.Session()
    with open("frozen_inference_graph.pb","rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        output = tf.import_graph_def(graph_def,input_map={"ImageTensor:0":img},return_elements=["SemanticPredictions:0"])
    result = sess.run(output)
    print(result[0].shape)
    return result[0]

get_semantic_labels("/home/yimo/Research/dataset/singleTestImages/05.jpg")
'''


'''
use deeplabv3+ predict single image,return semantic segmentation labels
Next: read from ros topic

'''
def predict_single(Limg):
    
    print(Limg.shape)
    #img.resize((513,513,3))
    #plt.imshow(img)
    #plt.imshow(img)
    #plt.show()
    Limg = np.expand_dims(Limg,axis=0).astype(np.uint8)

    graph = tf.Graph()
    INPUT_TENSOR_NAME='ImageTensor:0'
    OUTPUT_TENSOR_NAME='SemanticPredictions:0'
    graph_def=None
    graph_path = "./frozen_inference_graph.pb"
    with tf.gfile.FastGFile(graph_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    if graph_def is None:
        raise RuntimeError('Cannot find inference graph in tar archive')
    with graph.as_default():
        tf.import_graph_def(graph_def,name='')

    sess = tf.Session(graph=graph)
    result=sess.run(OUTPUT_TENSOR_NAME,feed_dict={INPUT_TENSOR_NAME:Limg})

    #print(type(result))
    #print(result[0])

    #cv2.imwrite('aaa.jpg',result.transpose((1,2,0)))
    #plt.imshow(result[0])
    mask = np.zeros((result[0].shape[0],result[0].shape[1]))
    for i in range(result[0].shape[0]):
        for j in range(result[0].shape[1]):
            if result[0][i,j] == 4:
                mask[i,j] = 1
    plt.imshow(mask) 
    plt.show()
    return mask

'''
use erosion to define the probability of traversability.

'''
def get_skeleton(mask):
    mask = cv2.copyMakeBorder(mask,10,10,10,10,cv2.BORDER_CONSTANT,value = [0,0,0])
    plt.imshow(mask) 
    plt.show()
    #skeleton =morphology.skeletonize(mask)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(mask,kernel)
    subtract = cv2.subtract(mask,erosion)
    res = []#store the coordinate and the level of road pixels
    #find the coordinates of level 1
    coordinate = np.where(subtract == 1)
    #print coordinate
    x = coordinate[0]
    y = coordinate[1]
    #print x
    #print y
    co = np.concatenate((x.reshape(x.shape[0],1),y.reshape(y.shape[0],1)),axis = 1)
    res.append(co)
    plt.imshow(subtract) 
    plt.show()
    '''
    keep erosion and record the coordinate
    '''
    while erosion.sum() !=0:
        tmp = cv2.erode(erosion,kernel)
        subtract = cv2.subtract(erosion,tmp)
        coordinate = np.where(subtract == 1)
        #print coordinate
        x = coordinate[0]
        y = coordinate[1]
        #print x
        #print y
        co = np.concatenate((x.reshape(x.shape[0],1),y.reshape(y.shape[0],1)),axis = 1)
        #print co
        res.append(co)
        erosion = tmp
    '''
    plot part 
    '''
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    ax1.imshow(mask, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('original', fontsize=20)

    ax2.imshow(erosion, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title('skeleton', fontsize=20)

    fig.tight_layout()
    plt.show()

    return res
'''
use pixel level from erosion to calculate the probability
first try Gaussian model
miu = N
theta = E[1/n*\sum{(x_i-miu)**2}]
'''
def traversability_vision(res,Limg):
    miu = len(res)
    X = np.array([i+1 for i in range(miu)])
    X_miu = np.array([(i+1 - miu)**2 for i in range(miu)])
    sigma = X_miu.sum()*(1.0/miu)
    sigma = sigma**0.5
    Y = stats.norm.pdf(X,miu,sigma)
    P = stats.norm.cdf(X,miu,sigma)
    #print P
    plt.plot(X,Y)
    plt.plot(X,P)
    plt.show()
    
    probability_visual = np.zeros((Limg.shape[0]+20,Limg.shape[1]+20,3),np.uint8)
    probability = np.zeros((Limg.shape[0]+20,Limg.shape[1]+20),np.uint8)
    #print probability.shape
    for i in range(miu):
        R = int(255*P[i]*2)
        G = int(255*P[i]*2)
        B = int(255*P[i]*2)
        #print R
        for co in res[i]:
            probability_visual[co[0],co[1],0] = R
            probability_visual[co[0],co[1],1] = G
            probability_visual[co[0],co[1],2] = B
            probability[co[0],co[1]] = P[i]*2
    plt.imshow(probability_visual)
    plt.show()
    return probability
 

'''
transform the probability to the world coordinate

'''

def grid_transform(probability,Depth,intrinsic_matrix):
    intrinsic_matrix_inverse = np.linalg.inv(intrinsic_matrix)
    print intrinsic_matrix_inverse
    #grid_map_single = np.zeros(())
    probability = probability[10:-10,10:-10]
    print probability.shape
    w_x_max = -10e9
    w_x_min  =10e9
    w_y_max = -10e9
    w_y_min = 10e9
    for u in range(probability.shape[0]):
        for v in range(probability.shape[1]):
            V_img = np.array([u,v,1]).reshape((3,1)) 
            print Depth[u,v]
            V_w = np.dot(intrinsic_matrix_inverse,V_img)
            #print V_w
            if V_w[0] > w_x_max:
                w_x_max = V_w[0]
            if V_w[0] < w_x_min:
                w_x_min = V_w[0]
            if V_w[1] > w_y_max:
                w_y_max = V_w[1]
            if V_w[1] < w_x_min:
                w_y_min = V_w[1]
            #grid_map[V_w[0],V_w[1]] = probability[u,v]
    print w_x_max
    print w_x_min
    print w_y_max
    print w_y_min

    #return grid_map_single


'''
fusion of grid_map from multiple frames

'''


    
def main():
    img_path = "/home/yimo/Research/dataset/singleTestImages/Limg_141.png"
    Dimg_path = "/home/yimo/Research/dataset/singleTestImages/depth_141.png"
    Dimg_test = cv2.imread(Dimg_path)
    print Dimg_test.shape
    #cv2.imshow("Depth",Dimg_test)
    #cv2.waitKey(20000)
    Limg = load_img(img_path)
    Limg = img_to_array(Limg)
    Dimg = load_img(Dimg_path)
    Dimg = img_to_array(Dimg)
    print Dimg.shape
    mask = predict_single(Limg)
    scale_x = 1280/480
    scale_y = 720/360
    intrinsic_matrix = np.array([[139.268*scale_x,0.0,240.5*scale_x],[0.0,139.268*scale_y,180.5*scale_y],[0.0,0.0,1.0]])
    res = get_skeleton(mask)
    probability = traversability_vision(res,Limg)
    grid_transform(probability,Dimg_test,intrinsic_matrix)

if __name__ == '__main__':
    main()
