import argparse
from posixpath import basename
from statistics import mode

from numpy.core.arrayprint import printoptions
import onnxruntime
import cv2
import numpy as np
import os,sys
import glob
import torch
import time
import torchvision
import glog,copy
import shutil
import glob
from tqdm import tqdm
CAFFE_ROOT = '/home/disk/tanjing/ambacaffe'
if os.path.join(CAFFE_ROOT, 'python') not in sys.path:
    sys.path.insert(0, os.path.join(CAFFE_ROOT, 'python'))
import caffe

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



#-----------------landmarks_detect-------------------------------------------
model_def = "prnet128_256_focus_defocus_4d_stream.prototxt"
model_weights = "prnet128_exp154.caffemodel"
boxnet = caffe.Net(model_def,model_weights,caffe.TEST)

def detect_lmk(bbox,img):
    face_crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    point3d = get_point3d(boxnet,face_crop)
    point3d = point3d + np.array([int(bbox[0]),int(bbox[1]),0]).reshape(-1,3)
    for i,point2d in enumerate(point3d[:,0:2]):
        point2d = point2d.astype(int)
        cv2.circle(img,(point2d[0],point2d[1]),2,(0,255,0))

def get_lmk(bbox,img):
    points = []
    face_crop = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    point3d = get_point3d(boxnet,face_crop)
    point3d = point3d + np.array([int(bbox[0]),int(bbox[1]),0]).reshape(-1,3)
    for i,point2d in enumerate(point3d[:,0:2]):
        point2d = point2d.astype(int)
        points.append([point2d[0],point2d[1]])
    points = np.array(points)
    return points

def get_point3d(net,image):
    height,width,channel = image.shape
    image = cv2.resize(image,(128,128))  
    if image.ndim == 3 and channel == 3:
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if image.ndim==2:
        image=image[:,:,np.newaxis]
    net_input=(torch.from_numpy(np.transpose(image,(2,0,1))).unsqueeze(0).cuda()-128.0)/128.0   
    face_input_numpy = net_input.cpu().float().numpy().copy()
    net.blobs['data'].data[:] = face_input_numpy.copy()[:].astype('float64')
    net.forward()
    caffe_pred=net.blobs['interp10'].data.copy()[0]
    pos_map=np.transpose(caffe_pred,(1,2,0))

    uv_kpt_ind = np.fromfile('uv_kpt_ind_lm67.txt',sep=' ').reshape(2,-1).astype(int)
    lmk67_temp = pos_map[uv_kpt_ind[1,:], uv_kpt_ind[0,:], :]
    lmk67_3d=(lmk67_temp*np.array([width,height,1]).reshape(-1,3)).astype(int)

    return lmk67_3d


#---------------------NMS--------------------------------------------------------------------
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

#---------------------img_preprocess-----------------------------------------------------------------
def img_preprocess(frame,imgsz):
    img = cv2.resize(frame,(imgsz,imgsz))
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray = img_gray[:,:,np.newaxis]
    #img = np.concatenate((img_gray, img_gray, img_gray), axis=-1)
    img = img_gray[:, :, ::-1].transpose(2, 0, 1).astype('float32')[np.newaxis, :, :, :]
    img /= 255.0  
    return img

#---------------------face_detect----------------------------------------------------------------------------
face_model = "exp30.onnx"
session = onnxruntime.InferenceSession(face_model)
in_name = [input.name for input in session.get_inputs()][0]
out_name = [output.name for output in session.get_outputs()]

imgsz = 480
anchors = np.fromfile('face_480_priorbox_220527.txt',sep=' ')
anchors = anchors.reshape(-1,5)
def py_cpu_nms(dets0, conf_thresh, iou_thresh):
    """Pure Python NMS baseline."""
    nc = dets0.shape[1] - 5
    dets = dets0[dets0[:, 4] > conf_thresh]
    dets = xywh2xyxy(dets)
    
    keep_all = []
    for cls in range(nc):
        dets_single = dets[np.argmax(dets[:,5:],axis=1)==cls]
        #print('dets_single %d'%cls,dets_single)
        x1 = dets_single[:, 0]
        y1 = dets_single[:, 1]
        x2 = dets_single[:, 2]
        y2 = dets_single[:, 3]
        scores = dets_single[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)  
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        keep_rect = dets_single[keep]
        #print('keep',keep)
        keep_all.extend(keep_rect)
    return keep_all


def np_sigmoid(x):
    return 1.0/(1.0+1.0/np.exp(x))
    
def decode_output(pred_raw_data,anchor_txt):
    pred_raw_data = np_sigmoid(pred_raw_data)
    pred_raw_data[:, 0] = (pred_raw_data[:, 0] * 2. - 0.5 + anchor_txt[:, 0]) * anchor_txt[:, 4] #x
    pred_raw_data[:, 1] = (pred_raw_data[:, 1] * 2. - 0.5 + anchor_txt[:, 1]) * anchor_txt[:, 4] #y
    pred_raw_data[:, 2] = (pred_raw_data[:, 2] * 2) ** 2 * anchor_txt[:, 2]  # w
    pred_raw_data[:, 3] = (pred_raw_data[:, 3] * 2) ** 2 * anchor_txt[:, 3]  # h
    
    return pred_raw_data

def face_detect(img):
    pred = session.run(out_name,{in_name: img})
    x1 = np.array(pred[0]).reshape(-1, 6)
    x2 = np.array(pred[1]).reshape(-1, 6)
    x3 = np.array(pred[2]).reshape(-1, 6)
    out_data_raw = np.vstack((x1,x2,x3))
    output_from_txt = decode_output(out_data_raw,anchors)
    pred = py_cpu_nms(output_from_txt, 0.5, 0.45)
    return pred


def face_detect_1(img):
    pred = session.run(out_name,{in_name: img})
    
    grid = [torch.zeros(1)] * 3
    z = []
    shapelist = [60,30,15]
    for i in range(len(pred)):
        pred[i] = torch.from_numpy(pred[i]).to(device)
        pred[i] = torch.reshape(pred[i],(1,3,shapelist[i],shapelist[i],6))  
        bs, na, ny, nx, no = pred[i].size()
        if grid[i].shape[2:4] != pred[i].shape[2:4]:
            grid[i] = make_grid(nx, ny).to(device)

        y = pred[i].sigmoid()
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid[i].to(device)) * stride[i]  # xy
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]  # wh
        z.append(y.view(1, -1, no))
    pred = torch.cat(z, 1)
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45)[0]
    return pred

def make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def scale_ratio(each,frame,imgsz):
    ratio = (frame.shape[0] /imgsz , frame.shape[1] / imgsz)
    each[[0, 2]] *= ratio[1]
    each[[1, 3]] *= ratio[0]
    return each
#--------------------bbox_enlarger-------------------------------------------------------------------    
def bbox_enlarger(im0,bbox,scale):
    w = bbox[2]-bbox[0]+1
    h = bbox[3]-bbox[1]+1

    bbox[0] = bbox[0]-h*scale[0]
    bbox[2] = bbox[2]+w*scale[0]
    bbox[3] = bbox[3]+h*scale[1]
    
    np.clip(bbox[0], 0, im0.shape[1])
    np.clip(bbox[1], 0, im0.shape[0])
    np.clip(bbox[2], 0, im0.shape[1])
    np.clip(bbox[3], 0, im0.shape[0])
    print(bbox)
    return bbox
def bbox_enlarger(img_width,img_height,bbox,scale):
    w=bbox[2]-bbox[0]+1
    h=bbox[3]-bbox[1]+1
    bbox0=np.clip(bbox[0]-w*scale[0],0,img_width)
    bbox1=np.clip(bbox[1]-h*scale[1],0,img_height)
    bbox2=np.clip(bbox[2]+w*scale[0],0,img_width)
    bbox3=np.clip(bbox[3]+h*scale[1],0,img_height)
    return np.array([bbox0,bbox1,bbox2,bbox3])

#--------------------crop_calling_area-------------------------------------------------------------------
def bbox_from_points(points):#获取landmark外接矩形框
    max_=np.max(points,axis=0)
    min_=np.min(points,axis=0)
    return [min_[0],min_[1],max_[0],max_[1]]

def crop_calling_area(lmks,h,w):
    eye_index = [41,37,28,32]
    center = np.mean(lmks[eye_index,:],axis = 0) 
    face_bbox=bbox_from_points(lmks)
    bbox_w = face_bbox[2]-face_bbox[0]
    bbox_h = face_bbox[3]-face_bbox[1]
    x0 =  np.clip(face_bbox[0] - 1*bbox_w,0,w-1)
    y0 =  np.clip(face_bbox[1] - 0.5*bbox_h,0,h-1)
    x1 =  np.clip(face_bbox[2] + 1*bbox_w,0,w)
    y1=  np.clip(face_bbox[3] + 0.5*bbox_h,0,h)
    return [x0,y0,x1,y1]
#-----------------------face_model--------------------------------------------------------------------------    
face_model = "exp30.onnx"
session = onnxruntime.InferenceSession(face_model)
in_name = [input.name for input in session.get_inputs()][0]
out_name = [output.name for output in session.get_outputs()]

imgsz = 480
stride = [8, 16, 32] 
anchor_grid = torch.tensor([[[[[[10., 13.]]], [[[16., 30.]]], [[[33., 23.]]]]],
                      [[[[[30., 61.]]], [[[62., 45.]]], [[[59., 119.]]]]],
                      [[[[[116., 90.]]], [[[156., 198.]]], [[[373., 326.]]]]]]).to(device)
#-------------------------------------------------------
from PIL import Image, ImageDraw, ImageFont
def drawBBox(frame,bbox,name):
    cv2.rectangle(frame,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,255),2)
    cv2.putText(frame, name, (int(bbox[0]),int(bbox[1]-10)), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,255),1)
    cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)

    # for i in range(0,len(bbox)):
    font = ImageFont.truetype("calibrii_____.ttf", 50, encoding="utf-8")
    draw.text((int(bbox[0]),int(bbox[3]+20)), name, (0,0,255), font=font)

    frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return frame

#-----------------------pytorch_detection--------------------------------------------------------------------------    
def show_result(image,pred_det,pred_mask,threshold):
    orig_img = copy.deepcopy(image)
    COLORS = [(255, 255, 0), (0, 255, 0), (0, 0, 255)]
    crop_w,crop_h=image.shape[1],image.shape[0]
    mask_full=np.zeros(shape=(crop_w,crop_h),dtype='uint8')
    detections = pred_det.detach().cpu().numpy()
    if pred_mask is not None:
        pred_mask=np.argmax(pred_mask[0].detach().cpu().numpy(),axis=0)
    
    points=[]
    for i in range(detections.shape[1]):
        j = 0
        while detections[0, i, j, 0] >= threshold:
            pt = detections[0, i, j, 1:] * np.array([crop_w,crop_h,crop_w,crop_h])
            cv2.rectangle(image,(int(pt[0]), int(pt[1])),(int(pt[2]), int(pt[3])), COLORS[i % 3], 3)
            cv2.putText(image, "{:.2f}".format(detections[0, i, j, 0]*100 ,), (int(pt[0]), int(pt[1])),cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[i % 3], 1, cv2.LINE_AA)
            points.append([int(pt[0]), int(pt[1]),int(pt[2]), int(pt[3])])
            print(detections[0, i, j, 0]*100)
            j += 1
    return points,image,

def pytorch_detection(model,cult_fram,show):

    # image = cv2.cvtColor(cult_fram,cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(cult_fram,cv2.COLOR_BGR2YUV)
    image=image[:,:,0]
    image = cv2.resize(image,(128,128))[:,:,np.newaxis]
    image = torch.from_numpy(image*1.0-128.0).permute(2, 0, 1).float()


    net_input=image.unsqueeze(0).cuda()
    image=(image.permute(1, 2, 0).numpy()+128).astype('uint8').copy()

    pred_det,pred_mask,_=model(net_input)
    points,drawed_img = show_result(cult_fram,pred_det,pred_mask,threshold=0.8)
    if show:
        cv2.namedWindow('cult',cv2.WINDOW_NORMAL)
        cv2.imshow('cult',drawed_img)
 
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return True,points,drawed_img
    return False,points,drawed_img
#-----------------------caffe_detection-------------------------------------------------------------------------- 
def detect_phone(caffe_model,image_file, conf_thresh=0.8, topn=5):
    '''
    SSD detection
    '''
    img_gray=cv2.cvtColor(image_file,cv2.COLOR_BGR2GRAY)
    img_gray_resized=(cv2.resize(img_gray,(128,128))[:,:,np.newaxis]*1.0-128)
    net_input=np.transpose(img_gray_resized,(2,0,1))
    caffe_model.blobs['data'].data[...] = net_input

    # Forward pass.
    detections = caffe_model.forward(['detection_out','mbox_priorbox'])
    priorbox = caffe_model.blobs['mbox_conf_flatten'].data[:,:]
    belt_results=detections['detection_out']
    h,w,_=image_file.shape
    phone_boxes=[]

    for i in range(belt_results.shape[2]):
        prob=belt_results[0,0,i,2]
        if prob<conf_thresh:
            continue

        bbox=np.array([belt_results[0,0,i,3]*w,belt_results[0,0,i,4]*h,belt_results[0,0,i,5]*w,belt_results[0,0,i,6]*h])

        bbox_enlarged=bbox
        phone_boxes.append([bbox_enlarged,prob])

    return phone_boxes

def caffe_detection(caffe_model,cult_fram,show):
    height,width,ch=cult_fram.shape
    phone_boxes = detect_phone(caffe_model,cult_fram)
    priorbox = caffe_model.blobs['mbox_priorbox'].data[0,:,:] # 所有default box的归一化坐标
    points=[]
    if len(phone_boxes) !=0:
        for i in range(len(phone_boxes)):
            smoke_boxe=phone_boxes[i][0]
            prob = phone_boxes[i][1]
            prob = str("%.2f%%" % (prob * 100))

            smoke_boxe=bbox_enlarger(width,height,smoke_boxe,[0.0,0.0]).astype(int)
            drawBBox(cult_fram,smoke_boxe,prob)
            
            points.append(int(smoke_boxe[i]) for i in range(len(smoke_boxe)))
            if show:
                cv2.namedWindow('cult',cv2.WINDOW_NORMAL)
                cv2.imshow('cult',cult_fram)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    return True,points,cult_fram
    return True,points,cult_fram

#----------------------------------------------------------------------------------------------
def generate_txt(trainval_dir,test_dir=None,trainval_txt=None,test_txt=None):
    
    dist_img_dir = "JPEGImages"
    dist_anno_dir = "Annotations"

    trainval_img_lists = glob.glob(trainval_dir + '/*.jpg')
    trainval_img_names = []
    for item in trainval_img_lists:
        temp1, temp2 = os.path.splitext(os.path.basename(item))
        trainval_img_names.append(temp1)



    if trainval_txt:
        trainval_fd = open(trainval_txt, 'w')
        for item in tqdm(trainval_img_names):
            trainval_fd.write(dist_img_dir+'/'+str(item) + '.jpg '+dist_anno_dir+'/'+str(item) + '.xml\n')

    if test_dir:
        test_img_lists = glob.glob(test_dir + '/*.jpg')
        test_img_names = []
        for item in test_img_lists:
            temp1, temp2 = os.path.splitext(os.path.basename(item))
            test_img_names.append(temp1)

        test_fd = open(test_txt, 'w')
        for item in tqdm(test_img_names):
            test_fd.write(dist_img_dir+'/'+str(item) + '.jpg '+dist_anno_dir+'/'+str(item) + '.xml\n')
#----------------------------------------------------------------------------------------------
def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects
            (top, left, bottom, right)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
  
    # computing area of each rectangles
    S_rec1 = (rec1[3] - rec1[1]) * (rec1[2] - rec1[0])
    S_rec2 = (rec2[3] - rec2[1]) * (rec2[2] - rec2[0])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0

#----------------------------------------------------------------------------------------------

def phone_detection(model_type,model,frame,show):
    if model_type=='pytorch':
        flag , points , drawed_img = pytorch_detection(model,frame,show)
    elif model_type=='caffe':
        flag , points , drawed_img =  caffe_detection(model,frame,show)
    return flag , points , drawed_img
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-model_type','--model_type',default='caffe',
                        help='caffe or pytorch model',type=str)
    parser.add_argument('-Lowrank','--Lowrank', dest='Lowrank',default=False,
                    help='pytorch Lowrank model',type=bool)
    parser.add_argument('-Binary','--Binary', dest='Binary',default=False,
                    help='pytorch Binary model',type=bool)
    parser.add_argument('-Purning','--Purning', dest='Purning',default=False,
                    help='pytorch Purned model',type=bool)
    parser.add_argument('-ip_camera','--ip_camera',dest='ip_camera',default=False,
                    help='use ip_camera',type=bool)
    parser.add_argument('-show_video','--show_video', dest='show_video',default=False,
                    help='show_video',type=bool)
    parser.add_argument('-croped_pic','--croped_pic', dest='croped_pic',default=False,
                    help='using croped pic to test',type=bool)


    parser.add_argument('--weights', nargs='+', type=str, default='/home/disk/qizhongpei/projects/yolov7/runs/train/yolov7-tiny3/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    args = parser.parse_args()

    return args

args = parse_args()
if args.model_type=='pytorch':
    sys.path.append('/home/disk/qizhongpei/ssd_pytorch/')
    from caffe_models.vgg2.phone_128_vgg_float_3 import phone_128_vgg_float
    from cnnc_util import torch_lowrank_layers,torch_binarize_layers,torch_mergebn_layers
    
    # Float
    model=phone_128_vgg_float()
    #snapshot_path ='/home/disk/yenanfei/OMS_phone/weights/float/OMS_phone_128_vgg_float_best.pth'
    snapshot_path ='/home/disk/qizhongpei/ssd_pytorch/weights/phone_best.pth'
    # snapshot_path ="/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/recult_dataset/uploaded_version/vgg_float/phone_128_vgg_float_ssd_best.pth"
    state_dict=torch.load(snapshot_path,map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict,strict=True)

    # Lowrank
    if args.Lowrank:
        torch_lowrank_layers(model.cpu(),percentale=0.9,has_bn=False)
        lowrank_snapshot_path = '/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/recult_dataset/uploaded_version/lowrank/phone_128_vgg_float_lowrank_newest1.pth'
        model.load_state_dict(torch.load(lowrank_snapshot_path),strict=True)
        glog.info("restore {}".format(lowrank_snapshot_path))
        
    # Binary
    if args.Binary:
        exclude_layer=['conv4_3_norm_mbox_loc','conv4_3_norm_mbox_conf','fc7_mbox_loc','fc7_mbox_conf',\
            'conv6_2_mbox_loc','conv6_2_mbox_conf','conv7_2_mbox_loc','conv7_2_mbox_conf','conv8_2_mbox_loc','conv8_2_mbox_conf']
        torch_binarize_layers(model.cpu())
        # torch_mergebn_layers(model)
        snapshot_path='/home/disk/yenanfei/DMS_phone/phone_model_pytorch/snapshot/recult_dataset/binary/phone_128_vgg_float_binary_newest20_ok.pth' 

        model.load_state_dict(torch.load(snapshot_path),strict=True)
        glog.info("restore {}".format(snapshot_path))
    
    # purning
    if args.Purning:
        pruning_snapshot_path = '/home/disk/yenanfei/OMS_phone/weights/purning/OMS_phone_128_vgg_purne_0.96_best.pth'
        state_dict = torch.load(pruning_snapshot_path)
        for name in state_dict.keys():
            if 'weight' in name:
                mask = name.replace('weight','mask')
                if mask in state_dict.keys():
                    state_dict[name]=state_dict[name]*state_dict[mask]
        model.load_state_dict(state_dict,strict=False)

        glog.info("restore {}".format(snapshot_path))

    model=model.cuda()
    model.eval()

if args.model_type=='caffe':
    #model=caffe.Net('/home/disk/yenanfei/DMS_phone/ssd_pytorch/phone_128_vgg_float_prune_96%.prototxt','/home/disk/yenanfei/DMS_phone/ssd_pytorch/phone_128_vgg_float_prune_96%.caffemodel',caffe.TEST)
    model=caffe.Net('/home/disk/qizhongpei/ssd_pytorch/test.prototxt','/home/disk/qizhongpei/ssd_pytorch/test.caffemodel',caffe.TEST)

#video_paths = list(glob.iglob('/home/disk/yenanfei/DMS_phone/PhoneDataset_recut/JPEGImages/*', recursive=True))
video_paths = list(glob.iglob('/home/disk/qizhongpei/projects/yolov7/inference/images/call/*', recursive=True))

rootdir = '/home/disk/qizhongpei/projects/yolov7/inference/images/call/'
#rootdir = '/home/disk/qizhongpei/projects/yolov7/inference/images/call_/'
list = os.listdir (rootdir)

for i,video_path in enumerate(tqdm(video_paths)):
    count = 0
    cap = cv2.VideoCapture(video_path)
    #while(True):
    ret, frame = cap.read()
    if (not ret):
        break     
    height,width,ch=frame.shape
    
    frame_ = frame.copy()
    if args.croped_pic:
        flag , points , drawed_img =phone_detection(args.model_type,model,frame,show=True)

    else:
        img = img_preprocess(frame,imgsz)
        pred = face_detect(img)   
    
        for n,each in enumerate(pred):
            
            each = scale_ratio(each,frame,imgsz)
            bbox=bbox_enlarger(frame.shape[1],frame.shape[0],each,scale=[0.1,0.1]).astype(int)

            if bbox[0]==bbox[2] or bbox[1]==bbox[3]:
                continue
            lmks = get_lmk(bbox,frame)
            area = crop_calling_area(lmks,height,width)
            cult_fram = frame[int(area[1]):int(area[3]),int(area[0]):int(area[2])]

            if cult_fram is not None:
                cult_fram_ = cult_fram.copy()
                # cv2.imshow("cult_fram_",cult_fram_)
                # cv2.waitKey(0)
                
                # for i in range(0, len(list)) :
           
            cv2.imwrite('/home/disk/qizhongpei/projects/yolov7/inference/images/crop/'+ f"{n}"+ "_"+list[i], cult_fram_)

         
                
#--------------------------------------------phone_detect----------------------------------------------    
source, weights, view_img, save_txt, imgsz_, trace = args.source, args.weights, args.view_img, args.save_txt, args.img_size, not args.no_trace
save_img = not args.nosave and not source.endswith('.txt')  # save inference images
webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    ('rtsp://', 'rtmp://', 'http://', 'https://'))

# Directories
save_dir = Path(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Initialize
set_logging()
device = select_device(args.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz_ = check_img_size(imgsz_, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, args.img_size)

if half:
    model.half()  # to FP16




dataset = LoadImages('/home/disk/qizhongpei/projects/yolov7/inference/images/crop/', img_size=128, stride=stride)

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz_, imgsz_).to(device).type_as(next(model.parameters())))  # run once
t0 = time.time()
for path, img, im0s, vid_cap in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=args.augment)[0]
    
    # Apply NMS
    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=args.classes, agnostic=args.agnostic_nms)
    t2 = time_synchronized()
    #print(len(pred))

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        #print(f'{s}Done. ({t2 - t1:.3f}s)')

        # Stream results
        if view_img:
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                #print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

if save_txt or save_img:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #print(f"Results saved to {save_dir}{s}")

print(f'Done. ({time.time() - t0:.3f}s)')


                
                   
                    

                                    #cv2.destroyallwindow()
                                
                    #                 flag , points , drawed_img =phone_detection(args.model_type,model,cult_fram_,show=True)
                    #                 frame[int(area[1]):int(area[3]),int(area[0]):int(area[2])] = drawed_img
                    #             if flag:
                    #                 break
    #cap.release()



