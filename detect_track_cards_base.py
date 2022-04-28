import argparse
import threading
import os
import platform
import shutil
import time
from pathlib import Path
from enum import Enum

import cv2
from matplotlib.pyplot import hist
import torch
import torch.backends.cudnn as cudnn
from numpy import append, random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

import cvzone
##
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
test_list = []
imageOverlayBJ = cv2.imread("images/blackjack.png", cv2.IMREAD_UNCHANGED)  ##
imageOverlayLose = cv2.imread("images/lose.png", cv2.IMREAD_UNCHANGED)  ##
imageOverlayWin = cv2.imread("images/win.png", cv2.IMREAD_UNCHANGED)  ##
imageOverlayTie = cv2.imread("images/tie.png", cv2.IMREAD_UNCHANGED)  ##

#Blackjack global variable decleration
dealer = []
dealer_num = []
player = []
playerH2 = []
playerextra = []
player_num = []
player_total = 0
bj_double = False
wallet = 1000
can_split = False
split = False
can_double = False
double = False


def should_double(hand,dealer,p_soft):
    global player_total
    if(p_soft):
        if hand[0][0] == 'A':
            i = 1
        else:
            i = 0
        if hand[i][0] == '8' and dealer[0][0] == '6':
            return True
        elif hand[i][0] == '7' and 1 < int(dealer[0][0]) < 7:
            return True
        elif hand[i][0] == '6' and 2 < int(dealer[0][0]) < 7:
            return True
        elif hand[i][0] == '5' and 3 < int(dealer[0][0]) < 7:
            return True
        elif hand[i][0] == '4' and 3 < int(dealer[0][0]) < 7:
            return True
        elif hand[i][0] == '3' and 4 < int(dealer[0][0]) < 7:
            return True
        elif hand[i][0] == '2' and 4 < int(dealer[0][0]) < 7:
            return True
        else:
            return False
    else:
        if player_total == 11:
            return True
        elif player_total == 10 and 1 < int(dealer[0][0]) <= 9 :
            return True
        elif player_total == 9 and 2 < int(dealer[0][0]) < 7:
            return True
        else:
            return False
        

def should_split(hand,dealer):
    if hand[0][0] == 'A' and hand[1][0] == 'A':
        return True
    elif hand[0][0] == '9' and hand[1][0] == '9' and dealer[0][0] != '7' and 1 < int(dealer[0][0]) <= 9 :
        return True
    elif hand[0][0] == '8' and hand[1][0] == '8':
        return True
    elif hand[0][0] == '7' and hand[1][0] == '7' and 1 < int(dealer[0][0]) < 8:
        return True
    elif hand[0][0] == '6' and hand[1][0] == '6' and 1 < int(dealer[0][0]) < 7:
        return True
    elif hand[0][0] == '4' and hand[1][0] == '4' and 4 < int(dealer[0][0]) < 7:
        return True
    elif hand[0][0] == '3' and hand[1][0] == '3' and 1 < int(dealer[0][0]) < 8:
        return True
    elif hand[0][0] == '2' and hand[1][0] == '2' and 1 < int(dealer[0][0]) < 8:
        return True
    else:
        return False

def should_hit(p_soft):
    global player_total
    global dealer_num
    if(p_soft):
        if player_total < 18:
            return True
        elif player_total == 18 and dealer_num > 8:
            return True
        else:
            return False
    else:
        if player_total < 11:
            return True
        elif player_total == 12 and dealer_num != 4 and dealer_num != 5 and dealer_num != 6:
            return True
        elif player_total == 13 and dealer_num > 6:
            return True
        elif player_total == 14 and dealer_num > 6:
            return True
        elif player_total == 15 and dealer_num > 6:
            return True
        elif player_total == 16 and dealer_num > 6:
            return True
        else:
            return False

def Neural_Network_hit(hand,p_soft):
    global dealer_num
    if p_soft:
        if hand < 18:
            return True
        if hand == 18 and dealer_num != 5 and dealer_num != 6 and dealer_num != 7 and dealer_num != 8 : 
            return True
        if hand == 19 and dealer_num == 11:
            return True
        else:
            return False
    else:
        if hand < 16:
            return True
        if hand == 16 and dealer_num != 5 and dealer_num != 6 and dealer_num != 7 and dealer_num != 8 : 
            return True
        if hand == 17 and dealer_num > 7:
            return True
        if hand == 18 and dealer_num > 8:
            return True
        if hand == 19 and dealer_num == 11:
            return True
        else:
            return False
    

def cards_to_numbers(hand):
    global dealer_num
    soft = False
    for x in hand:
        if x[0].isdigit():
            if x[0] == '1':
                dealer_num.append(10)
            else:
                dealer_num.append(int(x[0])) #this determines the value of the number
        else: #king queen jack or ace
            if x[0] == 'A':
                soft = True
                dealer_num.append(11)
            elif x[0] == 'J' or 'Q' or 'K':
                dealer_num.append(10)
    print(dealer_num)
    dealer_num = sum(dealer_num)
    if dealer_num > 21 and soft:
        dealer_num = dealer_num - 10
    else:
        dealer_num = dealer_num

def split_cards(xycord,labels):
    global bj_double
    global wallet
    global split
    global can_split
    global can_double
    for i,y in zip(xycord,range(len(xycord))):
        if i[0]<400:
            if(not dealer.count(labels[y])):
                    dealer.append(labels[y])
        else:
            if(split):
                if (not player.count(labels[y]) and not playerH2.count(labels[y])):
                    if(i[1]<300):
                        player.append(labels[y])
                    elif(i[1]>300):
                        playerH2.append(labels[y])
            else:
                if (not player.count(labels[y])):
                    player.append(labels[y])
                    if(can_split and not split):
                        if i[0]>700 and i[1]<480:
                            wallet = wallet-100
                            split=True
                    if(can_double and not bj_double):
                        if i[0]>700 and i[1]<120:
                            wallet = wallet-100
                            bj_double=True

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person  #BGR
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)

    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_boxes(img, bbox, object_id, identities=None, offset=(0, 0)):
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        # print("box muner", i)
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0
        # create new buffer for new object
        if id not in data_deque:
          data_deque[id] = deque(maxlen= opt.trailslen)
        color = compute_color_for_labels(object_id[i])
        # add center to buffer
        data_deque[id].appendleft(center)

        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue

            # generate dynamic thickness of trails
            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)

            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    return img

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    global bj_double
    global player
    global playerH2
    global dealer
    global dealer_num
    global wallet
    global split
    global player_total
    global can_split
    global can_double
    bet_made = False
    game_over = True
    game_overH2 = True
    strat = ''
    strat2 = ''
    blackjack_cal = 0
    blackjackH2_cal = 0
    dealer_cal = 0
    playerH2_total = 0
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    # attempt_download("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(opt.cfg, opt.img_size)  # .cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)  # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # money in the pot
            cv2.line(im0, (0, 580), (250, 580), [255, 0, 0], 30)
            cv2.putText(im0, 'wallet: $' + str(wallet), (5, 590), 0, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywh_bboxs = []
                confs = []
                oids = []
                xycord = []
                #Instantiate List
                labels = [] ##

                # Write results
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    xycord.append([x_c,y_c])
                    confs.append([conf.item()])
                    label = '%s' % (names[int(cls)])
                    color = compute_color_for_labels(int(cls))
                    UI_box(xyxy, im0, label=label, color=color, line_thickness=2)
                    oids.append(int(cls))
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    labels.append(label)
                    ##---Append Labels to List--##

#This section devides the screen 0 is the x axis and 1 is the y axis, numbers are pixle measurements
                player.clear()
                playerH2.clear()
                dealer.clear()
                split_cards(xycord,labels)
#end of section


                ##---Convert String in List to Integers--##
                labels = list(dict.fromkeys(labels))

                print("player: \n" + str(player))
                print("dealer: \n" + str(dealer))

                print(labels)
                
                p_soft = False
                p_softH2 = False
                dealer_num = []
                player_num = []
                playerH2_num = []
                cards_to_numbers(dealer)

                if (len(player) + len(dealer)) == 1 and game_over:
                    dealer_cal = 0
                    blackjack_cal = 0
                    blackjackH2_cal = 0
                    game_over = False
                    game_overH2 = False
                    if(split):
                        split = False
                    if bet_made:
                        bet_made = False
                    if bj_double:
                        bj_double = False
                    print('game reset')

                if (bj_double):
                    if(len(player)<4):
                        for x in player:
                            if x[0].isdigit():
                                if x[0] == '1':
                                    player_num.append(10)
                                else:
                                    player_num.append(int(x[0]))  # this determines the value of the number
                            else:  # king queen jack or ace
                                if x[0] == 'A':
                                    p_soft = True
                                    player_num.append(11)
                                elif x[0] == 'J' or 'Q' or 'K':
                                    player_num.append(10)
                        player_total = sum(player_num)
                        if player_total > 21 and p_soft:
                            player_total = player_total - 10
                        else:
                            player_total = player_total
                else:
                    for x in player:
                        if x[0].isdigit():
                            if x[0] == '1':
                                player_num.append(10)
                            else:
                                player_num.append(int(x[0]))  # this determines the value of the number
                        else:  # king queen jack or ace
                            if x[0] == 'A':
                                p_soft = True
                                player_num.append(11)
                            elif x[0] == 'J' or 'Q' or 'K':
                                player_num.append(10)
                    player_total = sum(player_num)
                    if player_total > 21 and p_soft:
                        player_total = player_total - 10
                    else:
                        player_total = player_total
                    
                    if (split):
                        for x in playerH2:
                            if x[0].isdigit():
                                if x[0] == '1':
                                    playerH2_num.append(10)
                                else:
                                    playerH2_num.append(int(x[0]))  # this determines the value of the number
                            else:  # king queen jack or ace
                                if x[0] == 'A':
                                    p_softH2 = True
                                    playerH2_num.append(11)
                                elif x[0] == 'J' or 'Q' or 'K':
                                    playerH2_num.append(10)
                        playerH2_total = sum(playerH2_num)
                        if playerH2_total > 21 and p_softH2:
                            playerH2_total = playerH2_total - 10
                        else:
                            playerH2_total = playerH2_total
                if not game_over:
                    strat = ''
                    can_double = False
                    can_split = False
                    if(split):
                        if Neural_Network_hit(player_total,p_soft):
                            strat = 'hit'
                        else:
                            strat = 'stand'
                        if Neural_Network_hit(playerH2_total,p_softH2):
                            strat2 = 'hit'
                        else:
                            strat2 = 'stand'
                    elif (len(player) == 2 and len(dealer) == 1):
                        if should_split(player,dealer) and not bj_double and not split:
                            cv2.rectangle(im0, (700, 480), (800, 540), [85, 45, 255], 15)
                            cv2.putText(im0, 'Split', (700, 545), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                            can_split = True
                            strat = 'split'
                        elif should_double(player,dealer,p_soft) and strat == ''  and not bj_double and not split:
                            strat = 'double'
                            can_double = True
                            cv2.rectangle(im0, (700, 120), (800, 60), [85, 45, 255], 15)
                            cv2.putText(im0, 'Double', (700, 120), 0, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                        elif should_hit(p_soft) and strat == '':
                            strat = 'hit'
                        else:
                            strat = 'stand'
                    else:
                        if Neural_Network_hit(player_total,p_soft):
                            strat = 'hit'
                        else:
                            strat = 'stand'
                ##BlackJack Logic Calculation
                    if not bet_made:
                        wallet = wallet-100
                        bet_made = True
                    blackjack_cal = player_total
                    blackjackH2_cal = playerH2_total
                    dealer_cal = dealer_num

                    # PlayerBlackjack logic
                    if player_total > 21:
                        strat = 'Bust'
                        game_over = True
                    
                    if(split):
                        # PlayerBlackjack logic
                        if playerH2_total > 21:
                            strat2 = 'Bust'
                            game_overH2 = True
                        elif playerH2_total == 10:
                            strat2 = "double"
                        elif playerH2_total > 16:
                            strat2 = 'stand'
                        elif playerH2_total < 17:
                            strat2 = 'hit'

                    #Dealer Blackjack Logic
                    if not len(dealer) < 1 and dealer_num > 17:
                        if(not game_over):
                            if (player_total < 21 and player_total > dealer_num) or dealer_num > 21:
                                #im0 = cvzone.overlayPNG(im0, imageOverlayWin, [178, 148])
                                strat = "Won"
                                game_over = True
                                wallet = wallet + 200
                                if bj_double:
                                    wallet = wallet + 200
                            elif player_total == dealer_num:
                                #im0 = cvzone.overlayPNG(im0, imageOverlayTie, [178, 148])
                                strat = "Drew"
                                game_over = True
                                wallet = wallet + 100
                                if bj_double:
                                    wallet = wallet + 100
                            elif player_total == 21:
                                #im0 = cvzone.overlayPNG(im0, imageOverlayBJ, [178, 148])
                                strat = "Blackjack"
                                game_over = True
                                wallet = wallet + 250
                                if bj_double:
                                    wallet = wallet + 250
                            elif player_total < dealer_num :
                                #im0 = cvzone.overlayPNG(im0, imageOverlayBJ, [178, 148])
                                strat = "Lose"
                                game_over = True
                        if not game_overH2:    
                            if (playerH2_total < 21 and playerH2_total > dealer_num) or dealer_num > 21:
                                #im0 = cvzone.overlayPNG(im0, imageOverlayWin, [178, 148])
                                strat2 = "Won"
                                game_overH2 = True
                                wallet = wallet + 200
                            elif playerH2_total == dealer_num:
                                #im0 = cvzone.overlayPNG(im0, imageOverlayTie, [178, 148])
                                strat2 = "Drew"
                                game_overH2 = True
                                wallet = wallet + 100
                            elif playerH2_total == 21:
                                #im0 = cvzone.overlayPNG(im0, imageOverlayBJ, [178, 148])
                                strat2 = "Blackjack"
                                game_overH2 = True
                                wallet = wallet + 250
                            elif playerH2_total < dealer_num :
                                #im0 = cvzone.overlayPNG(im0, imageOverlayBJ, [178, 148])
                                strat2 = "Lose"
                                game_overH2 = True
                            

                if (split):
                    cv2.line(im0, (510, 25), (800, 25), [255,125,0], 30)
                    cv2.putText(im0, 'Hand One Has ' + str(blackjack_cal), (510, 35), 0, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                    
                    cv2.line(im0, (550, 280), (800, 280), [255,125,0], 30)
                    cv2.putText(im0, 'Hand One ' + strat, (550,290), 0, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                
                    cv2.line(im0, (510, 325), (800, 325), [255,0,0], 30)
                    cv2.putText(im0, 'Hand Two Has ' + str(blackjackH2_cal), (510, 335), 0, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                
                    
                    cv2.line(im0, (550, 580), (800, 580), [255,0,0], 30)
                    cv2.putText(im0, 'Hand Two ' + strat2, (555, 590), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                
                else:
                    cv2.line(im0, (550, 25), (800, 25), [255, 0, 0], 30)
                    cv2.putText(im0, 'Player has ' + str(blackjack_cal), (555, 35), 0, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)

                    #player stratigy display
                    cv2.line(im0, (550, 580), (800, 580), [233, 94, 22], 30)
                    cv2.putText(im0, 'Player ' + strat, (555, 590), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                cv2.line(im0, (0, 25), (250, 25), [85, 45, 255], 30)
                cv2.putText(im0, 'Dealer has ' + str(dealer_cal), (5, 35), 0, 1, [225, 255, 255], thickness=2,lineType=cv2.LINE_AA)
                

                ##----------------------------------------------------------------------------------##

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)

                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, object_id,identities)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # Stream results
            if view_img:

                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')

    opt = parser.parse_args()
    print(opt)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()