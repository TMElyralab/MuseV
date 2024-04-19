import torch
import cv2
import numpy as np
import ffmpeg
import argparse
import time
import traceback
import scipy.signal as signal
import copy
from controlnet_aux import DWposeDetector
from controlnet_aux.dwpose import pose2map
from pprint import pprint

from pose_utils import get_video_meta_info, size_calculate, warpAffine_kps, Reader, Writer



'''
    Detect dwpose from img, then align it by scale parameters
    img: frame from the pose video
    detector: DWpose
    scales: scale parameters
'''
def align_img(img, pose_ori, scales, detect_resolution, image_resolution):

    body_pose = copy.deepcopy(pose_ori['bodies']['candidate'])
    hands = copy.deepcopy(pose_ori['hands'])
    faces = copy.deepcopy(pose_ori['faces'])

    '''
    计算逻辑:
    0. 该函数内进行绝对变换，始终保持人体中心点 body_pose[1] 不变
    1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
    2. 用点在图中的实际坐标来计算。
    3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
    4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
    注意：dwpose 输出是 (w, h)
    '''

    # h不变，w缩放到原比例
    H_in, W_in, C_in = img.shape 
    video_ratio = W_in / H_in
    body_pose[:, 0]  = body_pose[:, 0] * video_ratio
    hands[:, :, 0] = hands[:, :, 0] * video_ratio
    faces[:, :, 0] = faces[:, :, 0] * video_ratio

    # scales of 10 body parts 
    scale_neck      = scales["scale_neck"] 
    scale_face      = scales["scale_face"]
    scale_shoulder  = scales["scale_shoulder"]
    scale_arm_upper = scales["scale_arm_upper"]
    scale_arm_lower = scales["scale_arm_lower"]
    scale_hand      = scales["scale_hand"]
    scale_body_len  = scales["scale_body_len"]
    scale_leg_upper = scales["scale_leg_upper"]
    scale_leg_lower = scales["scale_leg_lower"]

    scale_sum = 0
    count = 0
    scale_list = [scale_neck, scale_face, scale_shoulder, scale_arm_upper, scale_arm_lower, scale_hand, scale_body_len, scale_leg_upper, scale_leg_lower]
    for i in range(len(scale_list)):
        if not np.isinf(scale_list[i]):
            scale_sum = scale_sum + scale_list[i]
            count = count + 1
    for i in range(len(scale_list)):
        if np.isinf(scale_list[i]):   
            scale_list[i] = scale_sum/count



    # offsets of each part 
    offset = dict()
    offset["14_15_16_17_to_0"] = body_pose[[14,15,16,17], :] - body_pose[[0], :] 
    offset["3_to_2"] = body_pose[[3], :] - body_pose[[2], :] 
    offset["4_to_3"] = body_pose[[4], :] - body_pose[[3], :] 
    offset["6_to_5"] = body_pose[[6], :] - body_pose[[5], :] 
    offset["7_to_6"] = body_pose[[7], :] - body_pose[[6], :] 
    offset["9_to_8"] = body_pose[[9], :] - body_pose[[8], :] 
    offset["10_to_9"] = body_pose[[10], :] - body_pose[[9], :] 
    offset["12_to_11"] = body_pose[[12], :] - body_pose[[11], :] 
    offset["13_to_12"] = body_pose[[13], :] - body_pose[[12], :] 
    offset["hand_left_to_4"] = hands[1, :, :] - body_pose[[4], :]
    offset["hand_right_to_7"] = hands[0, :, :] - body_pose[[7], :]

    # neck
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_neck)

    neck = body_pose[[0], :] 
    neck = warpAffine_kps(neck, M)
    body_pose[[0], :] = neck

    # body_pose_up_shoulder
    c_ = body_pose[0]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_face)

    body_pose_up_shoulder = offset["14_15_16_17_to_0"] + body_pose[[0], :]
    body_pose_up_shoulder = warpAffine_kps(body_pose_up_shoulder, M)
    body_pose[[14,15,16,17], :] = body_pose_up_shoulder

    # shoulder 
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_shoulder)

    body_pose_shoulder = body_pose[[2,5], :] 
    body_pose_shoulder = warpAffine_kps(body_pose_shoulder, M) 
    body_pose[[2,5], :] = body_pose_shoulder

    # arm upper left
    c_ = body_pose[2]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["3_to_2"] + body_pose[[2], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[3], :] = elbow

    # arm lower left
    c_ = body_pose[3]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["4_to_3"] + body_pose[[3], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[4], :] = wrist

    # hand left
    c_ = body_pose[4]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_left_to_4"] + body_pose[[4], :]
    hand = warpAffine_kps(hand, M)
    hands[1, :, :] = hand

    # arm upper right
    c_ = body_pose[5]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_upper)
 
    elbow = offset["6_to_5"] + body_pose[[5], :]
    elbow = warpAffine_kps(elbow, M)
    body_pose[[6], :] = elbow

    # arm lower right
    c_ = body_pose[6]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_arm_lower)
 
    wrist = offset["7_to_6"] + body_pose[[6], :]
    wrist = warpAffine_kps(wrist, M)
    body_pose[[7], :] = wrist

    # hand right
    c_ = body_pose[7]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_hand)
 
    hand = offset["hand_right_to_7"] + body_pose[[7], :]
    hand = warpAffine_kps(hand, M)
    hands[0, :, :] = hand

    # body len
    c_ = body_pose[1]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_body_len)

    body_len = body_pose[[8,11], :] 
    body_len = warpAffine_kps(body_len, M)
    body_pose[[8,11], :] = body_len

    # leg upper left
    c_ = body_pose[8]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["9_to_8"] + body_pose[[8], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[9], :] = knee

    # leg lower left
    c_ = body_pose[9]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["10_to_9"] + body_pose[[9], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[10], :] = ankle

    # leg upper right
    c_ = body_pose[11]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_upper)
 
    knee = offset["12_to_11"] + body_pose[[11], :]
    knee = warpAffine_kps(knee, M)
    body_pose[[12], :] = knee

    # leg lower right
    c_ = body_pose[12]
    cx = c_[0]
    cy = c_[1]
    M = cv2.getRotationMatrix2D((cx,cy), 0, scale_leg_lower)
 
    ankle = offset["13_to_12"] + body_pose[[12], :]
    ankle = warpAffine_kps(ankle, M)
    body_pose[[13], :] = ankle

    # none part
    body_pose_none = pose_ori['bodies']['candidate'] == -1.
    hands_none = pose_ori['hands'] == -1.
    faces_none = pose_ori['faces'] == -1.

    body_pose[body_pose_none] = -1.
    hands[hands_none] = -1. 
    nan = float('nan')
    if len(hands[np.isnan(hands)]) > 0:
        print('nan')
    faces[faces_none] = -1.

    # last check nan -> -1.
    body_pose = np.nan_to_num(body_pose, nan=-1.)
    hands = np.nan_to_num(hands, nan=-1.)
    faces = np.nan_to_num(faces, nan=-1.)

    # return
    pose_align = copy.deepcopy(pose_ori)
    pose_align['bodies']['candidate'] = body_pose
    pose_align['hands'] = hands
    pose_align['faces'] = faces

    return pose_align



def run_align_video_with_filterPose_translate_smooth(args):

    vidfn=args.vidfn
    imgfn_refer=args.imgfn_refer
    outfn_ref_img_pose=args.outfn_ref_img_pose
    outfn=args.outfn

    reader = Reader(args, vidfn)
    audio = reader.get_audio()
    height, width = reader.get_resolution()
    fps = reader.get_fps()
    print(audio)
    print(height)
    print(width)
    print(fps)

    H_in, W_in  = height, width
    H_out, W_out = size_calculate(H_in,W_in,args.detect_resolution) 
    H_out, W_out = size_calculate(H_out,W_out,args.image_resolution) 
    WH_out = np.array([W_out,H_out])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DWposeDetector(device=device)

    refer_img = cv2.imread(imgfn_refer)
    output_refer, pose_refer = detector(refer_img,detect_resolution=args.detect_resolution, image_resolution=args.image_resolution, output_type='cv2',return_pose_dict=True)
    body_ref_img  = pose_refer['bodies']['candidate']
    hands_ref_img = pose_refer['hands']
    faces_ref_img = pose_refer['faces']
    output_refer = cv2.cvtColor(output_refer, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outfn_ref_img_pose,output_refer)
    

    skip_frames = 0
    max_frame = args.max_frame
    pose_list, video_frame_buffer, video_pose_buffer = [], [], []
    for i in range(max_frame):
        img = reader.get_frame()
        if img is None: 
            break 
        else: 
            video_frame_buffer.append(img)

        if i < skip_frames:
            continue
       
        # estimate scale parameters by the 1st frame in the video
        if i==skip_frames:
            output_1st_img, pose_1st_img = detector(img, args.detect_resolution, args.image_resolution, output_type='cv2', return_pose_dict=True)
            body_1st_img  = pose_1st_img['bodies']['candidate']
            hands_1st_img = pose_1st_img['hands']
            faces_1st_img = pose_1st_img['faces']

            '''
            计算逻辑:
            1. 先把 ref 和 pose 的高 resize 到一样，且都保持原来的长宽比。
            2. 用点在图中的实际坐标来计算。
            3. 实际计算中，把h的坐标归一化到 [0, 1],  w为[0, W/H]
            4. 由于 dwpose 的输出本来就是归一化的坐标，所以h不需要变，w要乘W/H
            注意：dwpose 输出是 (w, h)
            '''
            
            # h不变，w缩放到原比例
            ref_H, ref_W = refer_img.shape[0], refer_img.shape[1]
            ref_ratio = ref_W / ref_H
            body_ref_img[:, 0]  = body_ref_img[:, 0] * ref_ratio
            hands_ref_img[:, :, 0] = hands_ref_img[:, :, 0] * ref_ratio
            faces_ref_img[:, :, 0] = faces_ref_img[:, :, 0] * ref_ratio

            video_ratio = width / height
            body_1st_img[:, 0]  = body_1st_img[:, 0] * video_ratio
            hands_1st_img[:, :, 0] = hands_1st_img[:, :, 0] * video_ratio
            faces_1st_img[:, :, 0] = faces_1st_img[:, :, 0] * video_ratio

            # scale
            align_args = dict()
            
            dist_1st_img = np.linalg.norm(body_1st_img[0]-body_1st_img[1])   # 0.078   
            dist_ref_img = np.linalg.norm(body_ref_img[0]-body_ref_img[1])   # 0.106
            align_args["scale_neck"] = dist_ref_img / dist_1st_img  # align / pose = ref / 1st

            dist_1st_img = np.linalg.norm(body_1st_img[16]-body_1st_img[17])
            dist_ref_img = np.linalg.norm(body_ref_img[16]-body_ref_img[17])
            align_args["scale_face"] = dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[5])  # 0.112
            dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[5])  # 0.174
            align_args["scale_shoulder"] = dist_ref_img / dist_1st_img  

            dist_1st_img = np.linalg.norm(body_1st_img[2]-body_1st_img[3])  # 0.895
            dist_ref_img = np.linalg.norm(body_ref_img[2]-body_ref_img[3])  # 0.134
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[5]-body_1st_img[6])
            dist_ref_img = np.linalg.norm(body_ref_img[5]-body_ref_img[6])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_arm_upper"] = (s1+s2)/2 # 1.548

            dist_1st_img = np.linalg.norm(body_1st_img[3]-body_1st_img[4])
            dist_ref_img = np.linalg.norm(body_ref_img[3]-body_ref_img[4])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[6]-body_1st_img[7])
            dist_ref_img = np.linalg.norm(body_ref_img[6]-body_ref_img[7])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_arm_lower"] = (s1+s2)/2

            # hand
            dist_1st_img = np.zeros(10)
            dist_ref_img = np.zeros(10)      
             
            dist_1st_img[0] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,1])
            dist_1st_img[1] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,5])
            dist_1st_img[2] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,9])
            dist_1st_img[3] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,13])
            dist_1st_img[4] = np.linalg.norm(hands_1st_img[0,0]-hands_1st_img[0,17])
            dist_1st_img[5] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,1])
            dist_1st_img[6] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,5])
            dist_1st_img[7] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,9])
            dist_1st_img[8] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,13])
            dist_1st_img[9] = np.linalg.norm(hands_1st_img[1,0]-hands_1st_img[1,17])

            dist_ref_img[0] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,1])
            dist_ref_img[1] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,5])
            dist_ref_img[2] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,9])
            dist_ref_img[3] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,13])
            dist_ref_img[4] = np.linalg.norm(hands_ref_img[0,0]-hands_ref_img[0,17])
            dist_ref_img[5] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,1])
            dist_ref_img[6] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,5])
            dist_ref_img[7] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,9])
            dist_ref_img[8] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,13])
            dist_ref_img[9] = np.linalg.norm(hands_ref_img[1,0]-hands_ref_img[1,17])

            ratio = 0   
            count = 0
            for i in range (10): 
                if dist_1st_img[i] != 0:
                    ratio = ratio + dist_ref_img[i]/dist_1st_img[i]
                    count = count + 1
            if count!=0:
                align_args["scale_hand"] = (ratio/count+align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/3
            else:
                align_args["scale_hand"] = (align_args["scale_arm_upper"]+align_args["scale_arm_lower"])/2

            # body 
            dist_1st_img = np.linalg.norm(body_1st_img[1] - (body_1st_img[8] + body_1st_img[11])/2 )
            dist_ref_img = np.linalg.norm(body_ref_img[1] - (body_ref_img[8] + body_ref_img[11])/2 )
            align_args["scale_body_len"]=dist_ref_img / dist_1st_img

            dist_1st_img = np.linalg.norm(body_1st_img[8]-body_1st_img[9])
            dist_ref_img = np.linalg.norm(body_ref_img[8]-body_ref_img[9])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[11]-body_1st_img[12])
            dist_ref_img = np.linalg.norm(body_ref_img[11]-body_ref_img[12])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_leg_upper"] = (s1+s2)/2

            dist_1st_img = np.linalg.norm(body_1st_img[9]-body_1st_img[10])
            dist_ref_img = np.linalg.norm(body_ref_img[9]-body_ref_img[10])
            s1 = dist_ref_img / dist_1st_img
            dist_1st_img = np.linalg.norm(body_1st_img[12]-body_1st_img[13])
            dist_ref_img = np.linalg.norm(body_ref_img[12]-body_ref_img[13])
            s2 = dist_ref_img / dist_1st_img
            align_args["scale_leg_lower"] = (s1+s2)/2

            ####################
            ####################
            # need adjust nan
            for k,v in align_args.items():
                if np.isnan(v):
                    align_args[k]=1

            # centre offset (the offset of key point 1)
            offset = body_ref_img[1] - body_1st_img[1]
        
    
        # pose align
        pose_img, pose_ori = detector(img, args.detect_resolution, args.image_resolution, output_type='cv2', return_pose_dict=True)
        video_pose_buffer.append(pose_img)
        pose_align = align_img(img, pose_ori, align_args, args.detect_resolution, args.image_resolution)
        

        # add centre offset
        pose = pose_align
        pose['bodies']['candidate'] = pose['bodies']['candidate'] + offset
        pose['hands'] = pose['hands'] + offset
        pose['faces'] = pose['faces'] + offset


        # h不变，w从绝对坐标缩放回0-1 注意这里要回到ref的坐标系
        pose['bodies']['candidate'][:, 0] = pose['bodies']['candidate'][:, 0] / ref_ratio
        pose['hands'][:, :, 0] = pose['hands'][:, :, 0] / ref_ratio
        pose['faces'][:, :, 0] = pose['faces'][:, :, 0] / ref_ratio
        pose_list.append(pose)

    # stack
    body_list  = [pose['bodies']['candidate'][:18] for pose in pose_list]
    body_list_subset = [pose['bodies']['subset'][:1] for pose in pose_list]
    hands_list = [pose['hands'][:2] for pose in pose_list]
    faces_list = [pose['faces'][:1] for pose in pose_list]
   
    body_seq         = np.stack(body_list       , axis=0)
    body_seq_subset  = np.stack(body_list_subset, axis=0)
    hands_seq        = np.stack(hands_list      , axis=0)
    faces_seq        = np.stack(faces_list      , axis=0)

    # smooth
    if args.smooth_method=='savgol':
        winlen=args.winlen
        polyorder=args.polyorder
        body_seq  = signal.savgol_filter(body_seq,  window_length=winlen, polyorder=polyorder, mode='nearest', axis=0)
        hands_seq = signal.savgol_filter(hands_seq, window_length=winlen, polyorder=polyorder, mode='nearest', axis=0)
        faces_seq = signal.savgol_filter(faces_seq, window_length=winlen, polyorder=polyorder, mode='nearest', axis=0)


    # concatenate and paint results
    H = 512 # paint height
    W1 = int((H/ref_H * ref_W)//2 *2)
    W2 = int((H/height * width)//2 *2)
    writer = Writer(args, None, H, 3*W1+2*W2, outfn, fps)
    writer_pose_only = Writer(args, None, H, W1, args.outfn_align_pose_video, fps)
    for i in range(len(body_seq)):
        pose_t={}
        pose_t["bodies"]={}
        pose_t["bodies"]["candidate"]=body_seq[i]
        pose_t["bodies"]["subset"]=body_seq_subset[i]
        pose_t["hands"]=hands_seq[i]
        pose_t["faces"]=faces_seq[i]

        ref_img = cv2.cvtColor(refer_img, cv2.COLOR_RGB2BGR)
        ref_img = cv2.resize(ref_img, (W1, H))
        ref_pose= cv2.resize(output_refer, (W1, H))
        
        output_transformed = pose2map(
                pose_t, 
                H_in, W_in, 
                args.detect_resolution, 
                args.image_resolution,
                include_body=True,
                include_face=False,
                include_hand=True,
                include_eye=False
                )
        output_transformed = cv2.resize(output_transformed, (W1, H))
        
        video_frame = cv2.resize(video_frame_buffer[i], (W2, H))
        video_pose  = cv2.resize(video_pose_buffer[i], (W2, H))

        res = np.concatenate([ref_img, ref_pose, output_transformed, video_frame, video_pose], axis=1)
        writer.write_frame(res)
        writer_pose_only.write_frame(output_transformed)

    writer.close()
    writer_pose_only.close()
    reader.close()
    print(f"pose_list len: {len(pose_list)}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input video, image or folder')
    parser.add_argument('--output', type=str, default='results', help='Output folder')
    parser.add_argument('--inputscale', type=float, default=1, help='The input scale of the video, reader')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored video')
    parser.add_argument('--fps', type=float, default=None, help='FPS of the output video')
    parser.add_argument('--ffmpeg_bin', type=str, default='ffmpeg', help='The path to ffmpeg')
    parser.add_argument('--vcodec', type=str, default='libx264', help='vcodec')
    parser.add_argument('--input_pix_fmt', type=str, default='rgb24', help='ffmpeg input pix_fmt')
    parser.add_argument('--crf', type=str, default='18', help='crf')
    parser.add_argument('--detect_resolution', type=int, default=512, help='detect_resolution')
    parser.add_argument('--image_resolution', type=int, default=720, help='image_resolution')
    parser.add_argument('--dtype', type=str, default='fp16', help='dtype')
    parser.add_argument('--smooth_method', type=str, default=None, help='Suffix of the restored video')
    parser.add_argument('--winlen', type=int, default=11, help='window length')
    parser.add_argument('--polyorder', type=int, default=1, help='polyorder')

    parser.add_argument('--max_frame', type=int, default=100, help='maximum frame number of the video to align')
    parser.add_argument('--vidfn', type=str, default="/workspace/user_code/projects/PoseAlign/videos/full/chen.mp4", required=True, help='Input video path')
    parser.add_argument('--imgfn_refer', type=str, default="/workspace/user_code/projects/PoseAlign/images/full/gibbon.jpg", required=True, help='refer image path')
    parser.add_argument('--outfn_ref_img_pose', type=str, default="./data/pose_align_results/ref_img_pose.jpg", required=True, help='output path of the pose of the refer img')
    parser.add_argument('--outfn_align_pose_video', type=str, default=None, required=True, help='output path of the aligned video of the refer img')
    parser.add_argument('--outfn', type=str, default=None, required=True, help='Output path of the alignment visualization')
    args = parser.parse_args()

    # args.imgfn_refer="/workspace/user_code/projects/PoseAlign/images/full/gibbon.jpg"
    # foldername = args.imgfn_refer.split('/')[-2]
    # ref_filename = args.imgfn_refer.split('/')[-1]
    # ref_name, suffix = ref_filename.split('.')
    # args.outfn_ref_img_pose='/workspace/user_code/projects/PoseAlign/results/' + foldername + '/' + 'ref_' + ref_name + "_dwpose" + '.' + suffix

    # args.vidfn='/workspace/user_code/projects/PoseAlign/videos/full/chen.mp4'
    # foldername = args.vidfn.split('/')[-2]
    # pose_filename = args.vidfn.split('/')[-1]
    # pose_name, suffix = pose_filename.split('.')
    # args.outfn=f'/workspace/user_code/projects/PoseAlign/results/' + foldername + '/' + 'ref_' + ref_name + "_video_" + pose_name +'.' + suffix

    run_align_video_with_filterPose_translate_smooth(args)


    
if __name__ == '__main__':
    main()
