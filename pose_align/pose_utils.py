import torch
import cv2
import numpy as np
from pprint import pprint
import ffmpeg
import argparse
import time
import traceback
import scipy.signal as signal



def get_video_meta_info(video_path):
    print(video_path)
    ret = {}
    probe = ffmpeg.probe(video_path)
    video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
    has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    ret['width'] = video_streams[0]['width']
    ret['height'] = video_streams[0]['height']
    ret['fps'] = eval(video_streams[0]['avg_frame_rate'])
    ret['audio'] = ffmpeg.input(video_path).audio if has_audio else None
    if 'nb_frames' in video_streams[0].keys():
        ret['nb_frames'] = int(video_streams[0]['nb_frames'])
    else:
        cap = cv2.VideoCapture(video_path)
        ret['nb_frames']=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return ret



# Calculate the resolution 
def size_calculate(h, w, resolution):
    
    H = float(h)
    W = float(w)

    # resize the short edge to the resolution
    k = float(resolution) / min(H, W) # short edge
    H *= k
    W *= k

    # resize to the nearest integer multiple of 64
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    return H, W



def warpAffine_kps(kps, M):
    a = M[:,:2]
    t = M[:,2]
    kps = np.dot(kps, a.T) + t
    return kps



class Reader:

    def __init__(self, args, video_path):
        self.args = args
        self.audio = None
        self.input_fps = None
        self.input_type = 'video'


        meta = get_video_meta_info(video_path)
        self.width = meta['width']
        self.height = meta['height']
        self.input_fps = meta['fps']
        self.audio = meta['audio']
        self.nb_frames = meta['nb_frames']
        
        self.height = int(self.height) // 2 * 2
        self.width  = int(self.width) // 2 * 2
        if args.fps is not None:
            self.stream_reader = (
                # ffmpeg.input(video_path, ss=0,t=2).filter('fps',fps=args.fps).filter('scale', self.width , self.height).output('pipe:', format='rawvideo', pix_fmt='rgb24',
                ffmpeg.input(video_path).filter('fps',fps=args.fps).filter('scale', self.width , self.height).output('pipe:', format='rawvideo', pix_fmt='rgb24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_reader = (
                # ffmpeg.input(video_path, ss=0,t=2).filter('scale', self.width , self.height).output('pipe:', format='rawvideo', pix_fmt='rgb24',
                ffmpeg.input(video_path).filter('scale', self.width , self.height).output('pipe:', format='rawvideo', pix_fmt='rgb24',
                                                loglevel='error').run_async(
                                                    pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))


    def get_resolution(self):
        return self.height, self.width

    def get_fps(self):
        if self.args.fps is not None:
            return self.args.fps
        elif self.input_fps is not None:
            return self.input_fps
        return 24

    def get_audio(self):
        return self.audio

    def __len__(self):
        return self.nb_frames

    def get_frame_from_stream(self):
        img_bytes = self.stream_reader.stdout.read(self.width * self.height * 3)  # 3 bytes for one pixel
        if not img_bytes:
            return None
        img = np.frombuffer(img_bytes, np.uint8).reshape([self.height, self.width, 3])
        return img

    def get_frame(self):
        if self.input_type.startswith('video'):
            return self.get_frame_from_stream()

    def close(self):
        if self.input_type.startswith('video'):
            self.stream_reader.stdin.close()
            # self.stream_reader.wait()



class Writer:

    def __init__(self, args, audio, height, width, video_save_path, fps):
        out_width, out_height = int(width), int(height)
        if out_height > 2160:
            print('You are generating video that is larger than 4K, which will be very slow due to IO speed.',
                  'We highly recommend to decrease the outscale(aka, -s).')

        if audio is not None:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt=args.input_pix_fmt, s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 video_save_path,
                                 pix_fmt='yuv420p',
                                 vcodec=args.vcodec,
                                 crf=args.crf,
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                 # acodec='copy').run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt=args.input_pix_fmt, s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 video_save_path, 
                                 pix_fmt='yuv420p', 
                                 vcodec=args.vcodec,
                                 crf=args.crf,
                                 loglevel='error').overwrite_output().run_async(
                                 # loglevel='error').run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd=args.ffmpeg_bin))

    def write_frame(self, frame):
        frame = frame.astype(np.uint8).tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()