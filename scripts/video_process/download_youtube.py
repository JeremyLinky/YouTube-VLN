import socks
import socket
from urllib import request
from pytube import YouTube, exceptions

from os import listdir, mkdir, rename, system
import numpy as np
from tqdm import tqdm
import os
import urllib
import time
import re
import argtyped
from pathlib import Path
import yt_dlp

# YouTube is not available in some regions and requires a proxy
proxy_ip = "cpu0"  # fill in your proxy ip
proxy_port =  int(1080) # fill in your proxy port

class Arguments(argtyped.Arguments, underscore=True):
    video_path: Path = "data/YouTube-VLN/videos.npy"
    # Use '--video_path' to specify a path

class NoStreamFound(Exception):
    pass

def files(folder,regex = '.*',reject=None):
    fil = os.popen(f'ls {folder}').read().split()
    fils = [ e for e in fil if re.search(regex,e)]
    if reject is not None:
        fils = [ e for e in fil if not re.search(reject,e)]
    return fils
args = Arguments()
urls = np.load(args.video_path)
url_prefix = 'https://www.youtube.com/watch?v='

print('Num videos:', urls.shape[0])
failures = []

try:
    mkdir('videos')
    print('creating videos subdirectory')
except:
    print('videos subdirectory already exists')

num_failed = 0
videos_path = 'data/YouTube-VLN/videos'
videos = files(videos_path)
sleep_time = 10

# remove extension to get ids
completed = [v[:-4]for v in videos]
remaining = set(urls) - set(completed)
print("Num Remaining: ",len(remaining))
ydl_opts = {
    'proxy': f'socks5://{proxy_ip}:{proxy_port}',
    # 'format': 'bestvideo+bestaudio',
    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
    'outtmpl': os.path.join(videos_path, '%(id)s.%(ext)s'),
}
for vid_id in tqdm(remaining):
    while True:
        print('starting', vid_id)

        url = url_prefix + vid_id

        try:
            print('url: ', url)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ytb_info=ydl.extract_info(url, download=False)
                ydl.download([url])
            print('finished!')
        except (yt_dlp.utils.ExtractorError,yt_dlp.utils.DownloadError) as n_e:
            if 'SME' in str(n_e):
                print("Failed on ", vid_id)
                failures.append(vid_id)
                break
        except (exceptions.VideoUnavailable, exceptions.RegexMatchError,
                NoStreamFound, urllib.error.HTTPError) as e:
            print(e)
            # too many requests error, exponential backoff 
            if type(e) == urllib.error.HTTPError and e.code == 429:
                print("backoff")
                time.sleep(sleep_time)
                sleep_time *= 2
                continue
            print("Failed on ", vid_id)
            failures.append(vid_id)
        sleep_time = 10
        break
print("failures:", failures)