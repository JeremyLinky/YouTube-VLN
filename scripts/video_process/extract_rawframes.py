from os import listdir, mkdir
import subprocess
import re
import multiprocessing.dummy as mp 

def extract_frames(videopath, dest, fps=1):

    try:
        mkdir(dest)
        print('creating ' + dest + ' subdirectory')
    except:
        print(dest + ' subdirectory already exists')

    output = subprocess.call([
        'ffmpeg', '-i', videopath, '-vf', 'fps=' + str(fps), dest + '/%04d.jpg'
    ]) # The frame rate is 0.5 sampled video frames
    if output:
        print('Failed to extract frames')

def extract_all_frames():
    try:
        mkdir('data/YouTube-VLN/raw_frames')
        print('creating frames subdirectory')
    except:
        print('frames subdirectory already exists')
    videos = listdir('data/YouTube-VLN/videos')

    def eaf(vid):
        vid_id = re.match('(.*).mp4', vid)[1]
        subdir = 'data/YouTube-VLN/raw_frames/' + vid_id
        try:
            mkdir(subdir)
            extract_frames('data/YouTube-VLN/videos/' + vid, subdir, fps=.5)
        except FileExistsError:
            print(f'skipping {vid}')
    
    vids = [vid for vid in videos]
    p=mp.Pool()
    p.map(eaf,vids)
    p.close()
    p.join()

if __name__ == "__main__":
    extract_all_frames()


