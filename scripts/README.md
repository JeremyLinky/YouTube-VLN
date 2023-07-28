# :satisfied: Instrutions for building up YouTube-VLN dataset
Install and activate the conda environment for YouTube-VLN dataset
```bash
conda env create -f scripts/env.yaml
conda activate YouTube-VLN
```
or install the environment by
```bash
pip install -r requirements.txt
```
Some packages may be missed you need to refer to the *requirements.txt* to install manually. 

# :tv: Firstly, we need to get the YouTube videos and frames

## 1. Prepare YouTube Videos
This step needs to download [videos.npy](https://drive.google.com/file/d/1yJSOnPsnSH6ndp92_5CCawEIHT6RoMhy/view?usp=drive_link) and put it into *data/YouTube-VLN*.
```bash
python -m scripts.video_process.download_youtube
# Download the videos according to the video ids
```

## 2. Extract frames from videos
```bash
python -m scripts.video_process.extract_rawframes
# Extract the raw frames of videos
```


# :mag: Secondly, we need to generate the features of frames

## 3. Extract bottom-up top-down features
This step needs the installation of [bottom-up top-down attention](https://github.com/peteanderson80/bottom-up-attention.git).

We split the features into 11 parts. In order to speed up the feature extraction, we use two servers to extract features respectively. One uses 8 GPUs to extract and generate the first 8 parts, and the other uses 3 GPUs to generate the last 3 parts. The number of gpus equals the number of workers, and *--start* indicates which part to start extracting features from.
```bash
python -m scripts.video_process.precompute_youtube_img_features_with_butd --gpu "0,1,2,3,4,5,6,7" --num-workers 8 --start 0 --num-splits 11

python -m scripts.video_process.precompute_youtube_img_features_with_butd --gpu "4,5,6" --num-workers 3 --start 8 --num-splits 8
```

## 4. Build an LMDB file
```bash
python -m scripts.video_process.convert_to_lmdb --output data/YouTube-VLN/youtube_img_features/img_features --tsv-folder data/YouTube-VLN/youtube_img_features
```


# :surfer: Thirdly, we need to filter the frames and generate the actions between key frames

## 5. Filter out the outdoor frames and frames with people
This step needs to download the file [io_places.txt](https://drive.google.com/file/d/1xdx89y7LBn8G1KUZP-u1ifg4BCPKFHGy/view?usp=drive_link) and put it into *data/YouTube-VLN/model4youtube*.
```bash
python -m scripts.video_process.filter_outdoor_resnet_place365
# Filter out the outdoor frames using wideresnet trained on places365
python -m scripts.video_process.filter_outdoor_maskrcnn_coco
# Filter out the frames with person using maskrcnn trained on coco
```

## 6. Obtain the image captions of all frames
```bash
python -m scripts.video_process.precomute_CLIP_features
# Precompute the CLIP features for each video
python -m scripts.video_process.generate_CLIP_captions
# Generate the captions for each video
```

## 7. Obtain the inverse action between two key frames
```bash
python -m scripts.inverse_action.main
```

## 8. Get the files for generating instructions
```bash
python -m scripts.video_process.genearate_Profiles
```

# :sparkles: Now you can prepare the dataset for pre-training!

## 9. Build dataset for train and test
```bash
python -m scripts.build_dataset.build_dataset

# json_file
python -m scripts.build_dataset.preprocess_dataset --csv  data/YouTube-VLN/Extra/ytb_test.tsv --name ytb_test
python -m scripts.build_dataset.preprocess_dataset --csv  data/YouTube-VLN/Extra/ytb_train.tsv --name ytb_train


# for val
python -m scripts.build_dataset.build_testset --captions data/YouTube-VLN/ytb/ytb_test.json --output data/YouTube-VLN/ytb/testset.json
```

## 10. Image merging
Create image merging datasets (if need)
```bash
# There are three method to merge image: max, least, all, adjacent, continue (default: max)
# max: Merging max_photo_per_merging frames before and after the current frame
# least: Exactly according to max_photo_per_merging
# all: Merge all frames of the same room type
# adjacent: Only continuous frames of the same type are considered (continuous means the difference between the two frames is 1)
# continue: Default to the same room until a new room type is encountered (only valid frames are considered)
python -m scripts.build_dataset.merge_photos --source data/YouTube-VLN/ytb/ytb_test.json --output data/YouTube-VLN/ytb/merge+ytb_test.json --merge-method max

python -m scripts.build_dataset.merge_photos --source data/YouTube-VLN/ytb/ytb_train.json --output data/YouTube-VLN/ytb/merge+ytb_train.json --merge-method max

# Generative tesetset 
python -m scripts.build_dataset.build_testset --captions data/YouTube-VLN/ytb/merge+ytb_test.json --output data/YouTube-VLN/ytb/merge+testset.json
```
