# DeepMIR HW2: Source Separation

## Overview
### Task 1: Train a source separation model
* Train a source separation model from scratch to separate vocal/non-vocal
stems.
* Model: You can use any source separation model, but your model parameter
should be lower than 10M.
* Open-Unmix model is recommended. (Model parameter: 9M) [[github](https://github.com/sigsep/open-unmix-pytorch)] [[paper](https://joss.theoj.org/papers/10.21105/joss.01667)]

    #### Requirement
* Need to report the model you choose and details of your implementation
(model architecture, model parameters, data augmentation, …).
* Need to report the testing result with SDR matrix for [25, 50, 150] epoch.
* Need to generate some separated audio as listening samples.

### Task 2: Griffin & Lim for phase estimation
* In Open-Unmix, we copy the phase from the mixture.
* Now, estimate phase using Griffin & Lim algorithm with your best epoch.
* Library: librosa.griffinlim. For more details, see lecture 5. slides.
* Need to report the testing result with SDR matrix on your best epoch in task 1.
* Need to generate some separated audio as listening samples.

### Optional Task
* [ ] Compare different models 
* [x] Data augmentation
* [ ] Estimate phase with Mel-Vocoder
* [x] Model structure modification

### Dataset: MUSDB18
* 150 full lengths music tracks (10h duration)
    * Training: 100 tracks (Include 14 tracks for validation in Open-Unmix)
    * Testing: 50 tracks
* All signals are stereophonic and encoded at 44.1kHz
* For more detail please refer to the source

### Evaluation
* Matrix: SDR (Source-to-Distortion Ratio)
* SDR is usually considered to be an overall measure of how good a source
sounds
* Compute in the time-domain


## Getting Started 
```bash
# Clone the repo:
git clone https://github.com/PANpinchi/DeepMIR_HW2_PANpinchi.git
# Move into the root directory:
cd DeepMIR_HW2_PANpinchi
```
## Environment Settings
```bash
# Create a virtual conda environment:
conda create -n deepmir_hw2 python=3.7

# Activate the environment:
conda activate deepmir_hw2

# Install PyTorch, TorchVision, and Torchaudio with CUDA 10.2
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

# Install additional dependencies from environment-gpu-linux-cuda10.yml:
conda env create -f scripts/environment-gpu-linux-cuda10.yml

# Install additional dependencies from requirements.txt:
pip install -r requirements.txt
```
## Download the Required Data
#### 1. Pre-trained Models
Run the commands below to download the pre-trained model.
```bash
cd open-unmix-pytorch/scripts

# The pre-trained Open-Unmix model for 25 epochs.
gdown --folder https://drive.google.com/drive/folders/1HbK8A9yaR07K90bXoU7wpoZT9g0uQBiU?usp=drive_link

# The pre-trained Open-Unmix model for 50 epochs.
gdown --folder https://drive.google.com/drive/folders/10nwH9NMCEdIvsXi3VOANzX9leaoCfvcN?usp=drive_link

# The pre-trained Open-Unmix model for 150 epochs.
gdown --folder https://drive.google.com/drive/folders/1D6iIV3QLEcusbSVfxwmbyEmtQj3gqyo0?usp=drive_link

# The pre-trained Open-Unmix model for 571 epochs.
gdown --folder https://drive.google.com/drive/folders/14pDvqW8xHE5M7d8V4PqLsPfldif6Z4BG?usp=drive_link

-------------------------------------------------------------------------

# The pre-trained model with data augmentation for 160 epochs.
gdown --folder https://drive.google.com/drive/folders/1YaAWaCFrqZ7_j7LrERi0KgUEXJPiJVsb?usp=drive_link

# The pre-trained model with data augmentation for 150 epochs. 
gdown --folder https://drive.google.com/drive/folders/1bKYGwniVeqHDmBzdIF_ZrApUMq7Z886F?usp=drive_link

-------------------------------------------------------------------------

# The pre-trained model with the LSTM layers removed for 160 epochs. 
gdown --folder https://drive.google.com/drive/folders/1woBsHWPIkUQL9uMru2TXLnntTDpoKCq9?usp=drive_link

# The pre-trained model with the LSTM layers removed for 150 epochs. 
gdown --folder https://drive.google.com/drive/folders/1vgcG1pnT1KkCx03us6Oqv4VslaMqUmaG?usp=drive_link

cd ../..
```
Note: `sdr_results.json`, `separator.json`, `vocals.chkpnt`, `vocals.json`, and `vocals.pth` files should be placed in the each folder.

#### 2. Datasets
The used training and testing sets can be downloaded from [MUSDB18](https://zenodo.org/records/1117372) and [MUSDB18-HQ](https://zenodo.org/records/3338373) Datasets.

You need to run the commands below to unzip the MUSDB18 and MUSDB18-HQ contents and put them in /datasets.
```bash
unzip musdb18.zip
mkdir MUSDB18
mv test train MUSDB18/
```
```bash
unzip musdb18hq.zip
mkdir MUSDB18-HQ
mv test train MUSDB18-HQ/
```

#### The data directory structure should follow the below hierarchy.
```
${ROOT}
|-- datasets/
|    |-- MUSDB18/
|    |    |-- test/
|    |    |    |-- Al James - Schoolboy Facination.stem.mp4
|    |    |    |-- AM Contra - Heart Peripheral.stem.mp4
|    |    |    |-- ...
|    |    |    |-- Zeno - Signs.stem.mp4
|    |    |-- train/
|    |    |    |-- A Classic Education - NightOwl.stem.mp4
|    |    |    |-- Actions - Devil's Words.stem.mp4
|    |    |    |-- ...
|    |    |    |-- Young Griffo - Pennies.stem.mp4
|    |-- MUSDB18-HQ/
|    |    |-- test/
|    |    |    |-- Al James - Schoolboy Facination/
|    |    |    |    |-- bass.wav
|    |    |    |    |-- drums.wav
|    |    |    |    |-- mixture.wav
|    |    |    |    |-- other.wav
|    |    |    |    |-- vocals.wav
|    |    |    |-- AM Contra - Heart Peripheral/
|    |    |    |    |-- bass.wav
|    |    |    |    |-- drums.wav
|    |    |    |    |-- mixture.wav
|    |    |    |    |-- other.wav
|    |    |    |    |-- vocals.wav
|    |    |    |-- .../
|    |    |    |-- Zeno - Signs/
|    |    |    |    |-- bass.wav
|    |    |    |    |-- drums.wav
|    |    |    |    |-- mixture.wav
|    |    |    |    |-- other.wav
|    |    |    |    |-- vocals.wav
|    |    |-- train/
|    |    |    |-- A Classic Education - NightOwl/
|    |    |    |    |-- bass.wav
|    |    |    |    |-- drums.wav
|    |    |    |    |-- mixture.wav
|    |    |    |    |-- other.wav
|    |    |    |    |-- vocals.wav
|    |    |    |-- Actions - Devil's Words/
|    |    |    |    |-- bass.wav
|    |    |    |    |-- drums.wav
|    |    |    |    |-- mixture.wav
|    |    |    |    |-- other.wav
|    |    |    |    |-- vocals.wav
|    |    |    |-- .../
|    |    |    |-- Young Griffo - Pennies/
|    |    |    |    |-- bass.wav
|    |    |    |    |-- drums.wav
|    |    |    |    |-- mixture.wav
|    |    |    |    |-- other.wav
|    |    |    |    |-- vocals.wav
|-- open-unmix-pytorch/
```

## 【Task 1: Train a source separation model】
* Train a source separation model from scratch to separate vocal/non-vocal
stems.
* Model: You can use any source separation model, but your model parameter
should be lower than 10M.
* Open-Unmix model is recommended. (Model parameter: 9M)

#### Training
```bash
cd open-unmix-pytorch/scripts

# Train an Open-Unmix model.
python train.py --root ../../datasets/MUSDB18-HQ --target vocals --nb-workers 24 --is_wav

# Train an Open-Unmix model with the LSTM layers removed.
python train.py --remove_lstm --root ../../datasets/MUSDB18-HQ --output remove_lstm --target vocals --nb-workers 24 --is_wav
```

#### Evaluation
```bash
cd open-unmix-pytorch/

# open-unmix for 25 epochs
python my_openunmix/evaluate.py --root ../datasets/MUSDB18-HQ --target vocals --residual accompaniment --is-wav --model scripts/open-unmix_25 --outdir out_task1_25 --evaldir out_task1_25_eval

# open-unmix for 50 epochs
python my_openunmix/evaluate.py --root ../datasets/MUSDB18-HQ --target vocals --residual accompaniment --is-wav --model scripts/open-unmix_50 --outdir out_task1_50 --evaldir out_task1_50_eval

# open-unmix for 150 epochs
python my_openunmix/evaluate.py --root ../datasets/MUSDB18-HQ --target vocals --residual accompaniment --is-wav --model scripts/open-unmix_150 --outdir out_task1_150 --evaldir out_task1_150_eval

# open-unmix for 571 epochs
python my_openunmix/evaluate.py --root ../datasets/MUSDB18-HQ --target vocals --residual accompaniment --is-wav --model scripts/open-unmix --outdir out_task1 --evaldir out_task1_eval


# data_augmentation for 150 epochs
python my_openunmix/evaluate.py --root ../datasets/MUSDB18-HQ --target vocals --residual accompaniment --is-wav --model scripts/data_augmentation_150 --outdir out_data_augmentation_150 --evaldir out_data_augmentation_150_eval


# remove_lstm for 150 epochs
python my_openunmix/evaluate.py --remove_lstm --root ../datasets/MUSDB18-HQ --target vocals --residual accompaniment --is-wav --model scripts/remove_lstm --outdir out_remove_lstm_150 --evaldir out_remove_lstm_eval
```

## 【Task 2: Griffin & Lim for phase estimation】
* In Open-Unmix, we copy the phase from the mixture.
* Now, estimate phase using Griffin & Lim algorithm with your best epoch.
* Library: librosa.griffinlim. 

#### Evaluation
```bash
# Griffin & Lim for phase estimation for 150 epochs
python my_openunmix/evaluate.py --root ../datasets/MUSDB18-HQ --target vocals --residual accompaniment --is-wav --use_griffinlim --model scripts/open-unmix_150 --outdir out_task2 --evaldir out_task2_eval
```

