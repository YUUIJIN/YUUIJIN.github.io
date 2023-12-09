# AIX.Deeplearning
# Title: 환자의 호흡 소리를 통한 호흡기 질환 분류
# Members
      유의진 해양융합공학과 dbdmlwls1224@naver.com
      윤정상 해양융합공학과 jeongsang9907@gmail.com
# Index
## Ⅰ. Proposal
- Motivation
- Goal
## Ⅱ. Datasets
## Ⅲ. Methodology
## Ⅳ. Evaluation & Analysis
## Ⅴ. Related Work
## Ⅵ. Conclusion: Discussion
# Ⅰ. Proposal (option A)
## Motivation
호흡기 질환에 있어 청진은 필수적인 검사입니다. 이는 비침습적이고 실시간이기 때문에 호흡기 질환의 진단과 진찰에 있어 유리합니다. 그러나 호흡음에 대한 정확한 해석은 임상의의 상당한 전문성을 요구하고 인턴이나 레지던트 등 실습생이 호흡음을 오인하는 경우도 발생합니다. 이러한 한계를 극복하기 위해 임상환경에서의 CNN  (Convolutional Neural Network)을 이용한 호흡음 자동 분류 알고리즘 개발이 진행되고 있습니다. 딥러닝 기반 호흡음 자동 분류는 임상의의 청진 부정확성을 보완할 수 있을 것이며, 호흡기 질환의 신속한 진단과 적절한 치료에 도움이 될 것이라고 생각하였습니다. 저희는 딥러닝 기반 호흡음 자동 분류를 통해 다양한 호흡기 질환 진단을 목표로 CNN 기반 호흡음 자동 분류 논문을 참고하여 모델을 구현하려고 하였습니다.
## Goal
저희의 최종적인 목표는 딥러닝 분류 모델의 구조적 이해와 구현입니다. 저희가 음성 분류에 사용할 구조는 CNN(Convolution Neural Network)입니다. wav파일(음성 파일)이 CNN을 통해 학습되는 정확한 알고리즘의 이해에 초점을 맞추려 합니다. 또한 이러한 음성 처리는 신호 처리 과정이 포함되기에 Spectrogram이나 (예를 들어 MFCC(Mel-Frequency Cepstal Coeffcient)) FFT, Pwelch 등 기본적인 신호 처리 개념의 이해를 목적으로 합니다. 모델 구현을 위해 호흡음 분류에 사용되는 데이터셋인 디지털 청진기와 의학적 녹음 기술을 사용한 wav 파일, 환자 진단 list를 사용합니다. 환자 진단 list에는 각 환자별 천식, 폐렴, 기관지염과 같은 호흡기 질환들이 분류되어 있습니다. 데이터셋의 일부를 활용해 모델을 train하며 데이터셋의 나머지를 활용해 모델을 test할 계획입니다.
# Ⅱ. Datasets
## 데이터셋은 코딩이 아닌 다운받은 원본 파일을 확인하는 식으로 보여드리겠습니다.

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/149877341/9135c1f2-4525-480c-8eea-94e8a1b4069b" alt="Image" width="60%" height="60%">
</div>

### 1. Demographic Info
이는 text 파일로 된 환자 정보입니다. <br/>  환자번호 / 나이 / 성별 / Adult BMI(kg/m^2) / Child Weight(kg) / Child Height(cm)

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/149877341/b4f94808-a1bf-42f3-a510-e3ab10cda052" alt="Image" width="20%" height="20%">
</div>

2번 라인에서의 예시 <br/>
[ 101번 환자 / 3살 / 여자 / 결측값 / 19kg / 99cm ]

### 2. Respiratory Sound Database

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/149877341/ac44c8ce-d837-40eb-ac07-e9c8bfbb31a4" alt="Image" width="50%" height="50%">
</div>

#### 1) Audio & Text Files

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/149877341/25033a27-8df2-47e5-a26f-d6b8d757a5b0" alt="Image" width="50%" height="50%">
</div>

우선, txt 파일과 wav 파일이 하나씩 쌍으로 있습니다. 총 920쌍으로 1840개의 데이터가 존재합니다. <br/> wav 파일은 10~90초 길이 사이의 랜덤하게 분포한 파일입니다. <br/>
이 시간대에는 총 6898개의 호흡 주기가 있고, 분류는 다음과 같습니다. <br/> 
- 1864개 = Crackle <br/>
- 886개 = Wheeze<br/>
- 506개 = Crackle & Wheeze <br/>
- 3642개 =  No Crackle & Wheeze <br/>
- Crackle = 수포음, Wheeze = 천명(쇳소리) 

#### A. Audio Files (wav.)

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/149877341/b2fbad54-7467-44f9-b92c-7b6de3ed9a5c" alt="Image" width="70%" height="70%">
</div>

[ 환자번호 _ recording index _ chest location _ 녹음 모드 _ 녹음 장비 ] <br/>
위와 같이 파일명이 구성되어 있습니다.
Chest location은 청진기를 가져다 대는 부분을 의미합니다.

위 그림을 예시로 들자면, <br/>
[ 101번 환자 _ 1b1 _ Anterior left _ 순차,단일 채널 _ Meditron ]

#### B. Text Files (txt.)

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/149877341/fa48bc27-6c13-4a25-a5bf-542c8f546871" alt="Image" width="20%" height="20%">
</div>

[ 호흡주기 시작 / 호흡주기 종료 / Crackles 유무 / Wheeze 유무 ]<br/>
이때, Crackle과 Wheeze의 유무는 0과 1로 나타내고 있습니다.


### 3. Patient Diagnosis

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/149877341/3c901371-4f2c-45e9-bb38-0072485e0263" alt="Image" width="20%" height="20%">
</div>

위 그림은 patient_diagnosis.xlsx로 환자번호별 진단명이며, 일부분을 표현하고 있습니다.

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/149877341/b1487ed5-bb73-4406-9414-3000ca628a2a" alt="Image" width="20%" height="20%">
</div>

진단명은 총 7가지가 있습니다. 

# Ⅲ. Methodology
## 1. Overview
- 분석에 기본적으로 필요한 파이썬 라이브러리를 가져옵니다.
```py
import wave
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
%matplotlib inline
import wave
import math
import scipy.io.wavfile as wf
``` 
- 데이터셋이 들어있는 파일을 확인합니다.
```py
 os.listdir('Respiratory_Sound_Database')
```
-  df_no_diagnosis: 환자번호 + 질병 외의 정보
-  diagnosis: 환자 번호 + 질병 정보
-  df: 위 두 변수를 환자 번호 기준으로 합침
  ```py
  df_no_diagnosis = pd.read_csv('Respiratory_Sound_Database/demographic_info.txt', names = ['Patient number', 'Age', 'Sex', 'Adult BMI(kg/m^2)', 'Child Weight(kg)','Child Height(cm)'], delimiter = ' ')
  diagnosis = pd.read_csv('Respiratory_Sound_Database/patient_diagnosis.csv', names = ['Patient number', 'Diagnosis'])
  df = df_no_diagnosis.join(diagnosis.set_index('Patient number'), on = 'Patient number', how = 'left')
```
![image](https://github.com/YUUIJIN/YUUIJIN.github.io/assets/134063047/f1f0911f-6b8f-4e11-94d3-305f98f0c0a2)

- df에서 각 질병 빈도수를 counting 합니다.
  ```py
  df['Diagnosis'].value_counts()
  ```
- txt 파일로 되어있는 audio 파일을 전처리합니다.
- filenames: 확장자 제거 후 list에 저장
- Extract_Annotation_Data: 파일 이름과 파일 주소를 가져와 데이터를 추출하는 함수
  ```py
  root = 'Respiratory_sound_Database/audio_and_txt_files'
  filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]
  def Extract_Annotation_Data(file_name, root):
      tokens = file_name.split('_')
      recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location', 'Acquisition mode', 'Recording equipment'])
      recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'],delimiter = '\t')
      return (recording_info, recording_annotations)
  i_list = []
  rec_annotations = []
  rec_annotations_dict = {}
  for s in filenames:
       (i,a) = Extract_Annotation_Data(s,root)
       i_list.append(i)
       rec_annotations.append(a)
       rec_annotations_dict[s] = a
  recording_info = pd.concat(i_list, axis = 0)
  recording_info.head()
  ```
<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/134063047/bf0ac7cd-094e-44c2-ba39-4e46b21ab196" alt="Image" width="70%" height="70%">
</div>

- 환자별 호흡 파일에서 호흡 소리 중 crackle과 wheeze의 정보를 파악한 후, 데이터를 나열화합니다.
- no_label_list: crackle과 wheeze 둘 다 없는 호흡
- crack_list: crack만 있는 호흡
- wheeze_list: wheeze만 있는 호흡
- both_sym_list: 둘 다 있는 호흡
```py
  no_label_list = []
  crack_list = []
  wheeze_list = []
  both_sym_list = []
  filename_list = []
  for f in filenames:
      d = rec_annotations_dict[f]
      no_labels = len(d[(d['Crackles'] == 0) & (d['Wheezes'] == 0)].index)
      n_crackles = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 0)].index)
      n_wheezes = len(d[(d['Crackles'] == 0) & (d['Wheezes'] ==1)].index)
      both_sym = len(d[(d['Crackles'] == 1) & (d['Wheezes'] == 1)].index)
      no_label_list.append(no_labels)
      crack_list.append(n_crackles)
      wheeze_list.append(n_wheezes)
      both_sym_list.append(both_sym)
      filename_list.append(f)
  file_label_df = pd.DataFrame(data = {'filename':filename_list, 'no label':no_label_list, 'crackles only':crack_list, 
                                     'wheezes only':wheeze_list, 'crackles and wheezees':both_sym_list})
```

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/134063047/aa087064-7b09-4471-ae2b-1cde42521e98" alt="Image" width="70%" height="70%">
</div>

## 2. Utility functions for reading .wav files
- read_wav_file: 주어진 파일을 열고, 데이터를 추출한 뒤 목표 샘플링 속도로 리샘플링
- resample: 현재 샘플링 속도와 목표 샘플링 속도에 따라 데이터를 리샘플링
- extract2FloatArr: 주어진 wav파일로부터 음성 데이터 추출. bps(비트 당 샘플)에 따라 데이터 정규화
- read24bitwave: 24비트 wav 파일에서 데이터 추출 후 16비트로 반환
- bitrate_channels: wav 파일에서 bps와 채널 수 추출
- slice_data: 시간 범위 설정
```py
w_labels = file_label_df[(file_label_df['crackles only'] != 0) | (file_label_df['wheezes only'] != 0) | 
                         (file_label_df['crackles and wheezes'] != 0)]

def read_wav_file(str_filename,target_rate):
    wav = wave.open(str_filename, mode = 'r')
    (sample_rate, data) = extract2FloatArr(wav,str_filename)
    
    if (sample_rate != target_rate):
        (_, data) = resample(sample_rate, data, target_rate)
        
        wav.close()
        return(target_rate, data.astype(np.float32))
    
def resample(current_rate, data, target_rate):
    x_original = np.linspace(0,100,len(data))
    x_resampled = np.linspace(0,100,int(len(data) * (target_rate / current_rate)))
    resampled = np.interp(x_resampled, x_original, data)
    return (target_rate, resampled.astype(np.float32))

def extract2FloatArr(lp_wave, str_filename):
    (bps,channels) = bitrate_channels(lp_wave)
    
    if bps in [1, 2, 4]:
        (rate, data) = wf.read(str_filename)
        divisor_dict = {1:255, 2:32768}
        if bps in [1,2]:
            divisor = divisor_dict[bps]
            data = np.divide(data, float(divisor))
        return (rate, data)
    
    elif bps == 3:
        return read24bitwave(lp_wave)
    
    else:
          raise Exception('Unrecognized wave format: {} bytes per samples'.format(bps))
            
def read24bitwave(lp_wave):
    nFrames = lp_wave.getnframes()
    buf = lp_wave.readframes(nFrames)
    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames,-1)
    short_output = np.empty((nFrames, 2), dtype = np.int8)
    short_output[:,:] = reshaped[:, -2:]
    short_output = short_output.view(np.int16)
    return (lp_wave.getframerate(), np.divide(short_output, 32768).reshape(-1))   

def bitrate_channels(lp_wave):
    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels())
    return (bps, lp_wave.getnchannels())

def slice_data(start, end, raw_data, sample_rate):
    max_ind = len(raw_data)
    start_ind = min(int(start * sample_rate),max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]
```
## 3. Distribution of respiratory cycle lengths
```py
duration_list = []
for i in range(len(rec_annotations)):
    current = rec_annotations[i]
    duration = current['End'] - current['Start']
    duration_list.extend(duration)

duration_list = np.array(duration_list)
plt.hist(duration_list, bins = 50)
plt.xlabel('Time(sec)')
plt.ylabel('Number')
plt.title('Respiration Cycle distribution')
print('longest cycle:{}'.format(max(duration_list)))
print('shortest cycle{}'.format(min(duration_list)))
threshold = 5
print('Fraction of samples less than {} seconds {}'.format(threshold, np.sum(duration_list < threshold)/len(duration_list)))
```

<div align="center">
  <img src="https://github.com/YUUIJIN/YUUIJIN.github.io/assets/134063047/4e663208-7305-42f3-989e-055c678a5d29" alt="Image" width="40%" height="40%">
</div>

## 4. Mel spectrogram implementation (With VTLP)
- n_rows: Melspectrogram에서 주파수의 갯수입니다(fft알고리즘에서 주파수의 개수).
- n_window: Window함수의 길이입니다.
- Sxx는 주파수의 power입니다.
- scipy.signal.spectrogram을 사용하여 cycle_info로부터 스펙트로그램을 계산합니다.
```py
import scipy.signal
def sample2MelSpectrum(cycle_info, sample_rate, n_filters, vtlp_params):
    n_rows = 175
    n_window = 512
    (f, t, Sxx) = scipy.signal.spectrogram(cycle_info[0], fs = sample_rate, nfft = n_window, nperseg = n_window)
    Sxx = Sxx[:n_rows, :].astype(np.float32)
    mel_log = FFT2MelSpectrogram(f[:n_rows], Sxx, sample_rate, n_filters, vtlp_params)
    mel_min = np.min(mel_log)
    mel_max = np.max(mel_log)
    diff = mel_max - mel_min
    norm_mel_log = (mel_log - mel_min) / diff  if (diff > 0) else np.zeros(shape = (n_filters, Sxx.shape[1]))
    if (diff == 0):
      print('Error: sample data is completely empty')
    labels = [cycle_info[1], cycle_info[2]]  #crackles, wheezes flags
    return (np.reshape(norm_mel_log, (n_filters,Sxx.shape[1],1)).astype(np.float32), label2onehot(labels))
```
- Freq2Mel과 Mel2Freq 함수는 Mel 스케일과 주파수를 상호변환하는 함수입니다.
```py
def Freq2Mel(freq):
    return 1125 * np.log(1 + freq / 700)

def Mel2Freq(mel):
    exponents = mel / 1125
    return 700 * (np.exp(exponents) - 1)
```
- VTLP(Variable Time and Frequency Tiling)은 Mel주파수를 변형시키는 함수입니다.
```py
def VTLP_shift(mel_freq, alpha, f_high, sample_rate):
    nyquist_f = sample_rate / 2
    warp_factor = min(alpha, 1)
    threshold_freq = f_high * warp_factor / alpha
    lower = mel_freq * alpha
    higher = nyquist_f - (nyquist_f - mel_freq) * ((nyquist_f - f_high * warp_factor) / (nyquist_f - f_high * (warp_factor / alpha)))
    
    warped_mel = np.where(mel_freq <= threshold_freq, lower, higher)
    return warped_mel.astype(np.float32)
```
-
```py
def GenerateMelFilterBanks(mel_space_freq, fft_bin_frequencies):
    n_filters = len(mel_space_freq) - 2
    coeff = []
     for mel_index in range(n_filters):
        m = int(mel_index + 1)
        filter_bank = []
        for f in fft_bin_frequencies:
            if(f < mel_space_freq[m-1]):
                hm = 0
            elif(f < mel_space_freq[m]):
                hm = (f - mel_space_freq[m-1]) / (mel_space_freq[m] - mel_space_freq[m-1])
            elif(f < mel_space_freq[m + 1]):
                hm = (mel_space_freq[m+1] - f) / (mel_space_freq[m + 1] - mel_space_freq[m])
            else:
                hm = 0
            filter_bank.append(hm)
        coeff.append(filter_bank)
    return np.array(coeff, dtype = np.float32)
```
-
```py
def FFT2MelSpectrogram(f, Sxx, sample_rate, n_filterbanks, vtlp_params = None):
    (max_mel, min_mel)  = (Freq2Mel(max(f)), Freq2Mel(min(f)))
    mel_bins = np.linspace(min_mel, max_mel, num = (n_filterbanks + 2))
    mel_freq = Mel2Freq(mel_bins)
    
    if(vtlp_params is None):
        filter_banks = GenerateMelFilterBanks(mel_freq, f)
    else:
        (alpha, f_high) = vtlp_params
        warped_mel = VTLP_shift(mel_freq, alpha, f_high, sample_rate)
        filter_banks = GenerateMelFilterBanks(warped_mel, f)
        
    mel_spectrum = np.matmul(filter_banks, Sxx)
    return (mel_freq[1:-1], np.log10(mel_spectrum  + float(10e-12)))
```
-
```py
def label2onehot(c_w_flags):
    c = c_w_flags[0]
    w = c_w_flags[1]
    if((c == False) & (w == False)):
        return [1,0,0,0]
    elif((c == True) & (w == False)):
        return [0,1,0,0]
    elif((c == False) & (w == True)):
        return [0,0,1,0]
    else:
        return [0,0,0,1]
```

## 5. Data prepation utility functions
- 이 함수는 wav 데이터에서 특정 주기(slice data)를 기반으로 샘플을 추출합니다.
- 이 함수의 출력인 sample_data = [파일이름, (wav 데이터, 호흡주기 시작, 호흡주기 끝, crackle여부 wheeze여부)]입니다.
```py
def get_sound_samples(recording_annotations, file_name, root, sample_rate):
    sample_data = [file_name]
    (rate, data) = read_wav_file(os.path.join(root, file_name + '.wav'), sample_rate)
    
    for i in range(len(recording_annotations.index)):
        row = recording_annotations.loc[i]
        start = row['Start']
        end = row['End']
        crackles = row['Crackles']
        wheezes = row['Wheezes']
        audio_chunk = slice_data(start, end, data, rate)
        sample_data.append((audio_chunk, start,end,crackles,wheezes))
    return sample_data
```
- 이 함수는 wav 데이터의 길이를 설정하여 분할합니다.
```py
def split_and_pad(original, desiredLength, sampleRate):
    output_buffer_length = int(desiredLength * sampleRate)
    soundclip = original[0]
    n_samples = len(soundclip)
    total_length = n_samples / sampleRate #length of cycle in seconds
    n_slices = int(math.ceil(total_length / desiredLength)) #get the minimum number of slices needed
    samples_per_slice = n_samples // n_slices
    src_start = 0 #Staring index of the samples to copy from the original buffer
    output = [] #Holds the resultant slices
    for i in range(n_slices):
        src_end = min(src_start + samples_per_slice, n_samples)
        length = src_end - src_start
        copy = generate_padded_samples(soundclip[src_start:src_end], output_buffer_length)
        output.append((copy, original[1], original[2]))
        src_start += length
    return output
```
-이 함수는 zero padding된 데이터를 생성합니다.
```py
def generate_padded_samples(source, output_length):
    copy = np.zeros(output_length, dtype = np.float32)
    src_length = len(source)
    frac = src_length / output_length
    if(frac < 0.5):
        #tile forward sounds to fill empty space
        cursor = 0
        while(cursor + src_length) < output_length:
            copy[cursor:(cursor + src_length)] = source[:]
            cursor += src_length
    else:
        copy[:src_length] = source[:]
    return copy
```
## 6. Data augmentation
- gen_time_strech 함수는 입력받은 original데이터를 무작위로 늘리거나 줄입니다.
- augment_list 함수는 audio_with_labels에 있는 데이터를 gen_time_stretch를 이용하여 증강시킵니다.
- split_and_pad_and_apply_mel_spect 함수는 데이터를 여러 번 분할과 padding하고, 멜 스펙트로그램으로 변환하며 VTLP 함수를 적용합니다. 
```py
def gen_time_stretch(original, sample_rate, max_percent_change):
    stretch_amount = 1 + np.random.uniform(-1,1) * (max_percent_change / 100)
    (_, stretched) = resample(sample_rate, original, int(sample_rate * stretch_amount)) 
    return stretched

def augment_list(audio_with_labels, sample_rate, percent_change, n_repeats):
    augmented_samples = []
    for i in range(n_repeats):
        addition = [(gen_time_stretch(t[0], sample_rate, percent_change), t[1], t[2] ) for t in audio_with_labels]
        augmented_samples.extend(addition)
    return augmented_samples

def split_and_pad_and_apply_mel_spect(original, desiredLength, sampleRate, VTLP_alpha_range = None, VTLP_high_freq_range = None, n_repeats = 1):
    output = []
    for i in range(n_repeats):
        for d in original:
            lst_result = split_and_pad(d, desiredLength, sampleRate) #Time domain
            if( (VTLP_alpha_range is None) | (VTLP_high_freq_range is None) ):
                #Do not apply VTLP
                VTLP_params = None
            else:
                #Randomly generate VLTP parameters
                alpha = np.random.uniform(VTLP_alpha_range[0], VTLP_alpha_range[1])
                high_freq = np.random.uniform(VTLP_high_freq_range[0], VTLP_high_freq_range[1])
                VTLP_params = (alpha, high_freq)
            freq_result = [sample2MelSpectrum(d, sampleRate, 50, VTLP_params) for d in lst_result] #Freq domain
            output.extend(freq_result)
    return output
```
- 이 코드는 주어진 데이터에 대해 Mel spectrogram으로 시각화합니다.
```py
str_file = filenames[11]
lp_test = get_sound_samples(rec_annotations_dict[str_file], str_file, root, 22000)
lp_cycles = [(d[0], d[3], d[4]) for d in lp_test[1:]]
soundclip = lp_cycles[1][0]

n_window = 512
sample_rate = 22000
(f, t, Sxx) = scipy.signal.spectrogram(soundclip, fs = 22000, nfft= n_window, nperseg=n_window)
print(sum(f < 7000))

plt.figure(figsize = (20,10))
plt.subplot(1,2,1)
mel_banks = FFT2MelSpectrogram(f[:175], Sxx[:175,:], sample_rate, 50)[1]
plt.imshow(mel_banks, aspect = 1)
plt.title('No VTLP')

plt.subplot(1,2,2)
mel_banks = FFT2MelSpectrogram(f[:175], Sxx[:175,:], sample_rate, 50, vtlp_params = (0.9,3500))[1]
plt.imshow(mel_banks, aspect = 1)
plt.title('With VTLP')
```
# Ⅳ. Evaluation & Analysis

