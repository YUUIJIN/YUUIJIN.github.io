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
## Download Dataset
### 1. Demographic Info
### 2. Respiratory Sound Database
### 3. Patient Diagnosis

# Ⅲ. Methodology
# Ⅳ. Evaluation & Analysis
## 1. Overview
- 분석에 기본적으로 필요한 파이썬 라이브러리를 가져옵니다.
  
      import wave
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import os
      %matplotlib inline
  
- 데이터셋이 들어있는 파일을 확인합니다.

      os.listdir('Respiratory_Sound_Database')

-  df_no_diagnosis: 환자번호 + 질병 외의 정보
-  diagnosis: 환자 번호 + 질병 정보
-  df: 위 두 변수를 환자 번호 기준으로 합침
  
       df_no_diagnosis = pd.read_csv('Respiratory_Sound_Database/demographic_info.txt', names = ['Patient number', 'Age', 'Sex', 'Adult BMI(kg/m^2)', 'Child Weight(kg)','Child Height(cm)'], delimiter = ' ')
       diagnosis = pd.read_csv('Respiratory_Sound_Database/patient_diagnosis.csv', names = ['Patient number', 'Diagnosis'])
       df = df_no_diagnosis.join(diagnosis.set_index('Patient number'), on = 'Patient number', how = 'left')

![image](https://github.com/YUUIJIN/YUUIJIN.github.io/assets/134063047/f1f0911f-6b8f-4e11-94d3-305f98f0c0a2)

- df에서 각 질병 빈도수를 counting 합니다.
  
       df['Diagnosis'].value_counts()

- txt 파일로 되어있는 audio 파일을 전처리합니다.
- filenames: 확장자 제거 후 list에 저장
- Extract_Annotation_Data: 파일 이름과 파일 주소를 가져와 데이터를 추출하는 함수
  
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

![image](https://github.com/YUUIJIN/YUUIJIN.github.io/assets/134063047/bf0ac7cd-094e-44c2-ba39-4e46b21ab196)

- 환자별 호흡 파일에서 호흡 소리 중 crackle과 wheeze의 정보를 파악한 후, 데이터를 나열화합니다.
- no_label_list: crackle과 wheeze 둘 다 없는 호흡
- crack_list: crack만 있는 호흡
- wheeze_list: wheeze만 있는 호흡
- both_sym_list: 둘 다 있는 호흡
  
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
![image](https://github.com/YUUIJIN/YUUIJIN.github.io/assets/134063047/aa087064-7b09-4471-ae2b-1cde42521e98)
