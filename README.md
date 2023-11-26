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
# Ⅲ. Methodology
# Ⅳ. Evaluation & Analysis
## Overview
- 분석에 기본적으로 필요한 파이썬 라이브러리를 가져옵니다.
  
      import wave
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import os
      %matplotlib inline
  
- 데이터셋이 들어있는 파일을 확인합니다

      os.listdir('Respiratory_Sound_Database')

-  df_no_diagnosis: 환자번호 + 질병 외의 정보
-  diagnosis: 환자 번호 + 질병 정보

       df_no_diagnosis = pd.read_csv('Respiratory_Sound_Database/demographic_info.txt', names = ['Patient number', 'Age', 'Sex', 'Adult BMI(kg/m^2)', 'Child Weight(kg)','Child Height(cm)'], delimiter = ' ')
       diagnosis = pd.read_csv('Respiratory_Sound_Database/patient_diagnosis.csv', names = ['Patient number', 'Diagnosis'])

