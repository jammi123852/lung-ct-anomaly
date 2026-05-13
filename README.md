\# Lung CT Anomaly



정상 폐 CT 데이터를 기반으로 폐 영역 전처리, 위치별 patch 분할, 학습용 데이터셋 구성을 수행하고, 이후 폐 이상탐지 모델을 개발하기 위한 프로젝트입니다.



\## Project Structure



```text

lung-ct-anomaly/

├─ preprocessing/

│  ├─ notebooks/

│  │  ├─ 00\_preprocess\_final.ipynb

│  │  ├─ 01\_extract\_patches\_resume.ipynb

│  │  ├─ 02\_build\_training\_ready\_dataset.ipynb

│  │  ├─ 03\_visualize\_and\_sanity\_check.ipynb

│  │  ├─ 04\_build\_slim\_training\_dataset\_optional.ipynb

│  │  └─ old/

│  ├─ docs/

│  └─ configs/

├─ model/

│  ├─ src/

│  ├─ configs/

│  └─ experiments/

├─ docs/

├─ scripts/

└─ data/

