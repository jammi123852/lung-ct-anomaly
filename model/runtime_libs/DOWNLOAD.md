# 대용량 런타임 파일 다운로드 (Google Drive)

GitHub 100MB 파일 한도 때문에 아래 대용량 파일은 저장소에 포함하지 않습니다.
**Google Drive에서 받아 압축 해제 후 지정 위치에 놓으세요.**

| 파일 | 압축 해제 위치 | 설명 | 다운로드 |
|------|----------------|------|----------|
| `lunar_env.zip` (≈2.6GB) | `model/lunar_env/` | Windows 임베디드 Python + 전체 라이브러리(torch, TotalSegmentator 등) | <Google Drive 링크 넣기> |
| `totalseg_data.zip` (≈1.5GB) | `model/pipeline_code/totalseg_data/` | TotalSegmentator nnU-Net 가중치 (`nnunet/results/...`) | <Google Drive 링크 넣기> |

> 모델 가중치(`pipeline_code/weights/`, ≈118MB)는 저장소에 포함되어 있어 별도 다운로드가 필요 없습니다.

## 압축 해제 후 디렉토리 구조 확인
```
model/
├── lunar_env/                      # ← lunar_env.zip 압축 해제
│   └── python.exe ...
├── pipeline_code/
│   ├── totalseg_data/              # ← totalseg_data.zip 압축 해제
│   │   └── nnunet/results/Dataset291_.../...
│   └── weights/                    # (저장소 포함)
└── start.bat
```

## B2(Backblaze) 자격증명
`pipeline_code/b2_secrets.json.example` → `b2_secrets.json` 로 복사 후 본인 키 입력.
(이 파일은 `.gitignore`로 저장소에 올라가지 않습니다.)
