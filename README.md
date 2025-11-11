# Omnilingual ASR 실전 튜토리얼

Facebook Research의 Omnilingual ASR을 활용한 다국어 음성 인식 실습 프로젝트

## 📖 프로젝트 개요

이 프로젝트는 **Omnilingual ASR** (Automatic Speech Recognition) 기술을 소개하고 실전 예제를 통해 구현 방법을 배우는 것을 목표로 합니다. 100개 이상의 언어를 지원하는 혁신적인 다국어 음성 인식 모델을 직접 체험해볼 수 있습니다.

## 🚀 주요 특징

- **다국어 지원**: 100개 이상의 언어를 단일 모델로 처리
- **실전 예제**: 초보자도 따라할 수 있는 상세한 튜토리얼
- **성능 최적화**: 배치 처리, GPU 가속 등 최적화 기법 포함
- **한국어 지원**: 한국어 주석과 설명 제공

## 📁 프로젝트 구조

```
omnilingual-asr/
├── README.md                    # 프로젝트 설명
├── omnilingual-asr-blog.html     # 블로그 포스팅 (티스토리용)
├── src/
│   └── test.py                # 통합 실전 예제 코드
├── myenv/                      # Python 가상환경
└── .gitignore                  # Git 추적 제외 파일
```

## 🛠️ 시작하기

### 사전 요구사항

- Python 3.8 이상
- PyTorch 1.9 이상
- 최소 8GB RAM (권장: 16GB)
- GPU 환경 (권장: RTX 3060 이상)

### 설치 단계

```bash
# 1. 저장소 클론
git clone https://github.com/jmpark333/omnilingual-asr.git
cd omnilingual-asr

# 2. 가상환경 생성 및 활성화
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# myenv\Scripts\activate  # Windows

# 3. 필요한 라이브러리 설치
pip install torch torchvision torchaudio
pip install transformers datasets
pip install librosa soundfile
pip install jiwer

# 4. 예제 실행
python src/test.py
```

## 📚 사용 가이드

### 1. 블로그 포스팅
`omnilingual-asr-blog.html` 파일은 티스토리에 바로 복사해서 사용할 수 있는 완성된 블로그 포스팅입니다. 다음 내용을 포함합니다:

- **기술적 개요**: Omnilingual ASR의 아키텍처와 원리
- **성능 비교**: 언어 그룹별 WER 성능 지표
- **실전 튜토리얼**: 5단계로 구성된 상세 예제
- **고급 기법**: 언어 자동 식별, 실시간 스트리밍
- **실제 사례**: 고객 서비스, 교육, 콘텐츠 제작 분야

### 2. 통합 예제 코드
`src/test.py`는 다음 기능을 제공하는 완전한 예제입니다:

```python
from src.test import OmnilingualASR

# ASR 모델 초기화
asr = OmnilingualASR()

# 단일 음성 파일 처리
result = asr.transcribe_audio("audio.wav")

# 배치 처리
results = asr.batch_transcribe(["audio1.wav", "audio2.wav"])

# 성능 평가
asr.evaluate_performance(predictions, ground_truths)
```

## 🎯 핵심 기능

### OmnilingualASR 클래스
- **모델 초기화**: `facebook/omniASR_CTC_1B` 모델 자동 로드
- **음성 전처리**: 정규화, 샘플링 레이트 변환
- **단일 파일 처리**: 개별 음성 파일 텍스트 변환
- **배치 처리**: 여러 파일 동시 처리 및 성능 최적화
- **성능 평가**: WER 계산 및 정확도 측정
- **샘플 생성**: 테스트용 음성 파일 자동 생성

### 지원 모델
- **omniASR_CTC_1B**: Facebook Research의 최신 다국어 ASR 모델
- **1억 파라미터**: 대규모 모델로 높은 정확도 제공
- **CTC 아키텍처**: 효율적인 시퀀스 학습

## 📊 성능 지표

| 언어 그룹 | 지원 언어 수 | WER(%) | 특징 |
|-----------|-------------|---------|-------|
| 고자원 언어 | 15개 | 8.2 | 영어, 중국어, 스페인어 등 |
| 중간 자원 언어 | 35개 | 12.5 | 한국어, 일본어, 독일어 등 |
| 저자원 언어 | 50개 | 18.7 | 아프리카, 동남아시아 언어 등 |

## 🎨 사용 예시

### 기본 사용법
```python
# 모델 로드
asr = OmnilingualASR()

# 음성 파일 텍스트 변환
text = asr.transcribe_audio("sample.wav")
print(f"인식 결과: {text}")
```

### 배치 처리
```python
# 여러 파일 동시 처리
files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = asr.batch_transcribe(files, batch_size=2)
for i, result in enumerate(results):
    print(f"파일 {i+1}: {result}")
```

### 성능 평가
```python
# WER 기반 성능 측정
predictions = ["안녕하세요", "음성 인식"]
ground_truths = ["안녕하세요", "음성 인식"]
wer_score = asr.evaluate_performance(predictions, ground_truths)
```

## ⚡ 최적화 팁

1. **GPU 활용**: CUDA 지원 시 자동으로 GPU 사용
2. **배치 크기 조절**: 메모리에 맞는 최적 크기 찾기
3. **모델 양자화**: 추론 속도 향상 가능
4. **캐싱 활용**: 반복 계산 결과 저장

## 🌍 실제 활용 분야

- **글로벌 고객 서비스**: 다국어 콜센터, 자동 응답 시스템
- **교육 분야**: 언어 학습 앱, 온라인 교육 플랫폼
- **콘텐츠 제작**: YouTube 자동 자막, 팟캐스트 텍스트화
- **접근성**: 청각 장애인을 위한 음성-텍스트 변환

## 🔮 미래 발전 방향

- **200개 언어 지원**: 현재 100개에서 더 많은 언어로 확장
- **실시간 번역 통합**: 음성 인식과 번역 결합
- **엣지 디바이스 최적화**: 스마트폰 등에서 로컬 실행
- **개인화된 모델**: 사용자별 맞춤형 ASR

## 📖 참고 자료

- [Facebook Research Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
- [Hugging Face Model Hub](https://huggingface.co/facebook/omniASR_CTC_1B)
- [Wav2Vec 2.0 Paper](https://arxiv.org/abs/2006.11477)

## 🤝 기여

이 프로젝트는 다음을 통해 기여할 수 있습니다:

1. **이슈 리포트**: 버그나 개선사항 제안
2. **풀 리퀘스트**: 코드 개선 및 새로운 기능 추가
3. **문서 개선**: 설명 추가 및 오타 수정

## 📄 라이선스

이 프로젝트는 MIT License 하에 배포됩니다. 자유롭게 사용, 수정, 배포할 수 있습니다.

## 👥 작성자

- **AI 기술 블로그**: 최신 AI 기술 동향과 실전 예제 제공
- **GitHub**: [jmpark333](https://github.com/jmpark333)

---

<div align="center">
  <p>🤖 Omnilingual ASR로 언어 장벽 없는 세상을 만들어요! 🌍</p>
  <p>⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! ⭐</p>
</div>
