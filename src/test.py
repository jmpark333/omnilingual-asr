#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Omnilingual ASR 실전 튜토리얼 통합 예제
Facebook Research의 Omnilingual ASR을 활용한 다국어 음성 인식 데모

작성자: AI 기술 블로그
버전: 1.0
"""

import torch
import librosa
import numpy as np
import time
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer
import os
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

class OmnilingualASR:
    """
    Omnilingual ASR을 쉽게 사용할 수 있는 클래스
    """
    
    def __init__(self, model_name="facebook/omniASR_CTC_1B"):
        """
        모델 초기화
        
        Args:
            model_name (str): 사용할 모델 이름
        """
        print("🚀 Omnilingual ASR 모델을 초기화합니다...")
        
        # GPU 사용 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 사용 디바이스: {self.device}")
        
        # 모델과 프로세서 로드
        print("📥 모델과 프로세서를 로드합니다...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        print(f"✅ 모델 로드 완료!")
        print(f"🌍 지원 언어 수: {self.processor.tokenizer.vocab_size}")
        print(f"🎯 모델 파라미터 수: {self.model.num_parameters():,}")
        
    def preprocess_audio(self, audio_path, target_sr=16000):
        """
        음성 파일을 전처리하는 함수
        
        Args:
            audio_path (str): 음성 파일 경로
            target_sr (int): 목표 샘플링 레이트
            
        Returns:
            numpy.ndarray: 전처리된 음성 데이터
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"❌ 음성 파일을 찾을 수 없습니다: {audio_path}")
        
        # 음성 파일 로드
        speech, sr = librosa.load(audio_path, sr=target_sr)
        
        # 정규화
        if np.max(np.abs(speech)) > 0:
            speech = speech / np.max(np.abs(speech))
        
        return speech
    
    def transcribe_audio(self, audio_path, language="korean"):
        """
        단일 음성 파일을 텍스트로 변환
        
        Args:
            audio_path (str): 음성 파일 경로
            language (str): 언어 설정 (현재는 참고용)
            
        Returns:
            str: 변환된 텍스트
        """
        print(f"🎵 음성 파일 처리 중: {audio_path}")
        
        # 음성 전처리
        speech = self.preprocess_audio(audio_path)
        
        # 입력 값으로 변환
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        
        # 모델 추론
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # 텍스트로 디코딩
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        
        return transcription[0]
    
    def batch_transcribe(self, audio_files, batch_size=4):
        """
        여러 음성 파일을 배치로 처리
        
        Args:
            audio_files (list): 음성 파일 경로 리스트
            batch_size (int): 배치 크기
            
        Returns:
            list: 변환된 텍스트 리스트
        """
        print(f"📦 배치 처리 시작 (파일 수: {len(audio_files)}, 배치 크기: {batch_size})")
        results = []
        
        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i+batch_size]
            batch_speeches = []
            
            try:
                # 배치 내 음성 파일 전처리
                for audio_file in batch_files:
                    speech = self.preprocess_audio(audio_file)
                    batch_speeches.append(speech)
                
                # 배치 입력으로 변환
                inputs = self.processor(batch_speeches, sampling_rate=16000, return_tensors="pt", padding=True)
                inputs = inputs.to(self.device)
                
                # 배치 추론
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                # 배치 디코딩
                predicted_ids = torch.argmax(logits, dim=-1)
                batch_transcriptions = self.processor.batch_decode(predicted_ids)
                
                results.extend(batch_transcriptions)
                
                print(f"✅ 배치 {i//batch_size + 1}/{(len(audio_files)-1)//batch_size + 1} 완료")
                
            except Exception as e:
                print(f"❌ 배치 처리 중 오류 발생: {e}")
                # 실패한 파일들은 빈 문자열로 처리
                for _ in batch_files:
                    results.append("")
        
        return results
    
    def evaluate_performance(self, transcriptions, ground_truths):
        """
        음성 인식 성능을 평가하는 함수
        
        Args:
            transcriptions (list): 예측 텍스트 리스트
            ground_truths (list): 정답 텍스트 리스트
            
        Returns:
            float: 평균 WER (Word Error Rate)
        """
        if len(transcriptions) != len(ground_truths):
            raise ValueError("❌ 예측 결과와 정답의 개수가 일치하지 않습니다.")
        
        total_wer = 0
        print("\n📊 성능 평가 결과:")
        print("=" * 60)
        
        for i, (pred, truth) in enumerate(zip(transcriptions, ground_truths)):
            current_wer = wer(truth, pred)
            total_wer += current_wer
            
            print(f"📝 파일 {i+1}:")
            print(f"   예측: {pred}")
            print(f"   정답: {truth}")
            print(f"   WER: {current_wer:.4f}")
            print("-" * 40)
        
        avg_wer = total_wer / len(transcriptions)
        print(f"🎯 평균 WER: {avg_wer:.4f} ({(1-avg_wer)*100:.2f}% 정확도)")
        
        return avg_wer
    
    def create_sample_audio(self, output_path="sample_audio.wav", duration=3, sample_rate=16000):
        """
        테스트용 샘플 음성 파일 생성
        
        Args:
            output_path (str): 출력 파일 경로
            duration (int): 지속시간 (초)
            sample_rate (int): 샘플링 레이트
        """
        print(f"🎙️ 샘플 음성 파일 생성 중: {output_path}")
        
        # 간단한 사인파 생성 (테스트용)
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 440Hz와 880Hz 주파수 조합 (A音符)
        frequency1 = 440  # A4
        frequency2 = 880  # A5
        
        # 두 주파수의 조합
        audio_data = 0.5 * np.sin(2 * np.pi * frequency1 * t)
        audio_data += 0.3 * np.sin(2 * np.pi * frequency2 * t)
        
        # 앰플리튜드 조절
        audio_data = audio_data * 0.8
        
        # WAV 파일로 저장
        import soundfile as sf
        sf.write(output_path, audio_data, sample_rate)
        
        print(f"✅ 샘플 음성 파일 생성 완료: {output_path}")
        return output_path


def main():
    """
    메인 실행 함수
    """
    print("=" * 60)
    print("🤖 Omnilingual ASR 실전 튜토리얼")
    print("=" * 60)
    
    try:
        # 1. ASR 모델 초기화
        asr = OmnilingualASR()
        
        # 2. 샘플 음성 파일 생성
        sample_files = []
        for i in range(3):
            sample_path = f"sample_audio_{i+1}.wav"
            asr.create_sample_audio(sample_path, duration=2+i)
            sample_files.append(sample_path)
        
        # 3. 단일 파일 테스트
        print("\n🎯 단일 파일 테스트:")
        print("-" * 40)
        single_result = asr.transcribe_audio(sample_files[0])
        print(f"📝 인식 결과: {single_result}")
        
        # 4. 배치 처리 테스트
        print("\n📦 배치 처리 테스트:")
        print("-" * 40)
        start_time = time.time()
        batch_results = asr.batch_transcribe(sample_files, batch_size=2)
        end_time = time.time()
        
        print(f"⏱️ 총 처리 시간: {end_time - start_time:.2f}초")
        print(f"🚀 평균 처리 속도: {len(sample_files)/(end_time - start_time):.2f} 파일/초")
        
        # 5. 성능 평가 (예제 데이터)
        print("\n📊 성능 평가 테스트:")
        print("-" * 40)
        
        # 예제 데이터 (실제 사용 시에는 실제 음성 파일과 정답 텍스트 필요)
        example_transcriptions = ["hello world", "test recognition", "audio processing"]
        example_ground_truths = ["hello world", "test recognition", "audio processing"]
        
        # 성능 평가 실행
        avg_wer = asr.evaluate_performance(example_transcriptions, example_ground_truths)
        
        # 6. 최종 요약
        print("\n" + "=" * 60)
        print("🎉 튜토리얼 완료!")
        print("=" * 60)
        print("✅ 성능 지표:")
        print(f"   - 평균 WER: {avg_wer:.4f}")
        print(f"   - 정확도: {(1-avg_wer)*100:.2f}%")
        print(f"   - 처리 속도: {len(sample_files)/(end_time - start_time):.2f} 파일/초")
        print("\n💡 팁:")
        print("   - 실제 음성 파일을 사용하려면 sample_audio.wav 파일들을 교체하세요")
        print("   - GPU를 사용하면 처리 속도가 크게 향상됩니다")
        print("   - 더 나은 성능을 위해 fine-tuning을 고려해보세요")
        print("\n🔗 참고: https://github.com/facebookresearch/omnilingual-asr")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("\n🛠️ 해결 방법:")
        print("1. 인터넷 연결 확인")
        print("2. 필요한 라이브러리 설치 확인: pip install torch transformers librosa soundfile jiwer")
        print("3. GPU 드라이버 확인 (CUDA 사용 시)")


if __name__ == "__main__":
    main()
