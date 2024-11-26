from datetime import datetime
from sqlmodel import Session
from database import engine
from models import Events
import torch
import numpy as np
import librosa
from utils.sms import sms_send
from env import SMS_TEST_SEND_LIST
from ai_models.classification.mobilenetV3_mfcc import mobilenet_v3_mfcc


class EventClassifier:
    def __init__(self, device):
        self.device = device
        self.model = mobilenet_v3_mfcc(num_classes=4)
        self.model.load_state_dict(torch.load('ai_models/classification/best_mobilenetV3_mfcc_model.pth'))
        self.model.to(device)
        self.class_names = ['경보음', '비명', '일반', '충격깨짐소리']
        self.model.eval()
        self.buffer = []

    def get_mfcc(self, npaudio: np.ndarray, n_mfcc=40, max_len=50, sr=16000):
        try:
            mfcc = librosa.feature.mfcc(y=np.concatenate(self.buffer), sr=sr, n_mfcc=n_mfcc)
            self.buffer = []
        except Exception as e:
            print(f"Error getting MFCC: {e}")
            return None
        # MFCC의 길이를 max_len으로 맞추기
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        return mfcc
    
    async def infer(self, npaudio: np.ndarray, room_name: str):
        """단일 WAV 파일에 대한 예측을 수행합니다."""
        if len(self.buffer) < 2:
            self.buffer.append(npaudio)
            return None, None
        else:
            self.buffer.append(npaudio)
            # 오디오 파일 로드 및 MFCC 변환
            mfcc = self.get_mfcc(npaudio)
            # MFCC를 모델 입력 형식으로 변환
            mfcc = np.expand_dims(mfcc, axis=(0, 1))  # (1, 1, n_mfcc, max_len) 형태로 변환
            mfcc_tensor = torch.FloatTensor(mfcc).to(self.device)
            # 예측 수행
            with torch.no_grad():
                outputs = self.model(mfcc_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            # 결과 반환
            predicted_label = self.class_names[predicted_class]
            confidence = probabilities[0][predicted_class].item()
            if confidence < 0.8:
                predicted_label = "일반"
            if predicted_label != "일반":
                with Session(engine) as session:
                    session.add(Events(room_name=room_name, event=predicted_label, time=datetime.now()))
                    session.commit()
                try:
                    await sms_send(SMS_TEST_SEND_LIST, f"{room_name}에서 {predicted_label} 이벤트가 발생했습니다.")
                except Exception as e:
                    print(f"SMS send error: {e}")
            
            return predicted_label, confidence