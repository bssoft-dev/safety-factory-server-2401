import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.event_classification import mobilenet_v3_mfcc
import torch, numpy as np, librosa
import os

def get_mfcc(npaudio: np.ndarray, n_mfcc=40, max_len=50, sr=16000):
    # MFCC 변환
    try:
        mfcc = librosa.feature.mfcc(y=npaudio, sr=sr, n_mfcc=n_mfcc)
        npaudio = []
    except Exception as e:
        print(f"Error getting MFCC: {e}")
        return None
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def infer(npaudio: np.ndarray, model, class_names, device):
    # 오디오 파일 로드 및 MFCC 변환
    mfcc = get_mfcc(npaudio)
    # MFCC를 모델 입력 형식으로 변환
    mfcc = np.expand_dims(mfcc, axis=(0, 1))  # (1, 1, n_mfcc, max_len) 형태로 변환
    mfcc_tensor = torch.FloatTensor(mfcc).to(device)
    # 예측 수행
    with torch.no_grad():
        outputs = model(mfcc_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    # 결과 반환
    predicted_label = class_names[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_label, confidence
    

if __name__ == "__main__":
    device = 'cuda:1'
    model = mobilenet_v3_mfcc(num_classes=4)
    model.load_state_dict(torch.load('ai_models/classification/best_mobilenetV3_mfcc_model.pth', weights_only=True))
    model.to(device)
    class_names = ['경보음', '비명', '일반', '충격깨짐소리']
    eng_class_names = ['siren', 'scream', 'normal', 'crash']
    model.eval()
    
    BASE_DIR = "tests/sounds/classification_test"
    total = 0; correct = 0
    print("========  Emergency Event Classification Test  ========")
    for i in range(4):
        ground_truth = class_names[i]
        sub_total = 0; sub_correct = 0
        for wav in os.listdir(os.path.join(BASE_DIR, ground_truth)):
            if wav.endswith(".wav"):
                npaudio = librosa.load(os.path.join(BASE_DIR, ground_truth, wav), offset=0.02, sr=16000)[0]
                res = infer(npaudio, model, eng_class_names, device)
                if res[0] == eng_class_names[i]:
                    sub_correct += 1
                sub_total += 1
                print(f"{sub_total} {wav}, Predict: {res[0]}, Match: {res[0] == eng_class_names[i]}")
        print("--------------------------------")
        print(f"{ground_truth} accuracy: {sub_correct / sub_total:.3f}")
        print("--------------------------------")
        total += sub_total
        correct += sub_correct
    print("--------------------------------")
    print(f"total accuracy: {correct / total: .3f}")
