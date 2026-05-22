import os
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# 폰트 깨짐 방지 (윈도우 환경 기준 한글 폰트 설정)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 데이터셋 로드 (폴더 트리 경로 반영)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# server 폴더를 빠져나가서 data/processed 내부의 csv를 조준합니다.
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "valid_processed.csv")
df = pd.read_csv(DATA_PATH)

# 컬럼명에 맞게 데이터 추출
sentences = df['player_input'].tolist()
raw_true_labels = df['label'].tolist() 

print(f"✅ 검증 데이터셋 로드 완료! 총 {len(sentences)}개 문장 평가 시작.")

# ==========================================
# 2. 감정 텍스트 -> 숫자 인덱스 매핑 및 이진화
# ==========================================
label_map = {
    "anger": 0,
    "anxiety": 1,
    "sadness": 2,
    "hurt": 3,
    "embarrassment": 4,
    "bad": 5
}
n_classes = len(label_map)
emotion_labels = list(label_map.keys())

# 정답지(csv 레이블)를 숫자로 강제 변환
true_labels = np.array([label_map[str(label).strip().lower()] for label in raw_true_labels])

# ROC 곡선을 계산하기 위해 정답지를 이진화(Binarize) 처리 (One-vs-Rest 방식)
true_labels_bin = label_binarize(true_labels, classes=list(range(n_classes)))

# ==========================================
# 3. 모델 및 토크나이저 로드
# ==========================================
print("🔄 두 모델 로딩 시작...")

# (1) TF-IDF + Logistic Regression 모델 로드
TFIDF_LR_MODEL_PATH = os.path.join(BASE_DIR, "models", "tfidf_6class_model.pkl")
tfidf_lr_model = joblib.load(TFIDF_LR_MODEL_PATH)

# (2) KLUE-BERT 모델 및 토크나이저 로드
BERT_MODEL_PATH = os.path.join(BASE_DIR, "models", "klue_bert_model")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
bert_model.eval()

print("✅ TF+LR 모델 및 KLUE-BERT 에셋 로드 완료!")

# ==========================================
# 4. 모델별 추론, 속도 측정 및 확률값 추출
# ==========================================

# --- (1) TF-IDF + LR 추론 ---
lr_start = time.time()
raw_lr_preds = tfidf_lr_model.predict(sentences)
# ROC를 위해 TF-IDF 모델의 예측 확률(Probability) 추출
lr_probs = tfidf_lr_model.predict_proba(sentences) 
lr_latency = (time.time() - lr_start) / len(sentences)

if isinstance(raw_lr_preds[0], str):
    lr_preds = [label_map[str(p).strip().lower()] for p in raw_lr_preds]
else:
    lr_preds = [int(p) for p in raw_lr_preds]

# --- (2) KLUE-BERT 추론 ---
bert_preds = []
bert_probs = []
bert_start = time.time()

with torch.no_grad():
    for text in sentences:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = bert_model(**inputs)
        
        # Softmax를 통과시켜 0~1 사이의 확률값으로 변환
        probs = F.softmax(outputs.logits, dim=1).numpy()[0]
        bert_probs.append(probs)
        bert_preds.append(np.argmax(probs))

bert_probs = np.array(bert_probs)
bert_latency = (time.time() - bert_start) / len(sentences)

print("✅ 두 모델의 추론 연산 전과정 종료!")

# ==========================================
# 5. 성능 지표 계산
# ==========================================
lr_acc = accuracy_score(true_labels, lr_preds)
lr_f1 = f1_score(true_labels, lr_preds, average='macro')

bert_acc = accuracy_score(true_labels, bert_preds)
bert_f1 = f1_score(true_labels, bert_preds, average='macro')

print(f"📊 TF+LR -> 정확도: {lr_acc:.2f} | F1-Score: {lr_f1:.2f}")
print(f"📊 BERT  -> 정확도: {bert_acc:.2f} | F1-Score: {bert_f1:.2f}")

# ==========================================
# 6. 결과 시각화 (5분할 대시보드 - ROC 포함)
# ==========================================
print("🎨 대시보드 그래프 생성 및 화면 출력 중...")
# 하단에 ROC 곡선을 추가하기 위해 세로 길이를 늘림 (10 -> 14)
fig = plt.figure(figsize=(16, 14))
models_label = ['TF+LR 모델', 'KLUE-BERT 모델']

# 6-1. 성능 비교 차트
ax1 = plt.subplot(3, 2, 1)
x = np.arange(len(models_label))
width = 0.35
ax1.bar(x - width/2, [lr_acc, bert_acc], width, label='Accuracy', color='#ff9999')
ax1.bar(x + width/2, [lr_f1, bert_f1], width, label='F1-Score', color='#66b3ff')
ax1.set_title('모델별 분류 성능 비교', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(models_label, fontsize=11)
ax1.set_ylim(0, 1.1)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# 6-2. 지연 시간 비교 차트
ax2 = plt.subplot(3, 2, 2)
latencies = [lr_latency, bert_latency]
bars2 = ax2.bar(models_label, latencies, color=['#ffcc99', '#99ff99'], width=0.4)
ax2.set_title('문장 1개당 평균 추론 지연 시간 (초)', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, max(latencies) * 1.3)
for bar in bars2:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}초', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# 6-3. TF+LR 혼동 행렬
ax3 = plt.subplot(3, 2, 3)
cm_lr = confusion_matrix(true_labels, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds', ax=ax3, cbar=False, 
            xticklabels=emotion_labels, yticklabels=emotion_labels)
ax3.set_title('TF+LR 모델 예측 분포 (Confusion Matrix)', fontsize=12, fontweight='bold')
ax3.set_xlabel('예측된 감정 (Predicted)')
ax3.set_ylabel('실제 감정 (True)')

# 6-4. KLUE-BERT 혼동 행렬
ax4 = plt.subplot(3, 2, 4)
cm_bert = confusion_matrix(true_labels, bert_preds)
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False,
            xticklabels=emotion_labels, yticklabels=emotion_labels)
ax4.set_title('KLUE-BERT 모델 예측 분포 (Confusion Matrix)', fontsize=12, fontweight='bold')
ax4.set_xlabel('예측된 감정 (Predicted)')
ax4.set_ylabel('실제 감정 (True)')

# 6-5. 추가된 구역: KLUE-BERT 다중 클래스 ROC 곡선 (하단 전체 통합)
ax5 = plt.subplot(3, 1, 3)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(true_labels_bin[:, i], bert_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax5.plot(fpr, tpr, color=colors[i], lw=2,
             label=f'ROC curve of {emotion_labels[i]} (AUC = {roc_auc:.2f})')

ax5.plot([0, 1], [0, 1], 'k--', lw=1.5)
ax5.set_xlim([0.0, 1.0])
ax5.set_ylim([0.0, 1.05])
ax5.set_xlabel('False Positive Rate (오탐률)', fontsize=11)
ax5.set_ylabel('True Positive Rate (정탐률)', fontsize=11)
ax5.set_title('KLUE-BERT 모델 감정별 다중 클래스 ROC 곡선 및 AUC', fontsize=13, fontweight='bold', pad=10)
ax5.legend(loc="lower right", fontsize=11)
ax5.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout(pad=3.0)

# 💾 파일로 고해상도 저장 완료
plt.savefig('final_evaluation_dashboard_with_roc.png', dpi=300)

# 📢 모니터 화면에 새 팝업 창으로 대시보드 강제 전시
plt.show() 

print("🚀 [성공] 대시보드 팝업창 종료 완료!")