# CHAD 동일 조건 Transformer 비교 코드

이 코드는 CHAD의 사람별 bbox/pose metadata를 이용하여 TCN, Temporal
Transformer, Actor-Pair Transformer를 같은 조건에서 비교합니다.

기본 실험 조건은 2초 관찰 + 이후 4초 label 평가입니다. RGB 영상은
사용하지 않습니다.

## 입력 CSV

사람 한 명의 한 프레임을 한 행으로 저장합니다. 필수 열은 다음과 같습니다.

    video_id,frame,person_id,label,split,bbox_x,bbox_y,bbox_w,bbox_h

- label: CHAD frame label, 정상 0 / 이상 1
- split: train, val, test
- 좌표는 가능하면 영상 폭·높이로 나눈 0~1 값
- pose 열은 선택 사항이며 이름이 pose_로 시작하면 자동 사용
- 예: pose_0_x, pose_0_y, pose_0_c

같은 영상에서 train/test가 갈라지지 않도록 공식 CHAD split을 CSV의 split
열에 먼저 기록해야 합니다.

## 설치

    cd chad_comparison
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt

CUDA 확인:

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

## 실행

    python chad_transformer_comparison.py \
      --csv chad_person_frames.csv \
      --fps 30 \
      --sample-fps 10 \
      --observation-sec 2 \
      --future-sec 4 \
      --models tcn transformer actor_pair_transformer \
      --seeds 41 42 43 \
      --output-dir results_chad

RTX 3060 6 GB이면 batch-size 32, 12 GB이면 64부터 시작하는 것을 권장합니다.

결과 파일:

    results_chad/per_seed_results.csv
    results_chad/summary_results.csv
    results_chad/test_scores_*.csv

## 제안 방법과 결합

제안 Actor-Pair Episode Memory 방법의 validation/test episode 점수를 다음
열을 가진 CSV로 저장합니다.

    sample_id,score,split

실행 시 다음을 추가합니다.

    --external-score proposed=proposed_scores.csv

그러면 동일 sample과 label을 사용해 Proposed의 ROC-AUC, PR-AUC, F1도
결과표에 추가됩니다.

## 논문 보고 시 주의

- feature 평균과 표준편차는 train set에서만 계산됩니다.
- anomaly threshold는 validation set에서 F1이 최대가 되는 값으로 결정됩니다.
- test set으로 threshold나 normalization을 조정하지 않습니다.
- 동일 test sample을 모든 모델에 사용합니다.
- 결과는 3개 seed의 평균과 표준편차로 보고합니다.
- actor_pair_transformer는 같은 프레임의 가장 가까운 사람과 상대 특징을 사용합니다.
- actor/pair 정답 주석이 따로 있으면 그 주석을 label로 사용해 별도로 평가합니다.
