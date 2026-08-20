[README.md](https://github.com/user-attachments/files/31258045/README.md)
# 고무패킹 ConvNeXt 검사 프로토타입

Hikrobot MV-CS050-10GC로 취득해 PC에 저장된 상단면 또는 측면 이미지 한 장을 받아, 6개 클래스 중 하나로 분류하고 OK/NG 결과를 JSON으로 내보내는 확장용 소스코드입니다. 각 사진은 다른 사진과 결합하거나 짝짓지 않고 독립적으로 처리합니다. 카메라 trigger·노출·GigE SDK 제어는 포함하지 않았고, 카메라 제어 프로그램과 이 프로젝트의 계약은 **이미지 파일명과 저장 폴더**로 분리했습니다.

## 현재 구현 범위

- 클래스: `good`, `shrinkage`, `thread_defect`, `incomplete_molding`, `burr`, `contamination`
- ImageNet 사전학습 ConvNeXt-Tiny 기반 단일 이미지 분류기
- 상단면과 측면 사진을 같은 신경망으로 각각 독립 처리하는 6-class head
- 항상 활성화되는 `severity` head와 클래스별 NG threshold
- 학습/검증, class imbalance weight, macro-F1, 클래스별 지표, confusion matrix
- 단일 건 추론
- 폴더 자동 감시, 완전쓰기 확인, 이미지 한 장 단위 처리
- SQLite 중복 처리 방지, 검사별 원자적 JSON 결과 저장

전체 흐름은 다음과 같습니다.

```text
Hikrobot 제어/저장 프로그램
        │  ID.jpg (상단면 또는 측면 한 장)
        ▼
runtime/inbox ── 파일 안정화 ──► ConvNeXt ──► 판정 정책
                                                           │
                         SQLite 상태 ◄──────────────────────┤
                         results/ID.json ◄──────────────────┘
```

## 가장 중요한 판정 전제

모델은 각 사진에서 결함 종류와 `0.0~1.0` severity를 함께 예측합니다. 클래스 확률은 결함 종류와 적용할 threshold를 선택하고, 최종 OK/NG는 severity로만 결정합니다.

```text
severity < 예측 결함 클래스 threshold  → OK
severity >= 예측 결함 클래스 threshold → NG + 해당 불량 종류
```

따라서 모든 라벨 행에 합의된 기준의 `severity`를 반드시 입력해야 합니다. YAML의 초기 `0.5` threshold는 실행 확인용일 뿐이며, validation 데이터로 클래스별 threshold를 보정한 뒤 `criteria_version`을 변경해야 합니다.

또한 현재 코드는 한 제품에 대표 결함이 하나라는 **single-label** 전제입니다. 두 종류 이상의 결함이 동시에 존재할 수 있고 모두 보고해야 한다면, 촬영 전에 라벨을 5개 독립 binary label로 바꾸고 multi-label sigmoid 구조로 수정해야 합니다.

## 1. 설치

Python 3.10 이상 환경에서 프로젝트 루트로 이동한 뒤 설치합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

NVIDIA GPU를 사용할 경우에는 시스템의 CUDA 환경에 맞는 PyTorch를 먼저 설치하는 편이 안전합니다. GPU가 없으면 `device: auto`가 자동으로 CPU를 사용합니다. 사전학습 weight는 첫 학습 때 내려받으므로 폐쇄망이면 미리 캐시에 넣거나 `model.pretrained: false`로 바꿔야 합니다.

## 2. 이미지 입력 형식

한 번의 학습 또는 추론에는 사진 한 장만 사용합니다. 상단면과 측면 사진을 모두 검사하려면 각 사진에 서로 다른 고유 ID를 부여합니다.

```text
P000001_TOP.jpg
P000001_SIDE.jpg
P000002_TOP.jpg
P000002_SIDE.jpg
```

예를 들어 `P000001_TOP`과 `P000001_SIDE`는 서로 독립된 `sample_id`이며 각각 별도의 결과 JSON을 생성합니다.

실시간 watcher는 Windows 파일명의 대소문자 충돌을 막기 위해 검사 ID를 대문자로 정규화합니다. 따라서 `p000001`과 `P000001`을 서로 다른 제품 ID로 사용하면 안 됩니다.

## 3. 라벨 manifest 채우기

`data/labels.csv`는 현재 header만 있습니다. 이미지 경로는 이 CSV가 있는 `data/` 폴더 기준 상대 경로 또는 절대 경로로 넣습니다.

```csv
sample_id,image_path,defect_type,severity,lot_id,capture_session,split
P000001_TOP,images/P000001_TOP.jpg,good,0.0,LOT01,S01,train
P000002_SIDE,images/P000002_SIDE.jpg,burr,0.8,LOT02,S02,val
```

각 열의 의미는 다음과 같습니다.

| 열 | 의미 |
|---|---|
| `sample_id` | 사진 한 장의 고유 검사 ID |
| `image_path` | 독립적으로 학습·검사할 이미지 경로 |
| `defect_type` | 6개 영문 class ID 중 하나 |
| `severity` | 필수 심각도 라벨 0.0~1.0 |
| `lot_id` | 생산 lot/촬영 묶음. split 누수 검사용 |
| `capture_session` | 촬영 세션/날짜/조명 설정 구분 |
| `split` | `train`, `val`, `test` 중 하나 |

`severity`는 모든 학습·검증 행에 있어야 하며 빈칸이면 manifest 검증이 중단됩니다.

같은 lot 또는 촬영 세션을 train과 val에 나누면 배경·조명 특징을 외워 성능이 부풀 수 있습니다. 기본 validator는 `lot_id`와 `capture_session` 중 하나라도 split 사이에 겹치면 중단하며, train/val에 6개 클래스가 모두 있는지도 확인합니다. 모든 클래스가 각 평가 split에 포함되도록 원본 제품 단위로 나누십시오.

라벨과 경로를 검사합니다.

```powershell
python scripts/validate_manifest.py --config configs/default.yaml
```

## 4. 학습

먼저 `configs/default.yaml`의 TODO와 경로를 확인합니다. 그 다음 실행합니다.

```powershell
python scripts/train.py --config configs/default.yaml
```

결과는 기본적으로 다음 위치에 생깁니다.

```text
runs/best.pt       validation severity MAE가 가장 낮았던 checkpoint
runs/last.pt       마지막 epoch checkpoint
runs/history.json  loss, macro-F1, 클래스별 지표, 오판 수치
```

기본 best 기준은 `train.checkpoint_metric: severity_mae`, `checkpoint_mode: min`입니다. false accept/reject는 클래스별 severity threshold를 적용해 계산됩니다. severity MAE뿐 아니라 `false_accept_rate`, `false_reject_rate`, 클래스별 recall도 함께 확인하십시오.

검사에서는 전체 accuracy만 보지 말고 다음 값을 함께 확인하십시오.

- 불량을 OK로 통과시킨 `false_accept_count/rate`
- 양품을 NG로 버린 `false_reject_count/rate`
- 6-class macro-F1과 결함별 recall
- confusion matrix
- 실제 검사 PC에서 batch=1 지연시간

런타임에 제품이 50개 미만이라는 조건과 학습 데이터 수는 별개입니다. 학습용 실물이 수십 개뿐이면 ConvNeXt가 배경이나 촬영 순서를 외우기 쉽습니다. 여러 lot·촬영 세션·위치 편차와 각 결함의 경미/심각 사례를 별도로 확보해야 합니다.

## 5. 이미지 한 건 시험

사진 한 장 입력:

```powershell
python scripts/predict.py `
  --config configs/default.yaml `
  --checkpoint runs/best.pt `
  --sample-id TEST001 `
  --image path\to\TEST001.jpg
```

## 6. 실시간 자동 감시

학습 checkpoint가 준비되면 아래 명령을 먼저 실행해 둡니다.

```powershell
python scripts/watch_folder.py `
  --config configs/default.yaml `
  --checkpoint runs/best.pt
```

카메라 제어 프로그램은 `runtime/inbox/`에 사진을 한 장씩 저장합니다. 파일이 쓰이는 도중 읽는 일을 줄이기 위해 다음 순서를 권장합니다.

1. 감시 정규식과 일치하지 않는 임시 이름(예: `P000001.tmp.jpg`)으로 저장
2. 저장과 close가 성공한 뒤 `os.replace()`로 `P000001.jpg`에 원자적 rename

임시 rename을 적용하지 못해도 watcher가 파일 크기와 수정 시간이 기본 3회 연속 같을 때만 읽고, Pillow decode까지 성공해야 추론합니다.

결과는 다음 두 곳에 기록됩니다.

- `runtime/results/<sample_id>.json`: PLC/상위 프로그램이 읽을 검사 결과
- `runtime/state/inspections.sqlite3`: 중복 이벤트와 재시작 시 재처리 방지 상태

성공한 원본 파일은 자동 삭제하거나 이동하지 않습니다. 카메라 프로그램이나 별도 보관 작업이 결과 JSON 생성 확인 후 archive해야 합니다. 동일 ID에 다른 이미지를 덮어쓰면 안전을 위해 충돌로 거부하므로 재촬영에는 새 ID를 발급하십시오. 새 모델로 과거 inbox 전체를 다시 평가하려면 별도 results/state 경로를 사용해야 이전 결과를 보존할 수 있습니다.

현재 폴더의 파일만 처리하고 종료하려면 다음을 사용합니다.

```powershell
python scripts/watch_folder.py --config configs/default.yaml --checkpoint runs/best.pt --once
```

`--once`는 안정화된 항목을 한 번 처리하는 점검용이며 장시간 backoff 재시도까지 기다리지 않습니다. 자동 재시도가 필요하면 상시 감시 모드로 실행하고 `retry_errors`, `retry_backoff_s`, `max_retry_attempts`를 설정하십시오.

## 7. 결과 JSON 예시

```json
{
  "sample_id": "P000123",
  "model": {
    "architecture": "convnext_tiny",
    "checkpoint_sha256_prefix": "a1b2c3d4e5f6"
  },
  "probabilities": {
    "good": 0.02,
    "shrinkage": 0.03,
    "thread_defect": 0.04,
    "incomplete_molding": 0.01,
    "burr": 0.88,
    "contamination": 0.02
  },
  "severity": 0.81,
  "decision": {
    "status": "NG",
    "defect_type": "burr",
    "defect_name_ko": "burr 불량",
    "observed_class": "burr",
    "confidence": 0.88,
    "defect_confidence": 0.88,
    "applied_rule": "class_specific_severity",
    "applied_threshold": 0.5,
    "criteria_version": "TODO-v0",
    "provisional": true
  }
}
```

`provisional: true`는 `criteria_version`이 아직 `TODO`여서 threshold가 임시라는 뜻입니다. 실제 기준을 확정할 때 `criteria_version`도 함께 변경해 결과 추적이 가능하게 하십시오.

## 8. 심각도/NG 기준을 확정하는 방법

객관적인 라벨링 기준으로 모든 항목의 `severity`를 채운 뒤 validation 결과로 설정을 바꿉니다.

```yaml
decision:
  severity_ng_thresholds:
    shrinkage: 0.65       # 모두 validation에서 별도 결정
    thread_defect: 0.60
    incomplete_molding: 0.55
    burr: 0.70
    contamination: 0.60
  criteria_version: CRITERIA-2026-02
```

이 숫자는 예시일 뿐 실제 기준이 아닙니다. test set을 threshold 선택에 사용하지 마십시오.

## 9. 촬영 시 확인할 사항

- 패킹 외곽의 burr가 crop되지 않도록 각 사진에 전체 형상을 포함
- 노출, gain, white balance, 조명 위치와 색온도를 고정하고 기록
- 이염 검출이 있으므로 Bayer 변환 및 RGB/BGR 순서를 고정
- 결함별로 몰아서 촬영하지 말고 class 순서를 섞어 시간/배경 shortcut 방지
- 같은 실물에서 촬영한 상단면·측면·연속 frame이 다른 split으로 나뉘지 않게 관리
- 초점 불량이나 payload 손상이 있는 사진은 억지 추론하지 않고 재촬영

현재 전처리는 center crop 대신 원본 전체를 정사각 padding한 후 resize합니다. 작은 burr가 224에서 사라진다면 먼저 광학 배율과 ROI를 개선하고, 그 다음 `image_size` 320/384를 검증하십시오.

## 10. 테스트

```powershell
python -m pytest
```

포함된 테스트는 class/판정 경계값, 이미지 padding, 파일 안정화 검사를 확인합니다. 실제 데이터가 생긴 뒤에는 알려진 이미지→예상 JSON, checkpoint save/load 동일성, 깨진 JPEG, 동일 이벤트 중복, 장시간 지연시간 테스트를 추가하십시오.

## 데이터 확보 후 반드시 채울 TODO

- `data/labels.csv`의 이미지 경로와 6-class 라벨
- 한 제품에 복수 결함이 가능한지 여부 및 대표 결함 선정 규칙
- 모든 사진의 객관적 `severity` 라벨
- `input.image_size`
- validation 기반 클래스별 severity NG threshold
- `decision.criteria_version`
- `inference.checkpoint`
- 실제 카메라 저장 폴더와 파일명 정규식
- 실제 검사 PC의 CPU/GPU latency와 처리 여유시간
