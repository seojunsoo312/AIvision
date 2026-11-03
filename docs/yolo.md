# yolo

작업대 위 브라켓을 찍어 자세 세 가지로 학습하고, 카메라 점군을 CAD와 맞춰 위치를 구하는 폴더입니다. 팔을 움직이는 코드와 집기 숫자는 여기 없습니다.

자세 이름은 `config.py`가 정본입니다. 0 세우기, 1 눕히기, 2 비스듬히.

## 실행

프로젝트 폴더에서 실행합니다. 카메라를 여는 명령은 펜던트와 독립적으로 켜집니다.

```bash
python yolo/train/capture.py                 # 사진 저장. SPACE=저장  q=종료
python yolo/train/label.py                   # 0/1/2 로 자세, 대각 두 점으로 박스
python yolo/train/kpt_label.py --images yolo/datasets/seat/images --split train   # (실험) 키포인트 라벨. 키 설명은 파일 맨 위
python -m yolo.pose.kpt_live                 # (실험) 실시간 키포인트 자세 + c 로 흰 브라켓 불량 판정. 가중치 v4n_ft2 필요 (없으면 v4n_ft, v4n)
python yolo/train/prepare.py                 # 학습/검증 목록
python yolo/train/train.py                   # 학습 → weights/best.pt
python yolo/train/detect.py                  # 실시간 박스
python yolo/pose/roi_cloud.py                # 점군을 CAD와 맞춰 로봇 기준 위치
```

`roi_cloud.py` 키는 `c` 등록, `r` 책상 평면 다시, `v` 점군 보기, `s` ply 저장, `q` 종료입니다.

`roi_cloud.py`는 CAD 맞춤과 로봇 기준 좌표가 처음부터 켜져 있습니다. 끄려면 `--no-cad`, `--no-base`입니다.

| 인자 | 기본 | 하는 일 |
|---|---|---|
| `--weights` | `yolo/weights/best.pt` | 다른 가중치 파일 |
| `--conf` | 0.5 | 이 점수 아래 박스는 버림 |
| `--plane-mm` | 3 | 책상 평면에서 이 높이(mm) 이하 점을 뺌 |
| `--no-plane` | 끔 | 책상 평면 제거를 하지 않음 |
| `--base` / `--no-base` | 켜짐 | 좌표를 로봇 베이스(mm)로. `eye_to_hand.json`을 읽음 |
| `--cad` / `--no-cad` | 켜짐 | 가장 위 물체 점군을 CAD와 맞춰 xyzrpy를 화면에 냄 |
| `--cad-every` | 2.5초 | 물체가 움직인 뒤 다시 맞추는 주기. 처음부터는 `c` |
| `--cad-move-mm` | 8 | 점군 중심이 이만큼 움직여야 다시 맞춤. 멈춰 있으면 축 고정 |
| `--cad-reset-mm` | 30 | 마지막 전체 등록 자리에서 이만큼 옮기면 추적 대신 처음부터 다시 등록. YOLO 자세가 바뀌거나 추적이 실패해도 처음부터 |
| `--box` / `--no-box` | 켜짐 | YOLO 박스를 화면에 그림. 고른(가장 위) 물체는 굵게 |
| `--no-noise-filter` | 끔 | Orbbec 노이즈 필터를 끔. 기본은 켜짐 |

## 파일

| 파일 | 하는 일 |
|---|---|
| `config.py` | 경로와 자세 이름. CAD는 `cad/model.py`로 읽습니다. |
| `data.yaml` | 학습이 읽는 클래스 목록. `prepare.py`가 이 PC 경로로 다시 씁니다. |
| `train/capture.py` | 학습용 사진을 `datasets/raw/images/`에 저장합니다. |
| `train/label.py` | 사진에 자세와 가로세로 박스를 그립니다. |
| `train/kpt_label.py` | 키포인트 라벨 (YOLO-pose, `sim-pose` 브랜치 실험). 시뮬 모델이 먼저 점을 찍고 사람은 보이는 점만 고칩니다. 8 · 9번(안쪽 접힌 곳)은 표시가 없어 자세에서 자동으로 채웁니다. `f` 테두리 자동 맞춤은 지금 자세에서 출발해 CAD 윤곽선이 사진 테두리선에 붙도록 위치 · 각도를 다듬습니다(흰 · 검은 브라켓 모두). `k` 는 같은 자리에서 다른 자세 종류(세우기 · 눕히기 · 비스듬히)로 바꿔 맞춥니다. `g` 책상 붙이기(기본 켜짐)는 세우기 · 눕히기 · 비스듬히 중 하나로 보고 x · y · 각도만 맞춰서 보이는 점 2개로도 됩니다. 저장은 맞춘 자세에서 CAD 로 다시 투영한 자리이고, 번호는 대칭 상관없이 찍어도 시뮬 라벨 규칙으로 다시 매깁니다. 카메라는 사진 폴더 옆 `calib/`(intrinsics.json, eye_to_hand.json)을 먼저 읽고, `split.json` 이 있으면 `--split train/test` 로 고릅니다. |
| `pose/kpt_live.py` | (실험, `sim-pose`) 카메라 → YOLO-pose 키포인트 → 책상 붙이기 자세를 실시간으로. `c` 정밀 분석 = 테두리 자동 맞춤 + 흰 브라켓 마커 불량 판정, 터미널에 펜던트 숫자(xyz · RPY). 뎁스는 안 쓰고 브라켓이 책상 위에 있다고 가정. 가중치 `vision/sim/out/pose_runs/v4n_ft2/weights/best.pt` (없으면 v4n_ft → v4n) 는 git 에 없어 따로 복사. |
| `pose/sim_defect_check.py` | (실험) 흰 브라켓 마커 불량 규칙 (CAD 윤곽 안, 구멍 자리 빼고 어두운 픽셀 수). `--real 폴더` 로 사진 판정, 없으면 합성 시험. |
| `train/boxes.py` | 박스를 라벨 txt(`class cx cy w h`)로 저장합니다. |
| `train/prepare.py` | 사진을 복사하지 않고 train/val 목록만 만듭니다. |
| `train/train.py` | YOLO11n으로 학습합니다. 결과는 `weights/best.pt`입니다. |
| `train/detect.py` | 학습한 가중치로 실시간 박스를 띄웁니다. |
| `pose/roi_cloud.py` | 카메라 루프. 박스에서 점군을 만들고(`vision/depth_cloud.py`) CAD와 맞춰 로봇 기준 위치를 구합니다. 펜던트의 등록 요청은 `motion/register_link.py` 약속으로 받습니다. |
| `pose/instances.py` | 검출마다 점군을 모으고 가장 위에 있는 하나를 고릅니다. |
| `pose/register.py` | 그 점군과 `cad/` STL을 맞춰 위치와 자세를 만듭니다. |
| `pose/debug/local_plane.py` | 화면에서 윗면을 보는 확인용입니다. 로봇에 넘기는 자세는 아닙니다. |
| `pose/view/view_cad.py` | `cad/` STL 축을 보여 줍니다. |
| `pose/view/view_cloud.py` | 저장한 ply를 보여 줍니다. |
| `datasets/` | 사진, 라벨, 학습 목록. |
| `weights/` | `best.pt`. 시작 가중치 `yolo11n.pt`는 없으면 처음 학습할 때 ultralytics가 받습니다. |
| `runs/` | 학습 로그, `roi/`의 ply와 등록 요청·상태 JSON. |

## 같이 쓰는 곳

- 카메라와 손눈 결과는 `vision/`입니다. 로봇 기준 위치는 `vision/calib_data/eye_to_hand.json`으로 만듭니다.
- 맞출 형상과 집기 숫자는 `cad/model.yaml`입니다.
- 펜던트 「물체 위치 불러오기」와 `load_pick_targets()`는 `roi_cloud.py`가 켜져 있어야 좌표를 받습니다. 주고받는 방법은 `motion/register_link.py`입니다. 요청 뒤 2초 안에 창이 받지 않거나, 받은 뒤 30초 안에 끝나지 않으면 「카메라 창이 응답하지 않습니다」가 납니다.
