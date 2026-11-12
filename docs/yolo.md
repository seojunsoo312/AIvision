# yolo

작업대 위 브라켓을 찾아 로봇 기준 6D 자세 (위치 + 방향) 를 구하는 폴더입니다. **6D 자세는 두 가지 방식, ① CAD 정합과 ② 키포인트로 구합니다** (그림: `docs/diagrams/pose/pose.html`). 팔을 움직이는 코드와 집기 숫자는 여기 없습니다.

자세 이름은 `config.py`가 정본입니다. 0 세우기, 1 눕히기, 2 비스듬히.

## 자세를 구하는 두 가지 방식

카메라 창 (`pose/roi_cloud.py`) 하나에서 두 방식 중 하나를 고릅니다. 실행할 때 `--pose`, 창에서는 `k` 로 바꿉니다.

| | ① CAD 정합 (`--pose cad`, 기본) | ② 키포인트 (`--pose kpt`) |
|---|---|---|
| 쓰는 영상 | 컬러 + 뎁스 | 컬러만 |
| 모델 | 박스 모델 `weights/box_best.pt` (학생이 5차시에 학습) | 키포인트 모델 `weights/kpt_best.pt` (학생이 6차시에 학습, 출발점은 기본 모델 `weights/kpt_base.pt`) |
| 라벨 도구 | `train/label.py` (박스 두 점 + 자세 종류, 모두 손으로) | `train/kpt_label.py` (점 10 개. 처음 몇 장은 손으로, `m` 이면 자동 찍기) |
| 자세 구하는 법 | 박스 안 뎁스 점군 → 책상 빼기 → CAD 를 점군에 맞춤 (책상 위 세 자세로 x · y · 각도 → ICP) | 키포인트 → 책상 붙이기 (세 자세 중 하나로, 손눈의 책상 높이로 깊이) |
| 여러 브라켓 | 모두 맞춤. 맨 위 하나가 펜던트 집기 대상 | 모두 맞춤. 펜던트에는 CAD 쪽이 고른 것과 같은 브라켓 |

두 방식 모두 같은 CAD (`cad/`) 좌표계의 자세를 내므로 뒷부분은 하나입니다.

- **검사 `i`**: 자세 → 테두리 자동 맞춤 → 흰/검 → 흰 브라켓 마커 양/불 (`pose/sim_defect_check.py` 의 `judge`). 라벨이 필요 없는 규칙입니다.
- **펜던트 「물체 위치 불러오기」 · `load_pick_targets()`**: 지금 고른 방식의 자세를 받습니다.

놓는 조건 (두 방식 공통):

- 브라켓이 **책상 위에** 세 자세 중 하나로 놓여 있어야 합니다. 다른 브라켓 위에 얹히거나 기댄 것은 자세가 틀립니다 (시뮬 더미 시험에서 0~2%, `vision/sim/RESULTS.md` 「쌓인 더미」).
- 브라켓끼리 **3 cm 이상 떨어뜨려** 놓습니다. 붙어 있으면 서로 가리고 박스 안에 옆 브라켓 점이 섞여 자주 틀립니다.

## 실행

프로젝트 폴더에서 실행합니다. 카메라를 여는 명령은 펜던트와 독립적으로 켜집니다.

① CAD 정합용 박스 모델 (수업에서 학생이 합니다):

```bash
python yolo/train/capture.py                 # 사진 저장. SPACE=저장  q=종료
python yolo/train/label.py                   # 0/1/2 로 자세, 대각 두 점으로 박스
python yolo/train/prepare.py                 # 학습/검증 목록
python yolo/train/train.py                   # 학습 → weights/box_best.pt. 기본 모델 box_base.pt + 섞음 세트, 15 에폭
python yolo/train/detect.py                  # 실시간 박스
python yolo/train/box_base.py --make-data --make-eval --train   # (강사, 노트북) 박스 기본 모델 만들기
```

`train.py` 는 박스 기본 모델 `weights/box_base.pt` 가 있으면 키포인트 학습과 같은 방법으로 합니다: 학생 사진 10번 반복 + 박스 섞음 세트 (`datasets/box_base/`) 300장, 15 에폭, 마지막 에폭 모델 (사진 20장, 노트북 RTX 약 5분).
기본 모델은 `box_base.py` 가 학생 자리 · 젯슨 실사진 68장 (키포인트 라벨을 박스 + 자세 종류로 바꿈) 과 시뮬 440장으로 만듭니다 (학습에 안 쓴 실사진 25 + 시뮬 100 에서 mAP50 0.99).
노트북 시험 (학생 사진 14장, 학습에 안 쓴 실사진 8장 · 시뮬 100장, mAP50):

| 학습 방법 | 실사진 | 시뮬 |
|---|---|---|
| 기본 모델 + 10번 반복 + 섞음 (기본값) | 0.995 | 0.995 |
| 기본 모델 + 10번 반복, 섞음 없이 | **0.884** (원래 실력을 잊음) | 0.991 |
| 기본 모델에서 반복 · 섞음 없이 가볍게 (`--no-mix`) | 0.995 | 0.995 (새로 배우는 것도 적음) |
| 기본 모델 없이 처음부터 (`--no-base`) | **0.47** | **0.44** |

반복은 학생 사진을 충분히 배우게, 섞음은 그동안 원래 실력을 잊지 않게 합니다. 키포인트도 같았습니다 (섞음 없이 10번 반복하면 시뮬 150 → 118, 반복 없이 가볍게 하면 유지되지만 위치 오차가 거의 그대로). 수업은 기본 모델이 본 적 없는 새 책상에서 하므로 실제로 배워야 해서 반복 + 섞음을 씁니다.

② 키포인트 라벨과 학습 (같은 사진을 써도 됩니다. 키 설명은 파일 맨 위):

```bash
python yolo/train/kpt_label.py                                          # 5차시 사진 → raw/kpt_labels. 처음엔 직접 찍기, m = 자동 찍기
python yolo/train/kpt_label.py --images 폴더 --labels 폴더 --auto      # 자동 찍기를 켜고 시작
python yolo/train/kpt_train.py                                          # 라벨한 사진 + 기본 섞음 세트로 미세조정 → weights/kpt_best.pt
python yolo/train/kpt_train.py --make-base                              # (강사, 노트북) 기본 섞음 세트 datasets/kpt_base 만들기
```

`kpt_train.py` 는 기본 모델 (`weights/kpt_base.pt`, 노트북의 v4n_ft2 를 복사한 것) 에서 시작해 학생 사진을 10번씩 반복하고, 기본 섞음 세트 (실사진 72 + 시뮬 400 중 300장) 를 섞어 15 에폭 학습합니다 (노트북 RTX 약 4분).
섞는 이유: 학생 사진 20장만으로 학습하면 원래 잘 하던 것을 잊습니다. 결과는 마지막 에폭 모델을 씁니다 (검증 세트 대부분이 섞음 세트라 "가장 좋은" 모델로 첫 에폭이 뽑히기 때문).
학습이 끝나면 `weights/kpt_best.pt` 가 생깁니다. `reset_class.py` 가 지웁니다.

- **카메라 창 (`roi_cloud.py`) 은 학생이 학습한 모델로만 켜집니다** (수업 순서에 맞춤). `weights/box_best.pt` (박스) 가 없으면 창이 안 켜지고, `weights/kpt_best.pt` 가 없으면 `k` · `--pose kpt` 를 못 씁니다. 기본 모델로 대신 켜지 않습니다.
- 라벨 도구의 자동 찍기 (`m`) 는 학습 전에 쓰므로 `kpt_best.pt` 가 없으면 기본 모델 (`kpt_base.pt`) 을 씁니다.
노트북 시험 (학생 자리 사진 20장): 학습에 안 쓴 시뮬 장면 150개 중 149개 성공 (기본 모델 150) — 원래 실력을 잊지 않음.

키포인트 라벨은 수업 사진 (`datasets/raw/images`) 이면 `datasets/raw/kpt_labels/` 에, 다른 사진이면 사진 폴더 옆 `labels/` 에 저장합니다 (`--labels` 로 바꿈). 박스 라벨이 있는 폴더면 덮어쓰지 않게 멈춥니다.

카메라 창:

```bash
python yolo/pose/roi_cloud.py                # ① CAD 정합 (기본)
python yolo/pose/roi_cloud.py --pose kpt     # ② 키포인트
python yolo/pose/roi_cloud.py --replay vision/sim/out/scenes   # 카메라 없이 저장한 장면으로 (n / p 넘기기)
# 학습 전 노트북에서 기본 모델로 보려면: --weights yolo/weights/box_base.pt --kpt-weights yolo/weights/kpt_base.pt
python -m yolo.pose.kpt_live --images 폴더 --calib 폴더          # 사진 폴더로 키포인트 자세 + c 검사
```

| 키 | 하는 일 |
|---|---|
| `k` | 자세 방식 바꾸기 (CAD 정합 ↔ 키포인트). 키포인트 가중치가 있어야 함 |
| `i` | 검사: 지금 방식의 자세로 흰/검 · 흰 브라켓 마커 양/불. 화면을 멈추고 터미널에 표 (펜던트에 넣을 xyz · RPY 포함). 아무 키나 → 다시 |
| `c` | CAD 처음부터 다시 등록 |
| `r` | 책상 평면 다시 찾기 |
| `v` | 점군 보기 (Open3D). 보이는 브라켓 점군 전부 + 정합된 CAD 점: 맨 위 (집을 것) 는 빨강, 나머지는 진하거나 연한 빨강. 정합에 실패한 브라켓은 점군만 |
| `s` | ply 저장. 리플레이용 장면 `runs/roi/scenes/scene_시각/` (rgb · 뎁스 · 캘리브) 도 같이 |
| `n` / `p` | (리플레이) 다음 / 이전 장면 |
| `q` | 종료 |

화면 윤곽 색 (검사 `i`): 초록 양품 · 빨강 불량 · 보라 검사 안 함 (검은 브라켓) · 주황 브라켓 아님 (윤곽이 사진 테두리와 안 맞음).

키포인트 방식은 펜던트 요청에 테두리 자동 맞춤으로 다듬은 자세를 바로 보냅니다. `load_all_pick_targets()` (모두 요청) 에는 지금 방식의 브라켓 전부를 `i` 검사와 같은 판정까지 해서 가까운 순으로 보냅니다. 키포인트 방식일 때는 CAD 자동 등록을 쉽니다.

CAD 정합도 브라켓을 여러 개 보여 줍니다. 맨 위 하나는 예전처럼 따로 등록 · 추적하고 (펜던트 집기용), 나머지는 다른 작업 스레드가 차례로 맞춥니다 (화면 윤곽 · 축, `i` 검사). 그 자리에서 `--cad-move-mm` 안이면 다시 계산하지 않고, 맞추기에 실패한 브라켓도 움직이기 전까지는 다시 해 보지 않습니다.

`roi_cloud.py`는 CAD 맞춤과 로봇 기준 좌표가 처음부터 켜져 있습니다. 끄려면 `--no-cad`, `--no-base`입니다.

| 인자 | 기본 | 하는 일 |
|---|---|---|
| `--weights` | `yolo/weights/box_best.pt` | 다른 가중치 파일 |
| `--conf` | 0.5 | 이 점수 아래 박스는 버림 |
| `--plane-mm` | 3 | 책상 평면에서 이 높이(mm) 이하 점을 뺌 |
| `--pose` | `cad` | 자세 방식. `cad` 뎁스 점군 + CAD 정합, `kpt` 키포인트 (손눈 캘리브와 키포인트 가중치 필요) |
| `--kpt-weights` | `weights/kpt_best.pt` | 키포인트 모델. 없으면 키포인트 방식을 못 씁니다 (기본 모델로 대신하지 않음) |
| `--replay` | 없음 | 카메라 대신 장면 폴더 (`rgb.png` · `depth_mm.npy` · `camera.json`, 시뮬 장면이나 `s` 로 저장한 것) |
| `--replay-noise` | 꺼짐 | 리플레이 뎁스에 실물 같은 노이즈 (시뮬 장면용) |
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
| `train/kpt_train.py` | 키포인트 모델 미세조정 (학생 사진 + 기본 섞음 세트 → `weights/kpt_best.pt`). `--make-base` 는 노트북에서 섞음 세트를 만듭니다. |
| `train/kpt_label.py` | 키포인트 라벨 (YOLO-pose, `sim-pose` 브랜치 실험). 처음에는 직접 찍고, `m` 을 누르면 자동 찍기가 켜져 모델이 점을 찍고 테두리 자동 맞춤까지 해 둡니다 (`--auto` 면 켜고 시작). 사람은 보이는 점만 고칩니다. 8 · 9번(안쪽 접힌 곳)은 표시가 없어 자세에서 자동으로 채웁니다. `f` 테두리 자동 맞춤은 지금 자세에서 출발해 CAD 윤곽선이 사진 테두리선에 붙도록 위치 · 각도를 다듬습니다(흰 · 검은 브라켓 모두). `k` 는 같은 자리에서 다른 자세 종류(세우기 · 눕히기 · 비스듬히)로 바꿔 맞춥니다. `g` 책상 붙이기(기본 켜짐)는 세우기 · 눕히기 · 비스듬히 중 하나로 보고 x · y · 각도만 맞춰서 보이는 점 2개로도 됩니다. 저장은 맞춘 자세에서 CAD 로 다시 투영한 자리이고, 번호는 대칭 상관없이 찍어도 시뮬 라벨 규칙으로 다시 매깁니다. 카메라는 사진 폴더 옆 `calib/`(intrinsics.json, eye_to_hand.json)을 먼저 읽고, `split.json` 이 있으면 `--split train/test` 로 고릅니다. |
| `pose/kpt_live.py` | (실험, `sim-pose`) 사진 폴더로 키포인트 자세 · `c` 검사를 보는 창. 카메라로 부르면 `roi_cloud.py --pose kpt` 를 엽니다. |
| `pose/pose_methods.py` | 카메라 창의 자세 방식 공통 부분: 키포인트 자세 작업 스레드, 검사 (`judge_all`), 펜던트 숫자, 리플레이 장면 읽기 · 저장. |
| `pose/sim_defect_check.py` | (실험) 흰 브라켓 마커 불량 규칙 (CAD 윤곽 안, 구멍 자리 빼고 어두운 픽셀 수). `judge()` 는 자세 하나 (어느 방식이든) 를 받아 판정합니다. `--real 폴더` 로 사진 판정, 없으면 합성 시험. |
| `train/boxes.py` | 박스를 라벨 txt(`class cx cy w h`)로 저장합니다. |
| `train/prepare.py` | 사진을 복사하지 않고 train/val 목록만 만듭니다. |
| `train/train.py` | 박스 모델 학습. 기본 모델 `weights/box_base.pt` 에서 시작 (없으면 YOLO11n 에서 처음부터). 결과는 `weights/box_best.pt`입니다. |
| `train/box_base.py` | (강사, 노트북) 박스 기본 모델 `weights/box_base.pt` 와 그 학습 데이터 겸 섞음 세트 `datasets/box_base/` 만들기. |
| `train/detect.py` | 학습한 가중치로 실시간 박스를 띄웁니다. |
| `pose/roi_cloud.py` | 카메라 루프. 박스에서 점군을 만들고(`vision/depth_cloud.py`) CAD와 맞춰 로봇 기준 위치를 구합니다 (`--pose kpt` 면 키포인트로). 펜던트의 등록 요청은 `motion/register_link.py` 약속으로 받습니다. |
| `pose/instances.py` | 검출마다 점군을 모으고 가장 위에 있는 하나를 고릅니다. |
| `pose/register.py` | 그 점군과 `cad/` STL을 맞춰 위치와 자세를 만듭니다. |
| `pose/debug/local_plane.py` | 화면에서 윗면을 보는 확인용입니다. 로봇에 넘기는 자세는 아닙니다. |
| `pose/view/view_cad.py` | `cad/` STL 축을 보여 줍니다. |
| `pose/view/view_cloud.py` | 저장한 ply를 보여 줍니다. |
| `datasets/` | 사진, 라벨, 학습 목록. |
| `weights/` | 가중치 네 개: `box_base.pt` · `kpt_base.pt` (기본 모델, 강사가 넣어 둠), `box_best.pt` · `kpt_best.pt` (학생이 학습, 초기화 때 지움). 예전 이름 `best.pt` · `bracket_base.pt` 만 있어도 켜집니다. 시작 가중치 `yolo11n.pt`는 없으면 처음 학습할 때 ultralytics가 받습니다. |
| `runs/` | 학습 로그, `roi/`의 ply와 등록 요청·상태 JSON. |

## 같이 쓰는 곳

- 카메라와 손눈 결과는 `vision/`입니다. 로봇 기준 위치는 `vision/calib_data/eye_to_hand.json`으로 만듭니다.
- 맞출 형상과 집기 숫자는 `cad/model.yaml`입니다.
- 펜던트 「물체 위치 불러오기」와 `load_pick_targets()`는 `roi_cloud.py`가 켜져 있어야 좌표를 받습니다. 주고받는 방법은 `motion/register_link.py`입니다. 요청 뒤 2초 안에 창이 받지 않거나, 받은 뒤 30초 안에 끝나지 않으면 「카메라 창이 응답하지 않습니다」가 납니다.
