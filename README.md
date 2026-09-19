# pre_updated — unified preprocessing for OD + Action Recognition

Line riêng cho task **Object Detection** của VCM: ảnh → preprocessing → codec đóng băng
(All-Intra) → decode → detector đóng băng → mAP. Đích là một con số BD-rate âm trên **trục mAP**
để điền ô Object Detection còn trống của báo cáo MPEG w21834.

Repo này phát triển nhánh OD thành một đường đo độc lập, đồng thời thêm primitive
**spatio-temporal importance tube** để cơ chế "giữ vật, giảm nền" dùng được cho cả
ảnh (`T=1`) và action recognition (`T>1`).

## Những sửa đổi chính của `pre_updated`

- evaluator COCO mAP nằm trực tiếp trong `evaluate.py`; OD không còn rơi nhầm vào
  evaluator classification;
- paired image bootstrap **có hoàn lại**, giữ multiplicity và xuất CI riêng cho
  từng arm;
- detector tạo mask mặc định là MobileNet-FPN, detector đánh giá là ResNet50-FPN
  (held-out analyzer; on-teacher chỉ được bật bằng cờ explicit);
- COCO dùng letterbox giữ aspect ratio thay vì squash ảnh;
- `loss.rho` của UP-VCM đã được nối thật vào objective;
- saliency gate AR tại eval dùng pseudo-label của source clip, không đọc ground-truth;
- `ImportanceTubeSuppress`: protected core exact-identity, feathered boundary,
  motion-gated temporal background stabilization;
- fixture COCO tự sinh và khai báo đầy đủ `Pillow`/`pycocotools`.

Lineage ban đầu tách ra sau khi **bốn hướng học-máy liên tiếp đo ra âm** trên chế độ ảnh:

| Hướng | Kết quả đo |
|---|---|
| Checkpoint AR zero-shot trên ảnh | vô hại cho mAP (ratio 1,02–1,03) nhưng **+2,29 % bit** (CI [+0,43,+4,86]), sandwich **+6,71 %** |
| Train PRE cho ảnh (proxy intra + loss detector) | **phá 25 % mAP** trước cả codec (ratio 0,75) |
| Đầu không gian (AR) | −5 pp so với kỷ lục |
| Temporal POST (AR) | −3 pp so với kỷ lục |

Đọc chung: cơ chế kiếm bit của thiết kế AR là **thời gian**; ở ảnh (T=1) nó vô dụng, còn lại chỉ
là các module *thêm/sửa* cấu trúc — thứ mAP không thưởng. Nên hướng đi ở đây đảo ngược:
**bỏ nền, giữ vật**, không dùng editor học được.

## Tài liệu

- **Spec:** [`docs/OD_DESIGN.md`](docs/OD_DESIGN.md) — bài toán, bằng chứng, thiết kế R0 (0 tham số)
  và R1 (gate học được ~1–2k tham số), interface, tiêu chí thành công/phản chứng.
- **Kế hoạch implement:** [`docs/superpowers/plans/2026-09-17-od-preprocessing.md`](docs/superpowers/plans/2026-09-17-od-preprocessing.md)
  — 6 task theo TDD, có decision gate giữa R0 và R1.

## Trạng thái

| Bước | Trạng thái |
|---|---|
| Core detection (data, analyzer, probe, pusher) | ✅ đã port, test xanh |
| R0 — mask từ detector + suppression nền | ✅ implement + test (`tests/test_mask_suppress.py`) |
| R0 full n=500, 5 QP, held-out detector | ✅ point estimate: H.264 −9.07%, H.265 −5.44%; chờ CI |
| R0.5 — dual-region PRE + high-QP Gaussian POST | ✅ code; screening design đã khóa |
| R0.5 halo8 + POST σ=1, held-out n=500 | ✅ QP45 chốt: H.264 −13.15%, H.265 −8.02%; gap PASS |
| R1 — gate học được | ⏸ chỉ mở nếu R0 dương |
| Importance-tube probe trên Kinetics | ❌ n=20: H.264 +9.80%, H.265 +8.57%; không scale detector-only tube |
| AR saliency V1, 3 evaluator × 2 codec | ❌ mọi BD-rate dương; best +25.10/+18.50%, source Top-1 giảm 19--23 pp |
| AR guarded saliency-motion V2 | 🚀 đã implement; source-confidence fallback, chờ screen 2 family × 3 evaluator |

## Chạy

```bash
pytest -q

# Một cell Kaggle chạy cả OD + AR với đúng hai dataset chuẩn
python ops/push_joint_probe.py --commit <sha> --account <acct> \
    --profile quick --slug pre-updated-joint-od-ar

# R0.5 screening: identity/mild-ROI PRE x identity/Gaussian POST
python ops/push_joint_probe.py --commit <sha> --account <acct> --skip-ar \
    --profile quick --n-od 100 --qps 30,35,40,45,50 --od-sigmas 4 \
    --od-roi-sigmas 0,1 --od-post-sigmas 0,1 --od-post-min-qp 45 \
    --bootstrap 0 --slug pre-updated-od-dualregion

# R0 trên Kaggle (eval-only, không train, ~20 phút)
python ops/push_detection_probe.py --commit <sha> --account <acct> \
    --script ops/probe_background_suppression.py \
    --extra-args "--sigmas 4,8,16 --score 0.5 --dilate 0.15" \
    --ckpt-dataset "" --n-images 500 --size 320 --bootstrap 1000 \
    --slug u9-probe-bgsuppress

# Probe chung OD→AR, không train (cần ffmpeg + index Kinetics)
python ops/probe_action_tubes.py --index data/index/kinetics_hash_split.json \
    --n-clips 200 --sigmas 4,8 --temporal-strengths 0,0.5

# AR V2: saliency + motion, mild blend, tự lùi về identity nếu teacher suy giảm
python ops/push_joint_probe.py --commit <sha> --account <acct> --skip-od \
    --profile confirmatory --ar-probe guarded --ar-split val --n-ar 200 \
    --ar-backbone mc3_18 --ar-guard-protect-fractions 0.65,0.8 \
    --ar-guard-motion-fractions 0.5 --ar-guard-max-blends 0.25,0.4 \
    --ar-guard-sigma 2 --ar-guard-retention 0.97 \
    --ar-guard-temporal-strength 0.1 \
    --slug preupd-ar-guard-context-mc3-v1

# OD checkpoint: evaluator tích hợp, COCO mAP + bootstrap CI
python evaluate.py --config configs/sandwich_coco_det.yaml \
    --ckpt outputs/sandwich_coco_det/checkpoints/preprocessor.pth \
    eval.bootstrap=1000
```

Cell copy/paste và cấu hình `quick`/`confirmatory`: [`docs/KAGGLE_JOINT_CELL.md`](docs/KAGGLE_JOINT_CELL.md).

## Ràng buộc (áp cho mọi thí nghiệm ở đây)

- Codec/bitstream/decoder **đóng băng**; chỉ can thiệp ở miền pixel trước encode và sau decode.
- Mọi số công bố dùng **held-out analyzer + paired bootstrap CI** và luật gap (`≥ −0.05` mọi QP, cả hai codec).
  Không có số on-teacher.
- Ảnh là đơn khung: `T=1`, codec intra-only (`codec.inter: false`).
- Box của torchvision là **xyxy**, COCO cần **xywh** — luôn đi qua `_coco_box`.
- Mọi split trong index phải **khác rỗng**: val rỗng sẽ âm thầm tắt model selection và early stopping.
- Run dài trên Kaggle: ghi diagnostics ra **file** trong output dir (cell bị cap 12h mất sạch stdout).
