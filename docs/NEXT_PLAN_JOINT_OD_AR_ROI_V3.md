# Kế hoạch tiếp theo: V3 codec-native ROI/QP cho OD và Action Recognition

**Tên file chuẩn:** `docs/NEXT_PLAN_JOINT_OD_AR_ROI_V3.md`  
**Trạng thái:** đã kiểm chứng V2; V3 CRF+AQ đã triển khai, chờ F0 Kaggle
**Ngày cập nhật protocol:** 2026-09-20

File này là nguồn quyết định duy nhất cho vòng phát triển tiếp theo. Không đổi
ngưỡng thành công sau khi đã thấy kết quả.

## 1. Kết luận ngắn

- **OD:** giữ nguyên phương pháp đã đóng băng làm baseline. Nó đã cho BD-rate âm
  trên mọi screen đã chạy, nhưng H.265 và cross-evaluator vẫn còn khoảng cách lớn
  tới mức `-15%`.
- **AR V1:** đóng hoàn toàn vì làm giảm source Top-1 19--23 điểm phần trăm và mọi
  BD-rate đều dương.
- **AR V2 guarded:** an toàn hơn rõ rệt, nhưng không đạt mục tiêu. Có `22/24` arm
  qua source/per-QP guardrail, song `0/24` arm đạt `BD-rate Top-1 <= -15%` đồng
  thời trên H.264 và H.265.
- **Quyết định:** không quét thêm `sigma`, blend, saliency fraction hoặc motion
  fraction trong họ blur/guard hiện tại. Hướng tiếp theo là **phân bổ QP theo ROI
  ngay trong codec**, dùng chung primitive cho OD và AR. Đây là thay đổi cơ chế,
  không phải tinh chỉnh thêm trên một frontier đã bão hòa.

## 2. Những gì đã được kiểm chứng

### 2.1 Tính toàn vẹn của sáu job AR V2

Sáu job sau đều ở trạng thái hoàn tất và có đủ artifact:

1. `wagur124705/preupd-ar-guard-context-r3d-v1`
2. `htran123456/preupd-ar-guard-context-mc3-v1`
3. `hoangminhhuy123/preupd-ar-guard-context-r2p1d-v1`
4. `shungg05/preupd-ar-guard-rate-r3d-v1`
5. `vtk269/preupd-ar-guard-rate-mc3-v1`
6. `nguyenhoanglan1232/preupd-ar-guard-rate-r2p1d-v1`

Các kiểm tra đã thực hiện:

- mỗi job có `probe_action_guarded.json`, `per_clip_records.npz`, log nội bộ,
  split JSON và gói `.tgz`;
- log nội bộ không có `Traceback`, `ERROR`, `Exception`, CUDA OOM hay timeout;
- cả sáu split có cùng SHA-256
  `1a7adb6ad3aac2fa3fc93767c9567e49441b35f7ad6375fe7e1949ef0507acb2`;
- thứ tự `clip_id` giống hệt giữa sáu job;
- mỗi NPZ có 177 array, mọi array dài 200, và có đúng 200 clip ID duy nhất;
- cấu hình chung đúng: `val`, 200 clip, 16 frame, kích thước 128,
  QP `30/35/40/45/50`, codec `h264/h265`, teacher `r3d_18`;
- evaluator đúng theo job: `r3d_18`, `mc3_18`, `r2plus1d_18`;
- anchor bpp giống hệt giữa sáu job; anchor Top-1 của cùng evaluator giống hệt
  giữa family context và family rate;
- BD-rate tính lại từ per-clip NPZ khớp JSON tới sai số số thực khoảng `1e-6`.

File `*.log` ở tầng artifact gốc có thể rỗng do cách Kaggle CLI đóng gói; log
thực sự trong `outputs/ar_guarded/ar_guarded.log` đầy đủ. Đây không phải lỗi chạy.

### 2.2 Kết quả tốt nhất nhưng vẫn an toàn của từng job

“An toàn” ở đây chỉ nghĩa là qua source gap và per-QP gap đã đăng ký trước;
không có arm nào qua mục tiêu BD-rate.

| Job | Arm an toàn tốt nhất theo worst-codec | H.264 BD-rate Top-1 | H.265 BD-rate Top-1 | Source gap |
|---|---|---:|---:|---:|
| context / R3D | `p65 m50 s2 a0.25 r0.97 t0.1` | -1.26% | -0.74% | 0.00 |
| context / MC3 | `p65 m50 s2 a0.40 r0.97 t0.1` | -1.30% | -0.95% | 0.00 |
| context / R2Plus1D | `p65 m50 s2 a0.25 r0.97 t0.1` | -4.86% | -0.06% | -0.01 |
| rate / R3D | `p50 m35 s3 a0.35 r0.95 t0.15` | -7.14% | -1.54% | 0.00 |
| rate / MC3 | `p65 m35 s3 a0.35 r0.95 t0.15` | -0.27% | -1.62% | 0.00 |
| rate / R2Plus1D | `p65 m35 s3 a0.55 r0.95 t0.15` | -2.55% | -1.69% | 0.00 |

Không có một exact arm nào chuyển thành nghiệm `-15%` trên nhiều backbone và cả
hai codec. Việc chọn “arm tốt nhất” trong bảng dùng chính validation data, nên
bảng này là bằng chứng khám phá, không phải xác nhận cuối.

### 2.3 Paired clip bootstrap

Đã bootstrap có hoàn lại 2.000 lần, seed `20260919`, trên arm an toàn tốt nhất
của từng job. CI dưới đây là percentile 95%; xác suất là tỷ lệ bootstrap có
`BD-rate <= -15%`.

| Job | Codec | Điểm | CI 95% | P(BD <= -15%) |
|---|---|---:|---:|---:|
| context / R3D | H.264 | -1.26% | [-7.57%, 5.09%] | 0.0000 |
| context / R3D | H.265 | -0.74% | [-4.81%, 3.94%] | 0.0000 |
| context / MC3 | H.264 | -1.30% | [-8.19%, 6.68%] | 0.0000 |
| context / MC3 | H.265 | -0.95% | [-5.68%, 3.77%] | 0.0000 |
| context / R2Plus1D | H.264 | -4.86% | [-9.23%, -0.26%] | 0.0000 |
| context / R2Plus1D | H.265 | -0.06% | [-3.94%, 4.05%] | 0.0000 |
| rate / R3D | H.264 | -7.14% | [-13.52%, -0.41%] | 0.0080 |
| rate / R3D | H.265 | -1.54% | [-7.65%, 4.43%] | 0.0005 |
| rate / MC3 | H.264 | -0.27% | [-6.44%, 6.01%] | 0.0000 |
| rate / MC3 | H.265 | -1.62% | [-6.36%, 2.99%] | 0.0000 |
| rate / R2Plus1D | H.264 | -2.55% | [-7.41%, 2.75%] | 0.0000 |
| rate / R2Plus1D | H.265 | -1.69% | [-5.89%, 2.88%] | 0.0000 |

Diễn giải: V2 có tín hiệu thật ở H.264 cho `rate/R3D` và `context/R2Plus1D`,
nhưng hiệu quả nhỏ, không chuyển sang H.265, và không đủ gần `-15%`. Bootstrap
này vẫn mang selection bias vì arm được chọn sau screen; nó chỉ củng cố quyết
định dừng family, không tạo claim xác nhận.

### 2.4 Đối chiếu lại OD

`D:\STUDY\LAB\wagur1\preprocessing_upgrade_10_OD` là cây source baseline, không
phải thư mục artifact. Bằng chứng chạy thực tế nằm trong
`D:\STUDY\LAB\proxy_v3\_kaggle_checks`. Bảy job tạo nên matrix OD đều có JSON,
per-image NPZ, log và `.tgz`; quét log không thấy lỗi runtime. Các số đọc lại từ
JSON khớp `docs/RESULTS_generalization_matrix.md`:

| Screen OD | H.264 BD-rate | H.265 BD-rate | Worst mAP gap |
|---|---:|---:|---:|
| Original held-out, n=500 | -13.145% | -8.017% | -0.00651 |
| New sample, n=500 | -15.412% | -7.959% | -0.00612 |
| FCOS cross-evaluator, n=200 | -7.119% | -7.298% | -0.02055 |
| RetinaNet cross-evaluator, n=200 | -7.291% | -5.122% | -0.01450 |
| Reversed analyzer/evaluator, n=200 | -8.240% | -3.148% | -0.01404 |

Vì vậy kết luận OD hợp lệ ở mức point estimate: mọi screen đều âm trên cả hai
codec và qua mAP guardrail. Tuy nhiên chỉ một H.264 screen chạm `-15%`, H.265
chưa chạm, và matrix chưa có paired CI hoàn chỉnh. Không được mô tả OD là đã
được xác nhận thống kê; frozen transform hiện tại là baseline mạnh để V3 phải
đánh bại trên cùng sample.

## 3. Vì sao chọn codec-native ROI/QP

V1 cho thấy phá pixel mạnh có đủ rate headroom nhưng phá action evidence. V2
cho thấy source guard giữ được accuracy, song vì hầu hết clip chỉ cho phép edit
nhẹ nên không còn đủ rate headroom. Cần tách hai mục tiêu:

- pixel đầu vào evaluator không bị làm mờ có chủ ý;
- codec được phép lượng tử hóa nền mạnh hơn và giữ vùng quan trọng tốt hơn.

FFmpeg hiện có `AV_FRAME_DATA_REGIONS_OF_INTEREST`; mã nguồn `libx264` và
`libx265` đều đọc ROI side-data và chuyển nó thành quantization offsets khi AQ
được bật. Filter `addroi` của FFmpeg cũng cho phép xâu chuỗi nhiều rectangle.
Nguồn kỹ thuật chính thức:

- [FFmpeg AVFrame ROI side-data](https://ffmpeg.org/doxygen/trunk/group__lavu__frame.html)
- [FFmpeg libx264 ROI implementation](https://www.ffmpeg.org/doxygen/trunk/libx264_8c_source.html)
- [FFmpeg libx265 ROI implementation](https://www.ffmpeg.org/doxygen/trunk/libx265_8c_source.html)
- [FFmpeg addroi filter](https://ffmpeg.org/ffmpeg-filters.html#addroi)

Điều này mới là bằng chứng rằng cơ chế tồn tại; chưa chứng minh binary Kaggle cụ
thể thực thi đúng. Vì vậy V3 bắt đầu bằng feasibility gate, không chạy thẳng một
screen lớn.

## 4. Cơ chế V3

### 4.1 Primitive dùng chung

Tạo một `TaskROIMap` rồi chuyển thành tối đa bốn rectangle, căn block codec:

- **OD:** box của detector tạo mask + halo 8 px; vùng box là ROI cần giữ;
- **AR:** hợp của gradient saliency và motion tube; lấy component/tile quan
  trọng nhất, làm mượt theo thời gian, rồi tạo một tube rectangle cố định cho
  toàn clip 16 frame;
- nếu sau block alignment diện tích ROI vượt 65% khung hình, giảm số tile theo
  importance thay vì mở rộng ROI vô hạn;
- ground-truth label không được dùng để tạo ROI.

Vì mỗi clip được encode bằng một lệnh riêng, tube cố định theo clip có thể đi qua
FFmpeg CLI `addroi`; chưa cần viết custom decoder hoặc gửi mask cho decoder.

### 4.2 Rate control và phân bổ QP

V3 là một family thí nghiệm mới dùng **CRF + AQ**, không tái sử dụng anchor CQP
của V1/V2. `libx264` vô hiệu AQ trong CQP nên ROI side-data có thể bị bỏ qua;
`libx265` không có cùng ràng buộc tuyệt đối, nhưng dùng chung CRF+AQ giúp protocol
hai codec nhất quán. Không so trực tiếp BD-rate V3/CRF với số V1/V2/CQP.

Trong mỗi arm:

1. thêm ROI rectangle trước với offset `0` hoặc `-2 QP`;
2. thêm rectangle toàn khung sau với background offset dương;
3. do region ưu tiên đứng trước, ROI giữ QP gốc/tốt hơn, còn phần không thuộc ROI
   nhận background offset;
4. bật AQ cho cả x264 và x265, chạy FFmpeg ở `loglevel=warning`, lưu stderr và
   hard-fail nếu log báo ROI bị bỏ qua/không hỗ trợ.

Arm được định nghĩa bằng **delta-QP yêu cầu**, không dùng trực tiếp một
`qoffset` chung vì x264 và x265 ánh xạ qoffset khác nhau. Adapter codec chuyển
`delta_qp` sang rational qoffset rồi ghi cả giá trị yêu cầu lẫn rational truyền
cho encoder vào JSON. AQ có thể làm QP block thực tế khác yêu cầu, nên báo cáo
không được gọi offset này là actual per-block delta-QP.

### 4.3 Accounting công bằng

- đếm toàn bộ byte của bitstream thật; không trừ header hoặc signaling;
- anchor và mọi control/arm dùng cùng codec, CRF grid, AQ mode/strength, preset,
  GOP và pixel format;
- không truyền file mask/ROI riêng cho decoder;
- báo riêng thời gian tạo map, thời gian encode và peak GPU memory;
- thêm **sham-ROI control** đi qua đúng ROI/filter path nhưng mọi offset bằng 0;
- thêm **global-QP control** có cùng background delta nhưng không bảo vệ ROI để
  chứng minh lợi ích đến từ phân bổ không gian, không chỉ từ đổi QP toàn cục.

## 5. Thiết kế thí nghiệm đã khóa

### Phase F0 — kiểm tra codec, chưa đánh giá mô hình

Chỉ dùng 20 clip AR và 20 ảnh OD trên một tài khoản.

Điều kiện PASS:

1. `ffmpeg -filters` có `addroi`, và cả `libx264`/`libx265` hiện diện;
2. AQ bật; log không chứa “skipping ROI”;
3. cùng input/CRF nhưng đổi ROI làm thay đổi bitstream size và reconstruction;
4. bitstream decode bằng ffmpeg chuẩn, không cần side file;
5. ROI `qoffset=0` + background offset chỉ làm thay đổi vùng ngoài ROI theo
   hướng dự kiến ở kiểm tra block-level.

Nếu một codec không qua F0 thì dừng, không phát sáu job. Khi đó mới cân nhắc một
bridge libavcodec nhỏ; không giả định CLI đã hỗ trợ.

### Phase D1 — development screen trên validation

Common setting AR: cùng hash split V2, 200 clip, 16 frame, stride 2, size 128.
F0 kiểm tra dải CRF theo bitrate/reconstruction không dùng nhãn; grid D1 được
khóa là CRF `24/30/36/42/48`. Evaluator gồm R3D-18, MC3-18 và R2Plus1D-18.

Sáu arm cố định:

| Arm | ROI importance budget | ROI delta-QP | Background delta-QP |
|---|---:|---:|---:|
| `roi50_bg3` | 50% | 0 | +3 |
| `roi50_bg6` | 50% | 0 | +6 |
| `roi65_bg6` | 65% | 0 | +6 |
| `roi65_bg9` | 65% | 0 | +9 |
| `roi50m2_bg6` | 50% | -2 | +6 |
| `roi65m2_bg9` | 65% | -2 | +9 |

Mỗi arm có global-QP control tương ứng. Không thêm arm sau khi thấy kết quả D1.

OD dùng cùng encoder primitive và cùng sáu arm trên 200 ảnh validation mới,
đồng thời chạy lại frozen `halo8 + POST sigma=1` trên đúng ảnh đó để so sánh
paired. Detector tạo map và evaluator tiếp tục tách biệt như thiết kế OD hiện
tại.

### Gate để đi tiếp

Một exact semantic arm AR chỉ được đi tiếp nếu đồng thời:

1. source gap bằng 0 theo thiết kế vì không sửa source pixel;
2. worst per-QP Top-1 gap `>= -0.05` trên cả hai codec;
3. Top-1 BD-rate `<= -15%` trên H.264 và H.265;
4. cùng arm đạt các điều kiện trên ở R3D-18 và ít nhất một held-out evaluator;
5. paired bootstrap 2.000 lần có upper CI `< 0%` trên cả hai codec;
6. tốt hơn global-QP control ở cùng task quality.

Nhánh OD đi tiếp nếu trên cùng ảnh nó tốt hơn frozen OD baseline ít nhất 2 điểm
phần trăm BD-rate ở cả hai codec, worst mAP gap `>= -0.05`, và không thua trên
held-out evaluator. Nếu AR đạt nhưng OD không đạt, giữ OD baseline cũ; không ép
một cơ chế chung bằng cách hạ gate.

Stop rule sớm: nếu sau D1 không arm nào đạt ít nhất `-8%` trên **cả hai codec**
ở R3D và một held-out evaluator, đóng ROI-QP family. Không mở grid dày hơn.

### Phase C1 — confirmatory, chỉ chạy nếu D1 PASS

- khóa đúng một arm trước khi đọc test;
- AR: 500 clip từ hash `test` chưa dùng, ba evaluator, hai codec;
- OD: 500 ảnh held-out chưa dùng cho tuning, analyzer/evaluator tách biệt;
- paired bootstrap 2.000 lần với seed ghi trước;
- báo point estimate, CI 95%, mọi per-QP gap, bitrate, latency và memory;
- không thay arm, threshold hay sample sau khi xem C1.

## 6. Phân tài khoản Kaggle, không chạy trùng

F0 chạy một job duy nhất trên `wagur124705`. Chỉ sau khi F0 PASS mới phát D1:

| Tài khoản | Block duy nhất |
|---|---|
| `wagur124705` | AR / H.264 / R3D-18 |
| `htran123456` | AR / H.264 / MC3-18 |
| `hoangminhhuy123` | AR / H.264 / R2Plus1D-18 |
| `shungg05` | AR / H.265 / R3D-18 |
| `vtk269` | AR / H.265 / MC3-18 |
| `nguyenhoanglan1232` | AR / H.265 / R2Plus1D-18 |

OD chạy sau AR D1 hoặc ghép vào job chỉ khi ước lượng runtime còn an toàn. Mỗi
tài khoản chỉ có một kernel V3 hoạt động; trước khi push phải kiểm tra trạng thái
để không tạo bản sao. Không bật watcher định kỳ. Pusher không đặt timeout cục
bộ; giới hạn runtime của chính Kaggle vẫn tồn tại và không thể xóa từ code.

## 7. File code cần tạo/sửa khi triển khai

| File | Việc cần làm |
|---|---|
| `src/codecs/roi.py` | build/validate chuỗi `addroi`, CRF+AQ, adapter delta-QP và kiểm tra log |
| `src/models/task_roi.py` | chuyển saliency-motion tube thành rectangle block-aligned có area cap |
| `ops/probe_joint_roi.py` | F0; anchor, sham, six arms, global controls, records, BD-rate và gate |
| `ops/push_roi_probe.py` | tạo Kaggle notebook/kernel, chia codec/evaluator, mặc định không local timeout |
| `tests/test_roi_codec.py` | identity, priority/overlap, block alignment, bit accounting, codec capability |
| `tests/test_task_roi.py` | area cap, temporal stability, không dùng ground-truth, deterministic map |
| `tests/test_roi_probe_cli.py` | schema JSON/NPZ, frozen grid và command generation |

Artifact bắt buộc:

- `outputs/roi_v3/probe_joint_roi.json`;
- `outputs/roi_v3/per_clip_records.npz`;
- `outputs/roi_v3/roi_bit_accounting.json`;
- `outputs/roi_v3/codec_capability.json`;
- `outputs/roi_v3/run.log`.

## 8. Thứ tự thực hiện

1. Implement F0, sham control và unit test codec capability.
2. Chạy local smoke trên 2 clip/2 ảnh bằng environment
   `D:\STUDY\AI\envs\ten_env`.
3. Commit/push GitHub một lần.
4. Push đúng một kernel F0 lên `wagur124705`, không timeout cục bộ.
5. Tải artifact F0 và kiểm tra năm điều kiện PASS.
6. Chỉ khi F0 PASS: push sáu block D1 theo bảng, không duplicate.
7. Gộp kết quả bằng `clip_id`, chạy paired bootstrap và áp gate đúng như mục 5.
8. Chỉ khi D1 PASS: khóa một arm và chạy C1; nếu FAIL thì chuyển sang learned
   rate-distortion preprocessor, không tiếp tục quét ROI.

## 9. Phương án dự phòng nếu V3 thất bại

Fallback duy nhất là preprocessor học được với differentiable codec proxy, loss
gồm task distillation từ nhiều evaluator + rate + source identity, rồi bắt buộc
đo lại bằng bitstream H.264/H.265 thật. Nhánh này tốn compute và có nguy cơ
overfit evaluator cao hơn nên chỉ mở sau khi ROI-QP bị phản chứng bởi stop rule.

Không quay lại hard blur, detector-only tube hoặc mở rộng grid V2: các hướng đó
đã có bằng chứng trực tiếp rằng hoặc phá action evidence, hoặc không đủ rate
headroom.
