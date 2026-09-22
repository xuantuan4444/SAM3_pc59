# Hướng dẫn đánh giá SAM3 (No-LLM) trên PASCAL-Context 59

Script `sam3_base_pc59_nollm.py` dùng để đánh giá mô hình SAM3 trên tập dữ liệu PASCAL-Context 59-class. (tự động lưu biểu đồ và hình ảnh trực tiếp thành file).

## 1. Cấu trúc thư mục

Tải toàn bộ thư mục này từ Google Drive về máy. Hãy đảm bảo cấu trúc thư mục được giữ nguyên đúng như sau trước khi chạy, hãy điều chỉnh đúng path nếu có điều chỉnh:

```text
sam3_base_pc59_nollm/
├── sam3_base_pc59_nollm.py         
├── data_voc2010/
│   ├── JPEGImages/                    
│   ├── SegmentationClassContext/      
│   ├── 59_labels.txt                  
│   └── pascal_context_val.txt        
└── weight_sam3/
    └── sam3.pt                        
```

download data VOC2010, các folder và file cần thiết tại:
```
https://drive.google.com/drive/folders/1dCwcG-Pyh_FwatlaPVYjBrcNs2yUwGWk?usp=sharing
```

download weight_sam3 tại:
```
https://drive.google.com/file/d/1FiUmJKX-CFvkKdcKecsdwikGJ2kVO7Eu/view?usp=sharing
```

## 2. Cài đặt môi trường

Cài đặt SAM3 trực tiếp từ repository chính thức:

```bash
pip install "git+https://github.com/facebookresearch/sam3.git"
```

Sau đó cài đặt các thư viện cần thiết khác:

```
pip install -r requirements.txt
```


# GenMask-SAM3 — Pipeline PASCAL-Context 59 (PC59)

Tinh chỉnh có mục tiêu (targeted class refinement) cho phân đoạn ngữ nghĩa SAM3 trên
PASCAL-Context 59 (59 lớp), thuộc nghiên cứu ablation GenMask-SAM3 (cùng với Cityscapes và
Pascal VOC). README này chỉ nói về nhánh PC59: cần file gì, cấu trúc thư mục ra sao, chạy
pipeline thế nào, và kết quả trả về là gì.

## Trạng thái hiện tại của dự án

Đã hoàn thành sẵn:
- ✅ `sam3_base_pc59_nollm.py` (SAM3 baseline, no-LLM, đủ 59 lớp) — đã chạy, có kết quả
- ✅ `target_classes.json` (danh sách lớp mục tiêu, chọn tự động) — đã có
- ✅ `data_pc59/adjust_prompt_pc59.json` (prompt ensemble sinh bởi LLM) — đã có

**Còn lại cần chạy:** SAM3 baseline (arm LLM) → get_coarse (cả 2 nhánh) → train (4 model) →
val (4 lần đánh giá) — tất cả gói gọn trong **1 lệnh duy nhất** nhờ orchestrator đã tự động
nhận diện phần nào xong rồi để bỏ qua (xem mục "Cách chạy" bên dưới).

## Sơ đồ tổng quan pipeline

```
SAM3 baseline (no-LLM)                     [ĐÃ XONG]
      |
      v
Chọn lớp mục tiêu tự động (BIC-gated: GMM hoặc ngân sách lỗi tích lũy)   [ĐÃ XONG]
      |
      +---------------------------------------+
      | nhánh no-LLM                          | nhánh LLM
      v                                        v
get_coarse (no-LLM)                    adjust_prompt_pc59.json          [ĐÃ XONG]
      |                                        |
      +---- train UNet+ASPP                    v
      +---- train UNet+ASPP+DINOv2      SAM3 baseline (LLM)
      |                                        v
      +---- val (đánh giá hybrid) x2     get_coarse (LLM)
                                          train UNet+ASPP / +DINOv2
                                          val (đánh giá hybrid) x2
```

Mỗi bước là 1 file `.py` độc lập, chạy riêng được. `run_pc59_pipeline_full.py` điều phối tất
cả bằng 1 lệnh, tự bỏ qua bước nào đã có kết quả sẵn.

## Danh sách file cần thiết

Toàn bộ script đặt trực tiếp ở thư mục gốc (không đặt trong thư mục con).

| File | Vai trò |
|---|---|
| `run_pc59_pipeline_full.py` | **File chính, chạy file này.** Điều phối toàn bộ pipeline, cả 2 nhánh. |
| `select_refinement_classes.py` | Chọn lớp mục tiêu tự động (Priority score + BIC-gated GMM/cumulative). |
| `generate_adjust_prompt_pc59.py` | Chỉ dùng cho nhánh LLM — sinh `adjust_prompt_pc59.json` từ `contexts_class_pc59.json` qua Gemini. |
| `sam3_base_pc59_nollm.py` | SAM3 baseline, no-LLM (prompt tên lớp thuần), đủ 59 lớp. |
| `sam3_baseline_pc59_llm.py` | SAM3 baseline, prompt ensemble LLM, đủ 59 lớp. |
| `get_coarse_pc59_nollm.py` / `get_coarse_pc59_llm.py` | Tính sẵn cache mask thô SAM3 cho các lớp mục tiêu. |
| `train_unetaspp_pc59_nollm.py` / `train_unetaspp_pc59_llm.py` | Train mạng refine UNet+ASPP. |
| `train_unetasppdinov2_pc59_nollm.py` / `train_unetasppdinov2_pc59_llm.py` | Train UNet+ASPP+DINOv2. |
| `val_train_unetaspp_pc59_nollm.py` / `val_train_unetaspp_pc59_llm.py` | Đánh giá hybrid (SAM3 + UNet+ASPP), đủ 59 lớp. |
| `val_train_unetasppdinov2_pc59_nollm.py` / `val_train_unetasppdinov2_pc59_llm.py` | Đánh giá hybrid (SAM3 + UNet+ASPP+DINOv2), đủ 59 lớp. |

Chuẩn bị dữ liệu (chạy 1 lần, tách riêng, không thuộc orchestrator):

| File | Vai trò |
|---|---|
| `prepare-pc59-mat-to-png-final.ipynb` | Chuyển label map `.mat` gốc thành mask PNG `SegmentationClassContext/` (pixel 0=nền, 1-59=lớp, theo alphabet). Chạy 1 lần, ví dụ trên Kaggle. |

## Cấu trúc thư mục

```
sam3_pc59/                              <- thư mục gốc, chạy mọi lệnh từ đây
|
|-- run_pc59_pipeline_full.py               <- chạy file này
|-- select_refinement_classes.py
|-- generate_adjust_prompt_pc59.py
|-- sam3_base_pc59_nollm.py
|-- sam3_baseline_pc59_llm.py
|-- get_coarse_pc59_nollm.py
|-- get_coarse_pc59_llm.py
|-- train_unetaspp_pc59_nollm.py
|-- train_unetaspp_pc59_llm.py
|-- train_unetasppdinov2_pc59_nollm.py
|-- train_unetasppdinov2_pc59_llm.py
|-- val_train_unetaspp_pc59_nollm.py
|-- val_train_unetaspp_pc59_llm.py
|-- val_train_unetasppdinov2_pc59_nollm.py
|-- val_train_unetasppdinov2_pc59_llm.py
|
|-- data_pc59/
|   |-- JPEGImages/                         <- ảnh VOC2010, <id>.jpg
|   |-- SegmentationClassContext/           <- mask GT (output của prepare-pc59-mat-to-png-final.ipynb)
|   |-- pascal_context_train.txt
|   |-- pascal_context_val.txt
|   |-- contexts_class_pc59.json            <- tự viết tay (nhánh LLM)           [ĐÃ CÓ]
|   `-- adjust_prompt_pc59.json             <- sinh ra (nhánh LLM)               [ĐÃ CÓ]
|
`-- weight_sam3/
    `-- sam3.pt                             <- checkpoint SAM3
|-- target_classes.json                     [ĐÃ CÓ]

# Mọi thứ dưới đây do chạy pipeline tự sinh ra 
|-- coarse_cache_pc59_nollm/ , coarse_cache_pc59_nollm.zip
|-- coarse_cache_pc59/       , coarse_cache_pc59.zip
|-- dinov2_cache_pc59_nollm/
|-- dinov2_cache_pc59/
|-- weights_unetaspp_pc59_nollm_v1/
|-- weights_aspp_pc59_nollm_v1/
|-- weights_unetaspp_pc59_llm_v1/
|-- weights_aspp_pc59_llm_v1/
`-- results_pc59/
    |-- 00_baseline_nollm_*                 [ĐÃ CÓ từ sam3 baseline nollm từ trước]
    |-- 01_val_unetaspp_nollm_*
    |-- 02_val_unetasppdinov2_nollm_*
    |-- 03_baseline_llm_*
    |-- 04_val_unetaspp_llm_*
    `-- 05_val_unetasppdinov2_llm_*
```

**Lưu ý về `59_labels.txt`:** các bản pipeline cũ từng đọc thứ tự lớp từ
`data_pc59/59_labels.txt`, nhưng cách này có bug (thứ tự trong file đó không khớp thứ tự
alphabet mà mask GT thật sự dùng). Toàn bộ script liệt kê ở trên dùng danh sách 59 lớp hardcode
sẵn, đã verify khớp đúng — `59_labels.txt` giờ chỉ cần cho notebook chuẩn bị dữ liệu
(`prepare-pc59-mat-to-png-final.ipynb`), không liên quan gì đến pipeline chính nữa.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch torchvision      # đúng bản CUDA của máy bạn
pip install 'git+https://github.com/facebookresearch/sam3.git' --no-deps
pip install iopath ftfy portalocker
pip install pandas numpy matplotlib pillow tqdm opencv-python-headless
pip install "albumentations>=2.0" scipy      # các file train (boundary loss)
pip install kneed scikit-learn                # select_refinement_classes.py

# Chỉ cần cho nhánh LLM (generate_adjust_prompt_pc59.py):
pip install google-genai pydantic python-dotenv
export GEMINI_API_KEY="..."        # hoặc đặt trong file .env
```

DINOv2 (ViT-S/14) tự tải qua `torch.hub` ngay lần đầu chạy 1 trong các file `+DINOv2`
(cần có mạng ở đúng thời điểm đó).

## Cách chạy

### Trạng thái hiện tại — chỉ cần 1 lệnh

Vì `per_class_metrics.csv` (baseline no-LLM), `target_classes.json`, và
`data_pc59/adjust_prompt_pc59.json` **đã có sẵn**, chỉ cần chạy:

```bash
python run_pc59_pipeline_full.py
```

Orchestrator tự nhận diện các bước đã xong (in ra `[skip] ...`) và **chỉ chạy phần còn lại**:
archive kết quả baseline no-LLM đã có → SAM3 baseline (LLM) → get_coarse (cả 2 nhánh) →
train UNet+ASPP + UNet+ASPP+DINOv2 (cả 2 nhánh) → val (đánh giá hybrid, cả 2 nhánh) — không
cần thêm cờ gì, không tốn thời gian chạy lại phần đã xong.

Nếu muốn theo dõi/giới hạn phạm vi chạy, có thể thêm:

| Cờ | Tác dụng |
|---|---|
| `--skip-nollm` | Bỏ hẳn nhánh no-LLM (chỉ chạy phần còn lại của nhánh LLM) |
| `--skip-llm` | Bỏ hẳn nhánh LLM (chỉ chạy phần còn lại của nhánh no-LLM) |
| `--force-baseline` | Ép chạy lại cả 2 baseline dù đã có kết quả |
| `--force-select` | Ép chọn lại lớp mục tiêu dù `target_classes.json` đã có |
| `--force-coarse` | Ép chạy lại get_coarse dù cache đã có |
| `--force-train` | Ép train lại toàn bộ dù checkpoint đã có |

Ví dụ: chỉ cần hoàn thiện nốt nhánh LLM (nhánh no-LLM coi như xong hẳn):

```bash
python run_pc59_pipeline_full.py --skip-nollm
```

### Chạy lại / tiếp tục sau khi bị ngắt

Mọi bước đều idempotent — bị ngắt giữa chừng thì chạy lại đúng lệnh cũ sẽ tiếp tục đúng chỗ
dừng, không chạy lại từ đầu.

## Kết quả

Toàn bộ nằm trong `results_pc59/`, mỗi tiền tố ứng với 1 lần đánh giá:

| Tiền tố | Là gì |
|---|---|
| `00_baseline_nollm_*` | SAM3 baseline, no-LLM, đủ 59 lớp |
| `01_val_unetaspp_nollm_*` | Đánh giá hybrid (SAM3 + UNet+ASPP), no-LLM |
| `02_val_unetasppdinov2_nollm_*` | Đánh giá hybrid (SAM3 + UNet+ASPP+DINOv2), no-LLM |
| `03_baseline_llm_*` | SAM3 baseline, prompt LLM, đủ 59 lớp |
| `04_val_unetaspp_llm_*` | Đánh giá hybrid (SAM3 + UNet+ASPP), LLM |
| `05_val_unetasppdinov2_llm_*` | Đánh giá hybrid (SAM3 + UNet+ASPP+DINOv2), LLM |

Mỗi tiền tố có 4 file:

- `*_summary_metrics.csv` — Pixel Accuracy, mIoU (đủ 59 lớp), Mean Dice
- `*_per_class_metrics.csv` — mỗi dòng 1 lớp: `Class`, `Source` (`"SAM3 + UNet+ASPP (hybrid)"`
  cho lớp mục tiêu / `"SAM3 (baseline)"` cho lớp còn lại — chỉ có ở file val), `IoU`, `Dice`,
  `GT Pixels`, `Pred Pixels` — sắp theo IoU tăng dần
- `*_per_class_iou_bar_chart.png` — biểu đồ cột IoU từng lớp
- `*_class_visualizations/` — vài ảnh mẫu mỗi lớp (GT vs. dự đoán vs. overlay lỗi)

Muốn tính $mIoU_w$ (lớp mục tiêu) và $mIoU_s$ (lớp còn lại) cho bảng ablation — lọc cột
`Source` trong `per_class_metrics.csv` của file val, lấy trung bình `IoU` theo từng nhóm.

Ngoài ra, ở thư mục gốc còn có:

- `target_classes.json` — danh sách lớp mục tiêu đã chọn, toàn bộ bảng xếp hạng Priority, và
  bằng chứng BIC (`delta_bic`, `evidence_strength`) đứng sau quyết định dùng GMM hay ngân sách
  lỗi tích lũy
- `results_pc59/01_refinement_priority_curve.png` — biểu đồ chẩn đoán cho bước chọn lớp — nên
  xem qua trước khi tin tưởng hoàn toàn vào lựa chọn tự động