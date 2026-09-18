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
