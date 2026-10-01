# 1. Init Stage

Chạy các lệnh sau trong terminal / cmd:

```python
git clone https://github.com/sattarov/AnoDDAE.git
cd AnoDDAE
python -m venv venv

venv\Scripts\activate # for windows

source venv/bin/activate # for linux
```

Repo gốc không có `requirements.txt` , repo này đã cập nhật `requirements.txt`.

```python
pip install -r requirements.txt
```

---

# 2. Dataset

- Trong repo gốc: config mặc định chạy dataset `17_InternetAds.npz`
- Trong Reproduction Protocol.md: chạy 3 dataset `campaign`, `annthyroid`, `satimage-2`

---

**B1: Tạo thư mục.** Từ thư mục gốc repo AnoDDAE dùng `mkdir ad_bench.`Tên thư mục trùng với `data.path: ad_bench` trong config .

**B2: Tải 3 file về.** Cách nhẹ nhất là tải từng file từ trang GitHub bằng nút "Download raw file". Hoặc có thể type trong terminal:

```python
curl -L -o ad_bench/2_annthyroid.npz https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/2_annthyroid.npz

curl -L -o ad_bench/5_campaign.npz https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/5_campaign.npz

curl -L -o ad_bench/31_satimage-2.npz https://github.com/Minqi824/ADBench/raw/main/adbench/datasets/Classical/31_satimage-2.npz
```

Đừng clone cả ADBench, vì repo này nặng và chỉ cần 3 file.

Trang GitHub của ADBench: Minqi824/ADBench: Official Implement of "ADBench: Anomaly Detection Benchmark", NeurIPS 2022.

**Nếu trong folder ad_bench đã có 3 file .npz thì thành công.**

---

# 3. Benchmarks

Note: trong source code gốc **chỉ có mô hình DDAE**, không có DAE hay DDPM. Nếu reproduce hoàn chỉnh cân nhắc cài thêm từ 2 repo kia để so sánh.

LINK:

1. DAE: https://dl.acm.org/doi/10.1145/1390156.1390294
2. DDPM: https://arxiv.org/abs/2006.11239

---

# 4. Reproduction

**Mục đích**: chạy mô hình DDAE chỉ với các settings sau — giảm tối đa thay đổi so với bài báo gốc.

- Dataset: `campaign`, `annthyroid`, `satimage-2`
- Learning: `unsupervised` & `semi-supervised`.
- Seed: `[111, 222, 333, 444, 555]`
- num_timesteps: `[50, 100]`

Tổng cộng: 3 dataset × 2 learning settings × 5 seeds x 2 timesteps settings = **60 lượt** 

---

Lưu ý:

- Repo gốc KHÔNG bao gồm source code DAE và DDPM **→ Nếu muốn full reproduction phải tải riêng.**
- Repo gốc KHÔNG triển khai DDAE-C.
- **File chính dùng để train model là `run_all.py`. Đây là file tự thêm vào, không phải trong repo gốc.**

---

Trước tiên nên dùng `python run_all.py --dry-run` để kiểm tra tên dataset và 60 tổ hợp; chế độ này chưa kiểm tra nội dung dữ liệu và không training.

**Chạy file run_all.py để bắt đầu chạy model DAE với các settings trên.**

```python
python run_all.py
```

Khi chạy `python run_all.py`, script sẽ chạy lần lượt **60 experiments DDAE**, rồi lưu kết quả sau mỗi lượt. Nó không dùng `run.py` và không sửa các file source/config gốc.

Các dòng này sẽ in ra trên terminal:

!image.png

---

### 4.1 Chuẩn bị

Script sẽ:

- Đọc `src/config.yaml` làm cấu hình nền.
- Tìm ba file dataset trong `ad_bench/`.
- Tạo 60 tổ hợp từ dataset, setting, seed và `T`.
- Tạo thư mục `output/ddae_runs/`.

Nếu thiếu dataset hoặc có nhiều file khớp cùng tên, script dừng trước training. Nếu `results.csv` đã tồn tại, script cũng dừng để tránh ghi lặp; khi đó dùng `--resume` hoặc output directory mới.

---

### 4.2 Chạy từng lượt

Thứ tự bắt đầu là:

```
campaign → unsupervised → seed 111 → T=50
campaign → unsupervised → seed 111 → T=1000
campaign → unsupervised → seed 222 → T=50
campaign → unsupervised → seed 222 → T=1000
...
```

Mỗi lượt thực hiện:

```
Reset seed
→ Load và kiểm tra dữ liệu
→ StandardScaler trên toàn bộ X
→ Split theo setting và seed
→ Tạo model mới
→ Train 100 epochs
→ Evaluate mỗi 10 epochs
→ Predict thêm lần cuối
→ Tính AP, PR-AUC, ROC-AUC
→ Ghi kết quả
```

Các giá trị 100 epochs và evaluation mỗi 10 epochs áp dụng nếu bạn giữ YAML gốc. Pipeline này bám theo `run.py` hiện có.

Những gì xảy ra trên terminal:

!image.png

---

### 4.3 Những gì bạn nhận được

```
output/ddae_runs/
├── results.csv
├── configs/
│   └── Config thực tế của từng lượt
└── errors/
    └── Traceback của các lượt lỗi
```

Terminal hiển thị tiến độ `[1/60]`, log evaluation và trạng thái `SUCCESS` hoặc `FAILED`.

- Một lượt lỗi: ghi lỗi vào CSV, rồi tiếp tục lượt tiếp theo.
- Nhấn `Ctrl+C`: dừng chương trình; các lượt đã ghi vẫn còn, nhưng lượt đang chạy chưa được lưu kết quả.
- Chạy lại với `-resume`: bỏ qua lượt thành công, chạy lại lượt lỗi hoặc chưa hoàn tất.

---

### 4.4 Những gì không xảy ra

- Không chạy DAE, DDPM hoặc DDAE-C.
- Không tự chuyển sang GPU; giữ mặc định CPU của repo.
- Không chọn epoch có metric tốt nhất.
- Không tự tổng hợp mean ± std.
- Không giảm epochs để debug.

---

### 4.5 Kết quả

- File kết quả được lưu trong `AnoDDAE\output\ddae_runs\results.csv`
- Ước lượng tổng thời gian chạy toàn bộ 60 lượt: **103 minutes** (phụ thuộc tốc độ CPU, trong đó dataset `campaign` có kích thước lớn nhất, cần train lâu).