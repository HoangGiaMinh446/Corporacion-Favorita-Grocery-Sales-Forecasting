# Corporación Favorita Grocery Sales Forecasting

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)](https://xgboost.readthedocs.io/)

> **Project Description:** Dự án xây dựng mô hình dự báo doanh số bán hàng (Time-Series Forecasting) cho chuỗi cửa hàng bán lẻ Corporación Favorita. Dự án áp dụng phương pháp tiếp cận đa mô hình, so sánh hiệu quả giữa Machine Learning truyền thống (Linear Regression, XGBoost) và Deep Learning (MLP, LSTM) để tối ưu hóa bài toán tồn kho.

## Methodology & Workflow
Quy trình phân tích và xây dựng mô hình trong `favorita_code.ipynb` được thực hiện qua 4 bước chính:
### Data Preprocessing & EDA
* **Time-Series Processing:** Chuyển đổi và xử lý dữ liệu chuỗi thời gian chuẩn xác.
* **Data Integration:** Hợp nhất (Merge) dữ liệu từ nhiều nguồn khác nhau: *Sales, Stores, Oil Price, Holidays*.
* **Visualization:** Phân tích trực quan xu hướng doanh số theo chuỗi thời gian và theo đặc điểm cửa hàng (Cluster/Type).

### Feature Engineering (Kỹ thuật đặc trưng)
Áp dụng các kỹ thuật nâng cao để trích xuất thông tin từ dữ liệu lịch sử:
* **Time Features:** Tạo các biến về chu kỳ thời gian (*DayOfWeek, Quarter, Season*).
* **Lag Features:** Tạo biến độ trễ doanh số (*Lag 7, 14, 28, 30 ngày*) để mô hình nắm bắt quy luật từ quá khứ.
* **Rolling Statistics:** Tính toán trung bình trượt (*Rolling Mean*) để nhận diện xu hướng ngắn hạn và dài hạn.

### Modeling Strategy
Chiến lược tiếp cận đa mô hình từ cơ bản đến Deep Learning:

* **Baseline Model:**
    * `Linear Regression`: Sử dụng làm mốc cơ sở để so sánh hiệu quả.
* **Machine Learning:**
    * `XGBoost`: Mô hình Gradient Boosting mạnh mẽ cho dữ liệu bảng, tích hợp kỹ thuật **Early Stopping** để ngăn chặn Overfitting.
* **Deep Learning (PyTorch):**
    * `MLP` (Multi-Layer Perceptron): Mạng nơ-ron truyền thẳng đa lớp.
    * `LSTM` (Long Short-Term Memory): Mạng nơ-ron hồi quy, tối ưu cho việc học các phụ thuộc chuỗi dài trong dữ liệu thời gian.

### Evaluation
* **Metrics:** Đánh giá hiệu năng dựa trên các chỉ số `RMSE` (Root Mean Squared Error), `MAE` (Mean Absolute Error), và `R²`.
* **Selection:** So sánh trực tiếp kết quả giữa 4 mô hình để lựa chọn giải pháp tối ưu nhất cho bài toán dự báo tồn kho.

## Tech Stack

Dự án kết hợp sức mạnh của Machine Learning truyền thống và Deep Learning hiện đại:

* **Core & Data:** `Python 3`, `Pandas`, `NumPy` - Xử lý và biến đổi dữ liệu quy mô lớn.
* **Visualization:** `Matplotlib`, `Seaborn` - Trực quan hóa dữ liệu EDA và biểu đồ so sánh mô hình.
* **Machine Learning:**
    * `Scikit-Learn`: Xây dựng Baseline model (Linear Regression) và các metric đánh giá.
    * `XGBoost`: Sử dụng thuật toán **Gradient Boosting** tối ưu cho dữ liệu bảng, tích hợp *Early Stopping*.
* **Deep Learning:**
    * `PyTorch`: Xây dựng và huấn luyện các kiến trúc mạng nơ-ron phức tạp:
        * **MLP:** Mạng nơ-ron đa lớp.
        * **LSTM:** Mạng nơ-ron hồi quy để bắt các chuỗi phụ thuộc thời gian dài (Long-term dependencies).

---

## Usage Guide

Để chạy dự án này trên máy cục bộ, vui lòng thực hiện theo các bước chuẩn bị sau:

### Prerequisites (Cài đặt môi trường)
Đảm bảo đã cài đặt Python và các thư viện cần thiết.
*(Khuyến nghị: Sử dụng môi trường ảo `conda` hoặc `venv` để tránh xung đột phiên bản ví dụ như Google Colab).*

```bash: pip install pandas numpy matplotlib seaborn scikit-learn xgboost torch

### Prepare Data
Đảm bảo các file dữ liệu .csv (items, stores, train_final,...) đã được đặt đúng trong thư mục data_preprocessing/ như cấu trúc thư mục ở dưới.

### Run the Notebook
Mở và chạy file notebook trong thư mục src/ tuần tự từ trên xuống dưới (Run All)

## Model Performance

Kết quả đánh giá trên tập kiểm thử (Test Set) chứng minh **XGBoost** là mô hình đạt hiệu quả dự báo cao nhất. Chi tiết so sánh hiệu năng giữa các mô hình như sau:

* **XGBoost (Gradient Boosting) - Best Model**
    * **Hiệu suất:** `RMSE: [0.59]` | `MAE: [0.46]` | `R²: [0.54]`
    * **Đánh giá:** Đạt độ chính xác cao nhất trên dữ liệu bảng có cấu trúc. Xử lý rất tốt các biến động ngắn hạn và tính mùa vụ nhờ cơ chế Boosting và Early Stopping.

* **LSTM (Deep Learning)**
    * **Hiệu suất:** `RMSE: [0.61]` | `MAE: [0.48]` | `R²: [0.51]`
    * **Đánh giá:** Thể hiện khả năng nắm bắt các mẫu hình phi tuyến tính phức tạp và chuỗi phụ thuộc dài (Long-term dependencies). Tuy nhiên, thời gian huấn luyện lâu hơn so với XGBoost.

* **Linear Regression (Baseline)**
    * **Hiệu suất:** `RMSE: [0.65]` | `MAE: [0.51]` | `R²: [0.43]`
    * **Đánh giá:** Đóng vai trò mốc tham chiếu cơ sở. Hiệu quả thấp hơn các mô hình phi tuyến nhưng có ưu điểm là đơn giản và dễ giải thích (Explainable).

## Project Structure

```text
favorita-forecasting/
├── data_preprocessing/       # Chứa dữ liệu đầu vào đã qua xử lý sơ bộ
│   ├── items.csv             # Thông tin sản phẩm
│   ├── stores.csv            # Thông tin cửa hàng
│   ├── holidays_final.csv    # Dữ liệu ngày lễ tết
│   ├── oil_final.csv         # Giá dầu (yếu tố kinh tế vĩ mô)
│   ├── transactions_final.csv # Lịch sử giao dịch
│   └── train_final_sample.csv       # Dữ liệu huấn luyện chính đã được cắt bớt làm mẫu nhỏ do train_final gốc chứa gần 5 tỷ bản ghi không thể upload lên github
│
├── src/                      # Source code chính
│   └── favorita_code.ipynb   # Jupyter Notebook chứa toàn bộ quy trình end-to-end (EDA -> Modeling)
└── README.md                 # Tài liệu hướng dẫn
