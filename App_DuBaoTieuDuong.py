import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CẤU HÌNH TRANG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Dự đoán nguy cơ tiểu đường",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# BƯỚC 1. XÁC ĐỊNH BÀI TOÁN
# ============================================================
# Bài toán: Dự đoán nguy cơ mắc bệnh tiểu đường dựa trên
# các thông số xét nghiệm máu và thông tin bệnh nhân.
# Đầu vào : Tuổi, Giới tính, Glucose, HbA1C, Cholesterol,
#            HDL, LDL, Triglycerid, Creatinine, Ure,
#            AcidUric, AST, ALT
# Đầu ra  : NguyCoTieuDuong (Có nguy cơ / Không có nguy cơ)
# Thuật toán: Logistic Regression, Random Forest, SVM


# ============================================================
# BƯỚC 2–6. ĐỌC DỮ LIỆU, TIỀN XỬ LÝ VÀ HUẤN LUYỆN MÔ HÌNH
# ============================================================

@st.cache_resource
def train_model():

    # --------------------------------------------------------
    # BƯỚC 2. ĐỌC DỮ LIỆU
    # --------------------------------------------------------

    df = pd.read_csv("CSDL_TieuDuong.csv")

    original_shape = df.shape

    # --------------------------------------------------------
    # BƯỚC 3. TIỀN XỬ LÝ DỮ LIỆU
    # --------------------------------------------------------

    # 3.1. LÀM SẠCH DỮ LIỆU
    # --------------------------------------------------------

    # Xóa khoảng trắng thừa trong dữ liệu văn bản
    object_columns = df.select_dtypes(include="str").columns
    for col in object_columns:
        df[col] = df[col].str.strip()

    # --- Đồng bộ dữ liệu cột GioiTinh ---
    df["GioiTinh"] = df["GioiTinh"].replace({
        "male":   "Nam",
        "Male":   "Nam",
        "female": "Nữ",
        "Female": "Nữ",
        "NU":     "Nữ",
        "Nu":     "Nữ"
    })

    # --- Đồng bộ nhãn cột NguyCoTieuDuong ---
    # Các giá trị hợp lệ: 1/TRUE/"Có nguy cơ" -> 1
    #                     0/FALSE/"Không có nguy cơ" -> 0
    def chuan_hoa_nhan(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().upper()
        if s in ["1", "TRUE", "CÓ NGUY CƠ", "CO NGUY CO"]:
            return 1
        if s in ["0", "FALSE", "KHÔNG CÓ NGUY CƠ",
                 "KHONG CO NGUY CO"]:
            return 0
        return np.nan

    df["NguyCoTieuDuong"] = df["NguyCoTieuDuong"].apply(
        chuan_hoa_nhan
    )

    # --- Xử lý tuổi bất hợp lý (hợp lệ: 1–120) ---
    df.loc[
        (df["Tuoi"] < 1) | (df["Tuoi"] > 120),
        "Tuoi"
    ] = np.nan

    # --- Xử lý Glucose bất hợp lý (hợp lệ: 1–50 mmol/L) ---
    df.loc[
        (df["Glucose"] < 1) | (df["Glucose"] > 50),
        "Glucose"
    ] = np.nan

    # --- Xử lý HbA1C bất hợp lý (hợp lệ: 3–20 %) ---
    df.loc[
        (df["HbA1C"] < 3) | (df["HbA1C"] > 20),
        "HbA1C"
    ] = np.nan

    # --- Xử lý Cholesterol bất hợp lý (hợp lệ: 1–20 mmol/L) ---
    df.loc[
        (df["Cholesterol"] < 1) | (df["Cholesterol"] > 20),
        "Cholesterol"
    ] = np.nan

    # --- Xử lý dữ liệu trùng lặp ---
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates(keep="first")
    df = df.reset_index(drop=True)

    cleaned_shape = df.shape

    # --------------------------------------------------------
    # 3.2. BIẾN ĐỔI DỮ LIỆU
    # --------------------------------------------------------

    # --- Mã hóa GioiTinh ---
    le = LabelEncoder()
    df["GioiTinh_ma"] = np.nan
    mask = df["GioiTinh"].notna()
    df.loc[mask, "GioiTinh_ma"] = le.fit_transform(
        df.loc[mask, "GioiTinh"]
    )

    # Thiết lập biến đầu vào X và biến đầu ra y
    feature_columns = [
        "Tuoi", "GioiTinh_ma",
        "Glucose", "HbA1C", "Cholesterol",
        "HDL", "LDL", "Triglycerid",
        "Creatinine", "Ure", "AcidUric",
        "AST", "ALT"
    ]

    X = df[feature_columns].copy()
    y = df["NguyCoTieuDuong"]

    # Loại bỏ các dòng nhãn không hợp lệ
    valid_rows = y.notna()
    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    # --------------------------------------------------------
    # 3.3. PHÂN CHIA DỮ LIỆU (70% Train, 30% Test)
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # Pipeline: điền giá trị thiếu bằng median + chuẩn hóa
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, feature_columns)
        ]
    )

    # --------------------------------------------------------
    # BƯỚC 4. CHỌN MÔ HÌNH & BƯỚC 5. HUẤN LUYỆN
    # --------------------------------------------------------

    models = {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42
            ))
        ]),
        "SVM": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", CalibratedClassifierCV(
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=42
                ),
                ensemble=False
            ))
        ])
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --------------------------------------------------------
        # BƯỚC 6. ĐÁNH GIÁ MÔ HÌNH
        # --------------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=["Không có nguy cơ", "Có nguy cơ"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model":          model,
            "accuracy":       accuracy,
            "report":         report_df,
            "confusion_matrix": cm
        }

    # Lưu dữ liệu đã làm sạch
    df.to_csv(
        "CSDL_TieuDuong_Final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "results":         results,
        "original_shape":  original_shape,
        "cleaned_shape":   cleaned_shape,
        "duplicate_count": duplicate_count,
        "missing_values":  df[feature_columns].isnull().sum(),
        "feature_columns": feature_columns
    }


# ============================================================
# HUẤN LUYỆN VÀ LẤY KẾT QUẢ
# ============================================================

try:
    data = train_model()
except FileNotFoundError:
    st.error(
        "Không tìm thấy file CSDL_TieuDuong.csv. "
        "Hãy đặt file CSV cùng thư mục với chương trình."
    )
    st.stop()
except Exception as error:
    st.error("Chương trình gặp lỗi khi xử lý dữ liệu.")
    st.exception(error)
    st.stop()


# ============================================================
# BƯỚC 7. XÂY DỰNG GIAO DIỆN STREAMLIT
# ============================================================

st.title("🩺 Dự đoán nguy cơ bệnh tiểu đường")
st.write(
    "Nhập các thông số xét nghiệm của bệnh nhân để "
    "dự đoán nguy cơ mắc bệnh tiểu đường."
)

# ---- SIDEBAR: THÔNG TIN MÔ HÌNH ----
st.sidebar.header("📊 Thông tin mô hình")
st.sidebar.write("Kích thước dữ liệu ban đầu:",
                 data["original_shape"])
st.sidebar.write("Kích thước sau làm sạch:",
                 data["cleaned_shape"])
st.sidebar.write("Số dòng trùng đã xóa:",
                 int(data["duplicate_count"]))
st.sidebar.write("Tỷ lệ Train / Test:", "70% / 30%")
st.sidebar.markdown("---")
st.sidebar.subheader("So sánh độ chính xác")
for name, res in data["results"].items():
    st.sidebar.write(
        f"**{name}:** {res['accuracy'] * 100:.2f}%"
    )

# ---- CHỌN MÔ HÌNH ĐỂ XEM ĐÁNH GIÁ ----
st.subheader("📈 Kết quả đánh giá mô hình")
model_choice = st.selectbox(
    "Chọn mô hình để xem chi tiết:",
    list(data["results"].keys())
)

chosen = data["results"][model_choice]

with st.expander(
    f"Xem đánh giá chi tiết – {model_choice}",
    expanded=True
):
    st.metric(
        label="Accuracy",
        value=f"{chosen['accuracy'] * 100:.2f}%"
    )

    st.subheader("Classification Report")
    st.dataframe(
        chosen["report"].round(3),
        use_container_width=True
    )

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        chosen["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Không có nguy cơ", "Có nguy cơ"],
        yticklabels=["Không có nguy cơ", "Có nguy cơ"],
        ax=ax
    )
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Ma trận nhầm lẫn – {model_choice}")
    st.pyplot(fig)
    plt.close(fig)

# ---- NHẬP THÔNG TIN BỆNH NHÂN ----
st.header("🔍 Nhập thông tin bệnh nhân")

col1, col2 = st.columns(2)

with col1:
    tuoi = st.number_input(
        "Tuổi", min_value=1, max_value=120,
        value=45, step=1
    )
    gioi_tinh = st.selectbox(
        "Giới tính", ["Nam", "Nữ"]
    )
    glucose = st.number_input(
        "Glucose (mmol/L)", min_value=1.0,
        max_value=50.0, value=5.5, step=0.1,
        help="Chỉ số đường huyết. Bình thường: 3.9–6.1 mmol/L"
    )
    hba1c = st.number_input(
        "HbA1C (%)", min_value=3.0,
        max_value=20.0, value=5.7, step=0.01,
        help="Đường huyết trung bình 3 tháng. Bình thường < 5.7%"
    )
    cholesterol = st.number_input(
        "Cholesterol (mmol/L)", min_value=1.0,
        max_value=20.0, value=5.5, step=0.1,
        help="Mỡ máu tổng. Bình thường < 5.2 mmol/L"
    )
    hdl = st.number_input(
        "HDL (mmol/L)", min_value=0.1,
        max_value=5.0, value=1.6, step=0.1,
        help="Cholesterol tốt. Bình thường > 1.0 mmol/L"
    )
    ldl = st.number_input(
        "LDL (mmol/L)", min_value=0.1,
        max_value=10.0, value=3.45, step=0.01,
        help="Cholesterol xấu. Bình thường < 3.4 mmol/L"
    )

with col2:
    triglycerid = st.number_input(
        "Triglycerid (mmol/L)", min_value=0.1,
        max_value=30.0, value=2.1, step=0.1,
        help="Mỡ máu trung tính. Bình thường < 1.7 mmol/L"
    )
    creatinine = st.number_input(
        "Creatinine (µmol/L)", min_value=10.0,
        max_value=500.0, value=85.0, step=0.1,
        help="Chức năng thận. Nam: 62–115, Nữ: 53–97 µmol/L"
    )
    ure = st.number_input(
        "Ure (mmol/L)", min_value=0.5,
        max_value=50.0, value=5.3, step=0.1,
        help="Chức năng thận. Bình thường: 2.5–7.5 mmol/L"
    )
    acid_uric = st.number_input(
        "Acid Uric (µmol/L)", min_value=50.0,
        max_value=1000.0, value=300.0, step=0.1,
        help="Gút. Nam < 420, Nữ < 360 µmol/L"
    )
    ast = st.number_input(
        "AST (U/L)", min_value=1.0,
        max_value=1000.0, value=25.0, step=0.1,
        help="Enzyme gan. Bình thường: 10–40 U/L"
    )
    alt = st.number_input(
        "ALT (U/L)", min_value=1.0,
        max_value=1000.0, value=20.0, step=0.1,
        help="Enzyme gan. Bình thường: 7–40 U/L"
    )

# Chọn mô hình để dự đoán
mo_hinh_du_doan = st.selectbox(
    "Chọn mô hình để dự đoán:",
    list(data["results"].keys()),
    key="predict_model"
)

# ---- NÚT DỰ ĐOÁN ----
if st.button(
    "🔬 Dự đoán nguy cơ tiểu đường",
    type="primary",
    use_container_width=True
):
    # Mã hóa giới tính giống lúc huấn luyện
    # Nam -> 1, Nữ -> 0  (LabelEncoder sắp xếp alphabetical)
    # "Nam" > "Nữ" về Unicode nên: Nữ=0, Nam=1
    gioi_tinh_ma = 1 if gioi_tinh == "Nam" else 0

    # Tạo DataFrame chứa thông tin bệnh nhân mới
    new_patient = pd.DataFrame([{
        "Tuoi":        tuoi,
        "GioiTinh_ma": gioi_tinh_ma,
        "Glucose":     glucose,
        "HbA1C":       hba1c,
        "Cholesterol": cholesterol,
        "HDL":         hdl,
        "LDL":         ldl,
        "Triglycerid": triglycerid,
        "Creatinine":  creatinine,
        "Ure":         ure,
        "AcidUric":    acid_uric,
        "AST":         ast,
        "ALT":         alt
    }])

    # Lấy mô hình đã chọn
    model_selected = data["results"][mo_hinh_du_doan]["model"]

    # Dự đoán nhãn và xác suất
    prediction  = model_selected.predict(new_patient)[0]
    probability = model_selected.predict_proba(new_patient)[0]
    prob_khong  = probability[0]   # Xác suất không có nguy cơ
    prob_co     = probability[1]   # Xác suất có nguy cơ

    # ---- HIỂN THỊ KẾT QUẢ ----
    st.subheader("📋 Kết quả dự đoán")
    st.write(f"**Mô hình sử dụng:** {mo_hinh_du_doan}")

    if prob_co >= 0.65:
        st.error(
            "⚠️ Mô hình dự đoán: Bệnh nhân **CÓ NGUY CƠ** "
            "mắc bệnh tiểu đường!"
        )
    elif prob_co <= 0.35:
        st.success(
            "✅ Mô hình dự đoán: Bệnh nhân **KHÔNG CÓ NGUY CƠ** "
            "mắc bệnh tiểu đường."
        )
    else:
        st.warning(
            "🔶 Kết quả chưa rõ ràng – "
            "cần theo dõi và kiểm tra thêm."
        )

    # Hiển thị xác suất
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            label="Xác suất không có nguy cơ",
            value=f"{prob_khong * 100:.2f}%"
        )
    with col_b:
        st.metric(
            label="Xác suất có nguy cơ",
            value=f"{prob_co * 100:.2f}%"
        )

    # Thanh trực quan
    st.write("Mức độ nguy cơ tiểu đường:")
    st.progress(float(prob_co))

    # ---- GIẢI THÍCH KẾT QUẢ ----
    st.subheader("💡 Giải thích và khuyến nghị")

    canh_bao = []

    if glucose > 6.1:
        canh_bao.append(
            f"- **Glucose** = {glucose} mmol/L "
            f"(cao hơn mức bình thường 3.9–6.1 mmol/L)"
        )
    if hba1c >= 5.7:
        canh_bao.append(
            f"- **HbA1C** = {hba1c}% "
            f"({'≥ 6.5% – ngưỡng chẩn đoán tiểu đường' if hba1c >= 6.5 else '5.7–6.4% – tiền tiểu đường'})"
        )
    if cholesterol >= 5.2:
        canh_bao.append(
            f"- **Cholesterol** = {cholesterol} mmol/L "
            f"(cao hơn mức bình thường < 5.2 mmol/L)"
        )
    if triglycerid >= 1.7:
        canh_bao.append(
            f"- **Triglycerid** = {triglycerid} mmol/L "
            f"(cao hơn mức bình thường < 1.7 mmol/L)"
        )
    if hdl < 1.0:
        canh_bao.append(
            f"- **HDL** = {hdl} mmol/L "
            f"(thấp hơn mức bình thường > 1.0 mmol/L)"
        )

    if canh_bao:
        st.write("**Các chỉ số đáng chú ý:**")
        for cb in canh_bao:
            st.markdown(cb)
    else:
        st.write(
            "Các chỉ số xét nghiệm trong phạm vi bình thường."
        )

    if 0.35 < prob_co < 0.65:
        st.info(
            "Xác suất hai chiều tương đối cân bằng. "
            "Mô hình chưa đủ căn cứ để kết luận mạnh. "
            "Khuyến nghị bệnh nhân nên thực hiện xét nghiệm "
            "bổ sung và tham khảo ý kiến bác sĩ."
        )

    if prediction == 1:
        st.warning(
            "**Lưu ý:** Kết quả này chỉ mang tính tham khảo. "
            "Vui lòng đến cơ sở y tế để được tư vấn và "
            "chẩn đoán chính xác."
        )
