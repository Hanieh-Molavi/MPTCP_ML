import pandas as pd
import numpy as np

# تولید جریان‌های MPTCP شبیه‌سازی‌شده
np.random.seed(42)  # برای تولید نتایج تکرارپذیر

num_flows = 10  # تعداد زیرجریان‌ها

data = {
    "path_id": range(1, num_flows + 1),
    "packet_size": np.random.randint(500_000, 2_000_000, size=num_flows),  # اندازه بسته (بیت)
    "bandwidth": np.random.randint(50_000_000, 200_000_000, size=num_flows),  # پهنای باند (bps)
    "distance": np.random.randint(10_000, 100_000, size=num_flows),  # فاصله مسیر (متر)
    "propagation_speed": 2e8,  # سرعت انتشار (ثابت: متر بر ثانیه)
    "queuing_delay": np.random.uniform(0.001, 0.01, size=num_flows),  # تأخیر صف (ثانیه)
    "processing_delay": np.random.uniform(0.001, 0.005, size=num_flows),  # تأخیر پردازش (ثانیه)
}

# تبدیل به DataFrame
flows_df = pd.DataFrame(data)
flows_df

# محاسبه تأخیر کل
def calculate_total_delay(row):
    transmission_delay = row["packet_size"] / row["bandwidth"]
    propagation_delay = row["distance"] / row["propagation_speed"]
    total_delay = transmission_delay + propagation_delay + row["queuing_delay"] + row["processing_delay"]
    return total_delay

flows_df["total_delay"] = flows_df.apply(calculate_total_delay, axis=1)
flows_df

from sklearn.externals import joblib

# بارگذاری مدل SVM
svm_model = joblib.load("svm_model.pkl")  # مسیر فایل مدل SVM

# ویژگی‌های ورودی برای پیش‌بینی
features = flows_df[["packet_size", "bandwidth", "distance", "queuing_delay", "processing_delay"]]

# پیش‌بینی مناسب بودن مسیر
flows_df["is_suitable"] = svm_model.predict(features)
flows_df
