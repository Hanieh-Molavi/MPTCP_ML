from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

# خواندن داده‌ها
file_path = "Data/2-mix-.csv"  # مسیر فایل
data = pd.read_csv(file_path)

# فرض کنید ستون 'label' برچسب داده‌ها و بقیه ستون‌ها ویژگی‌ها هستند
features = data.drop(columns=['target'])  # جایگزین با نام ستون برچسب
labels = data['target']  # جایگزین با نام ستون برچسب

# تقسیم داده‌ها به مجموعه آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# تعریف مدل SVM با فعال کردن احتمال‌ها
model = SVC(probability=True, random_state=42)
model.fit(X_train, y_train)

# پیش‌بینی احتمال‌ها
predicted_probs = model.predict_proba(X_test)[:, 1]  # احتمال کلاس مثبت

# محاسبه FPR و TPR برای منحنی ROC
fpr, tpr, thresholds = roc_curve(y_test, predicted_probs)

# محاسبه مقدار AUC
roc_auc = auc(fpr, tpr)

# رسم منحنی ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()
