# پیاده‌سازی، تنظیم پارامتر و مقایسه الگوریتم‌های Decision Tree و KNN

پروژه درس داده‌کاوی — نیمسال دوم سال تحصیلی ۱۴۰۴-۱۴۰۵

## 📌 معرفی پروژه

هدف این پروژه، آشنایی عملی با دو الگوریتم پایه‌ی طبقه‌بندی — **درخت تصمیم (Decision Tree)** و **K نزدیک‌ترین همسایه (KNN)** — از طریق پیاده‌سازی در **Python** و **RapidMiner**، تنظیم ابرپارامترها، و مقایسه عملکرد مدل‌ها با معیارهای استاندارد ارزیابی است.

## 📊 دیتاست

**Mobile Price Classification** ([منبع: Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification))

پیش‌بینی بازه‌ی قیمت گوشی موبایل (۴ کلاس: 0 تا 3) بر اساس مشخصات فنی دستگاه (RAM، باتری، وضوح صفحه، بلوتوث، وای‌فای، جی‌پی‌اس و ...).

- ستون هدف: `price_range`
- نوع مسئله: طبقه‌بندی چندکلاسه (Multi-class Classification)

## 🧠 مدل‌ها و ابرپارامترهای بررسی‌شده

| مدل | ابرپارامترها |
|---|---|
| Decision Tree | `criterion`: gini / entropy (یا gini_index / information_gain در RapidMiner) — `max_depth`: 1, 3, 5, 7, 10, بدون محدودیت |
| KNN | `K`: 3, 5, 7, 9, 11, 15 — `metric`: Euclidean / Manhattan |

## ⚙️ پیش‌پردازش

- تقسیم داده به آموزش/آزمون با نسبت ۸۰ به ۲۰ (Stratified)
- نرمال‌سازی داده‌ها با `StandardScaler` (فقط برای KNN؛ Decision Tree نیازی به نرمال‌سازی ندارد)

## 🏆 نتایج خلاصه

| پیاده‌سازی | بهترین مدل | بهترین تنظیمات | Accuracy |
|---|---|---|---|
| Python | Decision Tree | entropy, depth=10 | 88.00% |
| Python | KNN | Manhattan, K=15 | 64.75% |
| RapidMiner | Decision Tree | information_gain, depth=10 | 82.93% |
| RapidMiner | KNN | Euclidean/Manhattan, K=1 | 100%* |

\* دقت ۱۰۰٪ در K=1 به‌دلیل ریسک بیش‌برازش/نشت داده باید با احتیاط تفسیر شود؛ جزئیات در گزارش کامل آمده است.

جزئیات کامل نتایج، نمودارهای عمق درخت و K، ماتریس‌های درهم‌ریختگی، و پاسخ تشریحی به سؤالات تحلیلی پروژه در فایل گزارش موجود است.

## 📁 ساختار فایل‌ها

```
├── mobile_price_classification.py      # کد پایتون (پیش‌پردازش، آموزش، ارزیابی هر دو مدل)
├── train.csv                           # دیتاست
├── all_results_summary.csv             # جدول خلاصه همه اجراهای Python
├── decision_tree_knn_process.rmp       # فایل فرآیند RapidMiner
├── cm_Best_DecisionTree.png            # ماتریس درهم‌ریختگی بهترین مدل Decision Tree
├── cm_Best_KNN.png                     # ماتریس درهم‌ریختگی بهترین مدل KNN
├── tree_depth_vs_accuracy.png          # نمودار تأثیر عمق درخت بر دقت
├── knn_k_vs_accuracy.png               # نمودار تأثیر K بر دقت KNN
└── گزارش_پروژه_داده_کاوی.docx          # گزارش کامل پروژه
```

## 🚀 اجرا

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python mobile_price_classification.py
```
  

پروژه درس داده‌کاوی — نیمسال دوم ۱۴۰۴-۱۴۰۵
