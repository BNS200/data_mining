import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats
from scipy.stats import norm
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

ru_names = {
    "Age": "Возраст",
    "Avg Monthly GB Download": "Ср. загрузка ГБ/мес",
    "Avg Monthly Long Distance Charges": "Ср. плата за межгород",
    "CLTV": "Пожизненная ценность (CLTV)",
    "Monthly Charge": "Ежемесячная плата",
    "Number of Dependents": "Кол-во иждивенцев",
    "Number of Referrals": "Кол-во рекомендаций",
    "Population": "Население",
    "Satisfaction Score": "Оценка удовлетворенности",
    "Tenure in Months": "Стаж (месяцев)",
    "Total Charges": "Общие расходы",
    "Total Extra Data Charges": "Доп. плата за трафик",
    "Total Long Distance Charges": "Общая плата за межгород",
    "Total Refunds": "Общая сумма возвратов",
    "Total Revenue": "Общий доход"
}

df = pd.read_csv("train.csv")
print("Shape:", df.shape)
print("\nТипы:\n", df.dtypes)
print("\nПропуски:\n", df.isna().sum())
print("\nДубликаты:", df.duplicated().sum())
print("\nУникальные значения:\n", df.nunique())
print("\nЦелевая:\n", df["Churn"].value_counts(normalize=True))
drop_cols = [
    "Churn Category", "Churn Reason", "Customer Status", "Churn Score",
    "Customer ID", "Lat Long", "Latitude", "Longitude", "Zip Code", "City",
    "Country", "State", "Quarter",
]
df = df.drop(columns=drop_cols)
print("\nПосле удаления утечек и идентификаторов:", df.shape)
num_cols = [
    "Age", "Avg Monthly GB Download", "Avg Monthly Long Distance Charges",
    "CLTV", "Monthly Charge", "Number of Dependents", "Number of Referrals",
    "Population", "Satisfaction Score", "Tenure in Months",
    "Total Charges", "Total Extra Data Charges",
    "Total Long Distance Charges", "Total Refunds", "Total Revenue",
]

df_corr = df[num_cols].rename(columns=ru_names)

print("1. КОРРЕЛЯЦИОННАЯ МАТРИЦА")
corr = df_corr.corr()
print(corr.round(2))

plt.figure(figsize=(14, 12)) 
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 8})
plt.title("Корреляционная матрица")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("corr_matrix.png", dpi=120)
plt.show()


high_corr = [(i, j, round(corr.loc[i, j], 2))
             for i in corr.columns for j in corr.columns
             if i < j and abs(corr.loc[i, j]) > 0.9]
print("\nПары с |r| > 0.9:", high_corr if high_corr else "нет")

print("2. ЗАПОЛНЕНИЕ ПРОПУСКОВ")

cat_with_na = ["Internet Type", "Offer"]

df_dropna = df.dropna().copy()
print(f"Способ 1 (dropna): {df_dropna.shape[0]} строк "
      f"(потеряно {df.shape[0] - df_dropna.shape[0]})")

df_fill = df.copy()
for col in cat_with_na:
    df_fill[col] = df_fill[col].fillna(df_fill[col].mode()[0])
print(f"Способ 2 (mode): пропусков осталось {df_fill.isna().sum().sum()}")

df_unknown = df.copy()
for col in cat_with_na:
    df_unknown[col] = df_unknown[col].fillna("Unknown")
print(f"Способ 3 (категория 'Unknown'): пропусков осталось "
      f"{df_unknown.isna().sum().sum()}")

df_clean = df_unknown.copy()


print("3. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ")
cat_cols = ["Gender", "Contract", "Payment Method", "Internet Type", "Offer"]

df_ohe = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
print(f"Способ 1 (One-Hot): {df_ohe.shape[1]} столбцов")

df_label = df_clean.copy()
for col in cat_cols:
    df_label[col] = LabelEncoder().fit_transform(df_label[col])
print(f"Способ 2 (Label): {df_label.shape[1]} столбцов")

df_freq = df_clean.copy()
for col in cat_cols:
    freq = df_freq[col].value_counts(normalize=True)
    df_freq[col + "_freq"] = df_freq[col].map(freq)
df_freq = df_freq.drop(columns=cat_cols)
print(f"Способ 3 (Frequency): {df_freq.shape[1]} столбцов")

df_encoded = df_ohe.copy()

print("4. ВЫБРОСЫ")
out_cols = ["Monthly Charge", "Total Charges", "CLTV", "Total Revenue"]

col = "Total Revenue"
data = df_encoded[col].dropna()
z_scores = stats.zscore(data)
outliers = data[np.abs(z_scores) > 3]
plt.figure(figsize=(12, 6))
sns.histplot(data, bins=50, kde=False, color='skyblue', label='Распределение данных', alpha=0.6)
for outlier in outliers:
    plt.axvline(x=outlier, color='red', linestyle='-', alpha=0.8, lw=1.5)

mean_val = data.mean()
std_val = data.std()
plt.axvline(x=mean_val, color='black', linestyle='--', label=f'Среднее (μ={mean_val:.0f})')
plt.axvline(x=mean_val + 3*std_val, color='orange', linestyle=':', label=f'Среднее + 3σ ({mean_val + 3*std_val:.0f})')
plt.title(f'Распределение {col} и найденные Z-выбросы (красные линии)', fontsize=14)
plt.xlabel('Значение (Общий доход)', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("z_score_practice.png", dpi=120)
plt.show()

z = np.abs(stats.zscore(df_encoded[out_cols]))
mask_z = (z < 3).all(axis=1)
print(f"Z-оценка (|z|<3): удалено {df_encoded.shape[0] - mask_z.sum()} строк")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(out_cols):
    sns.boxplot(
        y=df_encoded[col], 
        ax=axes[i], 
        color="skyblue",
        flierprops={"marker": "o", "markerfacecolor": "red", "markersize": 5, "markeredgecolor": "darkred"}
    )
    axes[i].set_title(f"Ящик с усами: {col}", fontsize=12)
    axes[i].set_ylabel("Значение")    
plt.suptitle("Выбросы в финансовых признаках", fontsize=16)
plt.tight_layout()
plt.savefig("boxplot_outliers.png", dpi=120)
plt.show()

Q1 = df_encoded[out_cols].quantile(0.25)
Q3 = df_encoded[out_cols].quantile(0.75)
IQR = Q3 - Q1
mask_iqr = ~((df_encoded[out_cols] < Q1 - 1.5*IQR) |
             (df_encoded[out_cols] > Q3 + 1.5*IQR)).any(axis=1)
print(f"IQR (1.5*IQR): удалено {df_encoded.shape[0] - mask_iqr.sum()} строк")

df_final = df_encoded[mask_z].reset_index(drop=True)
print(f"Итог после удаления выбросов (Z): {df_final.shape}")

df_final.to_csv("dataset_preprocessed.csv", index=False)
df_final.to_parquet("dataset_preprocessed.parquet", index=False)
print("\nСохранено: dataset_preprocessed.csv / .parquet")

print("5. PCA")

X = df_final.drop(columns=["Churn"])
y = df_final["Churn"]

X_scaled = StandardScaler().fit_transform(X)

pca2 = PCA(n_components=2, random_state=42)
X_pca2 = pca2.fit_transform(X_scaled)
print(f"PCA-2: объяснённая дисперсия = {pca2.explained_variance_ratio_}, "
      f"сумма = {pca2.explained_variance_ratio_.sum():.3f}")

pca3 = PCA(n_components=3, random_state=42)
X_pca3 = pca3.fit_transform(X_scaled)
print(f"PCA-3: объяснённая дисперсия = {pca3.explained_variance_ratio_}, "
      f"сумма = {pca3.explained_variance_ratio_.sum():.3f}")

plt.figure(figsize=(9, 7))
sns.scatterplot(x=X_pca2[:,0], y=X_pca2[:,1], hue=y,
                palette={0:"#4C72B0", 1:"#C44E52"}, alpha=0.6, s=15)
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("PCA (2 компоненты): Churn")
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig("pca_2d.png", dpi=120)
plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
for cls, color in [(0,"#4C72B0"), (1,"#C44E52")]:
    m = (y == cls)
    ax.scatter(X_pca3[m,0], X_pca3[m,1], X_pca3[m,2],
               label=f"Churn={cls}", color=color, alpha=0.5, s=10)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
ax.set_title("PCA (3 компоненты): Churn")
ax.legend()
plt.tight_layout()
plt.savefig("pca_3d.png", dpi=120)
plt.show()

loadings = pd.DataFrame(
    pca2.components_.T,
    columns=["PC1","PC2"],
    index=X.columns
).sort_values("PC1", key=abs, ascending=False)
print("\nТоп-10 признаков по вкладу в PC1:")
print(loadings.head(10).round(3))
