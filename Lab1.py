import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


dataframe = pd.read_parquet("tickers.parquet")

print("Размер:", dataframe.shape)
print(dataframe.dtypes)

print("\n\t Типы данных")
print(dataframe.dtypes.value_counts())

print("\n\t Пропуски")
missing = dataframe.isna().sum().sort_values()
missing_pct = (dataframe.isna().mean() * 100).round(2)
print(pd.DataFrame({"missing": missing, "pct_%": missing_pct}))

print("\n\t Дубликаты")
print("Полных дубликатов строк:", dataframe.duplicated().sum())
print("Дубликатов по symbol:", dataframe["symbol"].duplicated().sum())

print("\n\t Категориальные переменные")
cat_cols = dataframe.select_dtypes(include=["str", "category", "bool"]).columns.tolist()
num_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
print("Категориальные:", cat_cols)
print("Числовые:", num_cols)
for c in cat_cols:
    print(f"  {c}: {dataframe[c].nunique()} уникальных")

print("\n\t Описательные статистики (числовые)")
stats = dataframe[num_cols].describe().T
stats['median'] = dataframe[num_cols].median()
stats['diff_mean_median'] = stats['mean'] - stats['median']
stats['ratio_mean_median'] = stats['mean'] / stats['median']
stats_final = stats.round(2)

print(stats_final)

# 3. Корреляционный анализ ----------
corr = dataframe[num_cols].corr()
print("\n\t Корреляционная матрица")
print(corr.round(3))
 
plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Корреляционная матрица числовых признаков")
plt.tight_layout()
plt.savefig("corr_matrix.png", dpi=150)
plt.show()


# 4.1 Гистограмма (числовая): распределение market_cap (log-шкала)
fig, ax = plt.subplots(figsize=(9, 5))
mc = dataframe["market_cap"].dropna()
mc = mc[mc > 0]
ax.hist(np.log10(mc), bins=50, color="steelblue", edgecolor="white")
ax.set_title("Гистограмма log10(market_cap)")
ax.set_xlabel("log10(market_cap)")
ax.set_ylabel("Частота")
plt.tight_layout()
plt.savefig("hist_market_cap.png", dpi=150)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 4.2 Диаграмма рассеивания с линией тренда и 95% доверительным интервалом
fig, ax = plt.subplots(figsize=(10, 6))
sub = dataframe[(dataframe["volume"] > 0) & (dataframe["market_cap"] > 0)].copy()
sub["log_mcap"] = np.log10(sub["market_cap"])
sub["log_vol"] = np.log10(sub["volume"])
sns.regplot(data=sub, x="log_mcap", y="log_vol",
            scatter_kws={'alpha': 0.3, 's': 10, 'color': 'darkorange'}, 
            line_kws={'color': 'red', 'linewidth': 2},                    
            ax=ax)
ax.set_title("Рассеивание с линией регрессии и 95% доверительным интервалом")
ax.set_xlabel("log10(market_cap)")
ax.set_ylabel("log10(volume)")
plt.tight_layout()
plt.savefig("scatter_mcap_volume_ci.png", dpi=150)
plt.show()

# 4.3 Ящик с усами: log10(volume) по топ-10 sector
top_sectors = dataframe["sector"].value_counts().head(10).index
sub_vol = dataframe[(dataframe["sector"].isin(top_sectors)) & (dataframe["volume"] > 0)].copy()

sector_translation = {
    "Financial Services": "Финансовые услуги",
    "Healthcare": "Здравоохранение",
    "Technology": "Технологии",
    "Industrials": "Промышленность",
    "Consumer Cyclical": "Потреб. циклический",
    "Consumer Defensive": "Потреб. базовый",
    "Energy": "Энергетика",
    "Basic Materials": "Сырьё",
    "Real Estate": "Недвижимость",
    "Communication Services": "Коммуникации",
    "Utilities": "Коммунальные услуги"
}
sub_vol["sector_ru"] = sub_vol["sector"].map(sector_translation)

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=sub_vol,
            x="sector_ru", y="volume", 
            log_scale=True, ax=ax)

ax.set_title("Ящик с усами: объём торгов по топ-10 секторам (лог-шкала)")
ax.set_xlabel("Сектор")
ax.set_ylabel("Объём торгов (логарифмическая шкала)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("box_volume_sector_ru.png", dpi=150)
plt.show()

# 4.4 Категориальная-категориальная: sector × exchange 
top_sectors = dataframe["sector"].value_counts().head(10).index
top_exch = dataframe["exchange"].value_counts().head(8).index

ct = pd.crosstab(dataframe[dataframe["sector"].isin(top_sectors)]["sector"],
                 dataframe[dataframe["sector"].isin(top_sectors)]["exchange"])

ct = ct[[c for c in ct.columns if c in top_exch]]
sector_translation = {
    "Financial Services": "Финансовые услуги",
    "Healthcare": "Здравоохранение",
    "Technology": "Технологии",
    "Industrials": "Промышленность",
    "Consumer Cyclical": "Потреб. циклический",
    "Consumer Defensive": "Потреб. базовый",
    "Energy": "Энергетика",
    "Basic Materials": "Сырьё",
    "Real Estate": "Недвижимость",
    "Communication Services": "Коммуникации",
    "Utilities": "Коммунальные услуги"
}

ct.index = ct.index.map(sector_translation)
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)

ax.set_title("Тепловая карта: Сектор×Биржа")
ax.set_xlabel("Биржа")
ax.set_ylabel("Сектор")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("cat_cat_sector_exchange_ru.png", dpi=150)
plt.show()

# 4.5 Агрегирование: средний market_cap по секторам
agg = (dataframe.groupby("sector")["market_cap"]
         .agg(["mean", "median", "count"])
         .sort_values("mean", ascending=False)
         .head(10))

sector_translation = {
    "Financial Services": "Финансовые услуги",
    "Healthcare": "Здравоохранение",
    "Technology": "Технологии",
    "Industrials": "Промышленность",
    "Consumer Cyclical": "Потреб. циклический",
    "Consumer Defensive": "Потреб. базовый",
    "Energy": "Энергетика",
    "Basic Materials": "Сырьё",
    "Real Estate": "Недвижимость",
    "Communication Services": "Коммуникации",
    "Utilities": "Коммунальные услуги"
}

agg.index = agg.index.map(sector_translation)

fig, ax = plt.subplots(figsize=(11, 6))
agg["mean"].plot(kind="barh", ax=ax, color="seagreen")

ax.set_title("Агрегирование: средний market_cap по топ-10 секторам")
ax.set_xlabel("Средний market_cap")
ax.set_ylabel("Сектор")
ax.invert_yaxis()  

plt.tight_layout()
plt.savefig("agg_mcap_sector_ru.png", dpi=150)
plt.show()

# 4.6 Числовая-категориальная: распределение quote_type
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=dataframe, x="quote_type", ax=ax)
ax.set_title("Числовая-категориальная: количество по quote_type")
plt.tight_layout()
plt.savefig("count_quote_type.png", dpi=150)
plt.show()
