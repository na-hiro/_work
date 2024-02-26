import urllib.error
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

url = "https://indexes.nikkei.co.jp/nkave/historical/nikkei_500_stock_average_daily_jp.csv"
save_path = r'C:\Users\nagai\Desktop\DS_work\33_get\savename.csv'
opener=urllib.request.build_opener()
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)
urllib.request.urlretrieve(url, save_path)

# タブ区切りの文字を読み込む
df = pd.read_csv(save_path, encoding='shift_jis')
df = df.drop(df.shape[0]-1)
print(df)

df["データ日付"] = pd.to_datetime(df["データ日付"], format='%Y/%m/%d')
df = df.set_index('データ日付')
df = df[df.index > dt.datetime(2023,1,1)]
#df = df[df['データ日付'] > dt.datetime(2024,1,1)]
#df.reset_index()
print(df)


#不必要なカラムを削除。今回の指数の値は全て終値で示します。
df = df.drop(['始値', '高値', '安値'], axis=1)
df = df.sort_index(ascending=True)
print(df)

#名前の変更
df.rename(columns={'終値':'Nikkei500','業種別（水産）終値':'Fishery', '業種別（鉱業）終値':'Mining', '業種別（建設）終値':'Construction', '業種別（食品）終値':'Foods',\
'業種別（繊維）終値':'Fiber','業種別（パルプ・紙）終値':'Pulp & paper', '業種別（化学）終値':'Chemicals', '業種別（医薬品）終値':'Pharmaceuticals',\
'業種別（石油）終値':'Petroleum','業種別（ゴム）終値':'Rubber', '業種別（窯業）終値':'Glass & Ceramics','業種別（鉄鋼）終値':'Steel','業種別（非鉄・金属）終値':'Nonferrous metals',\
'業種別（機械）終値':'Machinery','業種別（電気機器）終値':'Electric machinery','業種別（造船）終値':'Shipbuilding','業種別（自動車）終値':'Automotive',\
'業種別（輸送用機器）終値':'Transportation instruments','業種別（精密機器）終値':'Precision instruments', '業種別（その他製造）終値':'Other manufacturing','業種別（商社）終値':'Trading companies',\
'業種別（小売業）終値':'Retail','業種別（銀行）終値':'Banking','業種別（その他金融）終値':'Other financial services','業種別（証券）終値':'Securities',\
'業種別（保険）終値':'Insurance','業種別（不動産）終値':'Real estate','業種別（鉄道・バス）終値':'Railway/bus','業種別（陸運）終値':'Land transport',\
'業種別（海運）終値':'Marine transport', '業種別（空運）終値':'Air transport', '業種別（倉庫）終値':'Warehousing', '業種別（通信）終値':'Communications',\
'業種別（電力）終値':'Electric power', '業種別（ガス）終値':'Gas', '業種別（サービス）終値':'Services'}, inplace=True)


print(df)
for x,y in df.items():
    print(x)
    print(y)
    #if x != "データ日付":
    m=df[x].mean()
    df[x]=df[x].div(m)
    print(df[x])

# 転置
df = df.transpose()
print(df.drop(df.columns[[0]], axis=1))
print(df)
df = df.reset_index(drop=True)
print(df)
"""
df.plot(figsize=(30,10),fontsize=8,linewidth=1,layout=(8,8), subplots=True, grid=True, xlabel = 'Date')
plt.show()

df2 = df/df.iloc[0,:]*100
df3 = df2.iloc[:, [0,36]]
df3.plot(figsize=(8,5),fontsize=10, grid=True, color = ['b','hotpink'])
plt.xlabel('Date')
plt.ylabel('Relative value')
plt.show()
"""
print("end")


#共通事前処理

#余分なワーニングを非表示にする
import warnings
warnings.filterwarnings('ignore')

#ライブラリのimport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#データフレーム表示用関数
from IPython.display import display

#表示オプション調整
#NumPyの浮動少数点の表示精度
np.set_printoptions(suppress=True, precision=4)
#pandasでの浮動少数点の表示精度
pd.options.display.float_format = '{:.4f}'.format
#データフレームですべての項目を表示
pd.set_option("display.max_columns",None)
#グラフのデフォルトフォント設定
plt.rcParams["font.size"] = 14
#乱数の種
random_seed = 123

print(type(df))
#df = df.dropna(how='any', axis=1)
#df = df.drop("データ日付", axis=1)

df_clustering = df

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_clustering_sc = sc.fit_transform(df_clustering)

#kmeans = KMeans(n_clusters=5, random_state=0)
kmeans = KMeans(n_clusters=5, init='k-means++',
    n_init=10, max_iter=1000, tol=0.0001, verbose=0,
    random_state=0, copy_x=True, algorithm='lloyd')
clusters = kmeans.fit(df_clustering_sc)
df_clustering["cluster"] = clusters.labels_
print(df_clustering["cluster"].unique())
print(df_clustering)

centers = kmeans.cluster_centers_
print(centers)
for i, lists in enumerate(centers):
    plt.plot(lists)
plt.close()

from sklearn.decomposition import PCA
X = df_clustering_sc
pca = PCA(n_components=2)
pca.fit(X)
x_pca = pca.transform(X)
pca_df = pd.DataFrame(x_pca)
pca_df["cluster"] = df_clustering["cluster"]

for i in df_clustering["cluster"].unique():
    tmp = pca_df.loc[pca_df["cluster"]==i]
    for index in tmp.index:
        if index == 0:
            target = tmp
            print(target)
    print(i)
    print(tmp)
    plt.scatter(tmp[0], tmp[1])

print(target)

"""
for i in df_clustering["cluster"].unique():
    tmp = pca_df.loc[pca_df["cluster"]==i]
    print(i)
    print(tmp)
    plt.scatter(tmp[0], tmp[1])
plt.show()
"""