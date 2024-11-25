import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from PIL import Image 
import PIL 

#veri çekme fonksiyonu
def main(assetname, period):
    if "USD" in assetname:
        assetname = assetname
    else:
        assetname += ".IS"
    df = yf.Ticker(assetname).history(interval='1d',period=period)
    df = df[df.columns[:4]]
    df['Open'] = df['Open'].astype(np.float64)
    df['High'] = df['High'].astype(np.float64)
    df['Low'] = df['Low'].astype(np.float64)
    df['Close'] = df['Close'].astype(np.float64)
    df['Return'] = (df['Close']/df['Close'].shift(7)-1)*100
    df.reset_index(inplace=True)
    df['assetname'] = [assetname for i in range(len(df))]

    #çekilen kapanış verisine rsi verilerini ekleme
    return rsi_adding(df[['Date','assetname','Close', 'Return']])

#çekilen kapanış verisine rsi verilerini ekleme
def rsi_adding(df):
    rsi_dict = {}
    for j in range(14,210,56):
        rsi_column_name = f'RSI{j}'
        rsi_dict[rsi_column_name] = ta.rsi(df['Close'], j)
        # df['RSI'+str(j)]=ta.rsi(df['Close'],j)

    rsi_df = pd.DataFrame(rsi_dict, index=df.index)
    df = pd.concat([df, rsi_df], axis=1)

    return df

#close, date, assetname sütunlarından kurtulup kümeleme için kullanılacak variable lar ayrılıyor
def train_determination(df):
    train_list=[]

    for i in df.columns:
        if 'RSI' in i:
            train_list.append(i)

    df_train = df[train_list]
    return df_train

#her indikatör verisi kendi içerisinde scale ediliyor
def scaling(df_train):
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_data = scaler.fit_transform(df_train)
    scaled_df = pd.DataFrame(scaled_data, columns=df_train.columns)

    return scaled_df

#kümeleme tüm hisselerin o tarihteki indikatör verileri alınarak yapılıyor, mesela x hissesinde kümelerden herhangi biri olmayabilir
#kümeleme yapılıyor, tek yöntemle belki başka yöntemler denenebilir
def clustering(scaled_train, actual_data, cluster_number ,sort=True):
    hierarchical = AgglomerativeClustering(n_clusters=cluster_number)
    labels_train = hierarchical.fit_predict(scaled_train)
    actual_data['category'] = labels_train

    if sort:
        list_for_sort = actual_data.columns[4:]
        data_with_sorted_clusteres = sort_categorization(actual_data, list_for_sort)
    
    return data_with_sorted_clusteres

#görselleştirme
def plot_stocks(data, stock):
    # Assigns plotly as visualization engine
    pd.options.plotting.backend = 'plotly'
    # Arbitrarily 6 colors for our 6 clusters
    custom_colorscale = [
        [0, 'green'],  # 0, yeşil
        [1, 'red']     # 1, kırmızı
    ]
    # Create Scatter plot, assigning each point a color based
    # on it's grouping where group_number == index of color.
    fig = data.plot.scatter(
        x='Date',
        y="Close",
        color=data['new_category'],
        color_continuous_scale=custom_colorscale,
    )


    # Configure some styles
    layout = go.Layout(
        plot_bgcolor='#efefef',
        showlegend=False,
        # Font Families
        font_family='Monospace',
        font_color='#000000',
        font_size=20,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            ))
    )

    # fig.update_yaxes(type="log")
    fig.update_layout(layout, yaxis_type="log", title=stock)
    # Display plot in local browser window
    fig.show()

def new_plot(data, stock):
    fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Original Scatter", "Lines by Category"])
    
    custom_colorscale = [
        [0, 'green'],  # 0, yeşil
        [1, 'red']     # 1, kırmızı
    ]

    fig.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'],
    mode='markers',
    marker=dict(
        color=data['new_category'],
        colorscale=custom_colorscale,
        showscale=True
    ),
    name="Original Scatter",
    # visible=True if category == categories[0] else False,  # İlk kategori görünür
    hovertemplate=(
        "<b>Date</b>: %{x}<br>" +     # X ekseni değeri
        "<b>Close</b>: %{y}<br>" +   # Y ekseni değeri
        "<b>Category</b>: %{marker.color}<extra></extra>"  # new_category değeri
    )),
    row=1,col=1)

    for i in range(4,len(data.columns)-2):
        fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data[data.columns[i]],
        mode='markers',
        marker=dict(
            color=data['new_category'],
            colorscale=custom_colorscale,
            showscale=True
        ),
        name="Original Scatter",
        hovertemplate=(
            "<b>Date</b>: %{x}<br>" +     # X ekseni değeri
            "<b>Close</b>: %{y}<br>" +   # Y ekseni değeri
            "<b>Category</b>: %{marker.color}<extra></extra>"  # new_category değeri
        )),
        row=2, col=1)

        layout = go.Layout(
            plot_bgcolor='#efefef',
            showlegend=False,
            # Font Families
            font_family='Monospace',
            font_color='#000000',
            font_size=20,
            xaxis=dict(
                rangeslider=dict(
                    visible=False
                ))
        )

    fig.update_yaxes(type="log", row=1)
    fig.update_layout(layout, title=stock)
    # Display plot in local browser window
    fig.show()

def create_transition_matrix(data, return_data,categories):
    # Geçiş sayım matrisini başlat
    transition_matrix = np.zeros((len(categories), len(categories)))
    return_matrix = np.zeros((len(categories), len(categories)))

    # Geçişleri say
    for i in range(len(data) -8):
        getiri = return_data[i+8]

        current_state = data[i]  # Kategoriler 0-indexli olacak
        next_state = data[i + 1]
        transition_matrix[current_state, next_state] += 1
        
        return_matrix[current_state, next_state] += getiri
    
    return_matrix = return_matrix/transition_matrix

    # Sütun bazında normalizasyon yaparak olasılık matrisi oluştur
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    transition_df = pd.DataFrame(transition_matrix, index=categories, columns=categories)
    return_df = pd.DataFrame(return_matrix, index=categories, columns=categories)
    expected_df = transition_df*return_df
    
    return transition_df, return_df, expected_df

def plot_heatmap(data, categories):
    fig1 = go.Figure(data=go.Heatmap(
    z=data.values,  # Olasılık matrisini kullan
    x=categories,          # Sütun isimleri
    y=categories,               # Satır isimleri
    colorscale='Viridis',      # Renk skalası
    colorbar=dict(title="Olasılık")  # Renk barı başlığı
))

    # Grafik başlıkları ve eksenler
    fig1.update_layout(
        title="Geçiş Olasılık Matrisi",
        xaxis=dict(title="Sonraki Durum"),
        yaxis=dict(title="Mevcut Durum"),
        autosize=True
    )

    # Grafiği göster
    fig1.show()

#her bir hisse ayrı ayrı görselleştirileceği için o hissenin verilerini ayırıyoruz
def cutting_each_stock(data, stock_name):
    plotting_data = pd.DataFrame()
    plotting_data = data[data['assetname']==stock_name]

    return plotting_data

#gelen kümeler tek bir indikatöre göre olmadığı için görüntü güzel olmuyor:
#rsi14 ün düşük olduğu veriler kırmızıyken orta seviye olduğu veriler yeşil olabilir
#bizse her kümedeki indikatörlerin ortalamasına göre kümelerin ismini değiştiriyoruz
#algoritmanın '0' ismiyle kümelediği 5.olurken '9' olarak kümelediği 0.olabilir
def sort_categorization(clustered_df, column_list):
    df_means = clustered_df.groupby('category')[column_list].mean()
    df_means["Average_RSI"] = df_means.mean(axis=1)

    # 2. Average_RSI sütununa göre kategorileri sıralayıp yeni bir kategori sırası oluştur
    sorted_categories = df_means.sort_values(by='Average_RSI').index
    category_mapping = {cat: i for i, cat in enumerate(sorted_categories)}

    # 3. Orijinal DataFrame'deki kategorileri bu yeni sıralamaya göre değiştir
    clustered_df['new_category'] = clustered_df['category'].map(category_mapping)

    return clustered_df

bist30=["ULKER","AKBNK","ALARK","ASELS","ENKAI","EREGL","FROTO","GARAN","HEKTS","ISCTR","KCHOL","KRDMD","MGROS","PETKM","SAHOL","SASA","SISE","TCELL","THYAO","TOASO","TUPRS","YKBNK","DOAS","BIMAS","TTKOM","KOZAL","EKGYO","PGSUS","KONTR"]

#her bir hissenin verilerini alt alta birleştriyor.
#date ve close eşleşmesi korunuyor.
#her birinin indikatör değerleri alt alta ekleniyor.
#kümelemede topluca kullanılmak için
x=0
for i in bist30:
    if x == 0:
        data = main(i, "5y")
        x+=1
    else:
        df_1 = main(i, "5y")    
        data = pd.concat([data, df_1], axis=0, ignore_index=True)

data.dropna(inplace=True)
data_train = train_determination(data)
data_train_scaled = scaling(data_train)
clustered_data = clustering(data_train_scaled, data, 50)

stocks_list = clustered_data['assetname'].unique()

categories = list(range(0, 50))
transition_matrix, return_matrix, expected_matrix = create_transition_matrix(clustered_data['new_category'].values, clustered_data['Return'].values, categories)

plot_heatmap(expected_matrix, categories)
plot_heatmap(transition_matrix, categories)
plot_heatmap(return_matrix, categories)

for i in stocks_list:
    data_to_plot = cutting_each_stock(clustered_data, i)
    new_plot(data_to_plot, i)



import seaborn as sns
import matplotlib.pyplot as plt




