import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from pytickersymbols import PyTickerSymbols
from sklearn.decomposition import PCA

#veri çekme fonksiyonu
def main(assetname, period):
    # if "USD" in assetname:
    #     assetname = assetname
    # else:
    #     assetname += ".IS"
    df = yf.Ticker(assetname).history(interval='1d', start='2017-09-30', end='2023-09-30')
    if df.empty:
        return df
    else:
        df = df[df.columns[:4]]
        df['Open'] = df['Open'].astype(np.float64)
        df['High'] = df['High'].astype(np.float64)
        df['Low'] = df['Low'].astype(np.float64)
        df['Close'] = df['Close'].astype(np.float64)
        df['Return'] = (df['Close']/df['Close'].shift(1)-1)
        df.reset_index(inplace=True)
        df['assetname'] = [assetname for i in range(len(df))]
        df = rsi_adding(df, df[['Close']])
        # df = dmi_adding(df, df[['High','Low','Close']])
        #çekilen kapanış verisine rsi verilerini ekleme
    return df

#çekilen kapanış verisine rsi verilerini ekleme
def rsi_adding(df, input):
    rsi_dict = {}
    for j in range(14,210,56):
        rsi_column_name = f'RSI{j}'
        rsi_dict[rsi_column_name] = ta.rsi(input['Close'], j)
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
        if 'DMI' in i:
            train_list.append(i)

    df_train = df[train_list]
    return df_train

def dmi_adding(df, input):
    dmi_dict = {}
    for j in range(14,210,56):
        dmi_column_name = f'DMI{j}'
        dmi = ta.adx(input['High'], input['Low'],input['Close'], j)
        dmi['Difference_dm'] = dmi[dmi.columns[1]]-dmi[dmi.columns[2]]
        dmi_dict[dmi_column_name] = dmi['Difference_dm']      
     
    dmi_df = pd.DataFrame(dmi_dict, index=df.index)
    df = pd.concat([df, dmi_df], axis=1)

    return df    

#her indikatör verisi kendi içerisinde scale ediliyor
def scaling(df_train):
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_data = scaler.fit_transform(df_train)
    scaled_df = pd.DataFrame(scaled_data, columns=df_train.columns)

    return scaled_df

#kümeleme tüm hisselerin o tarihteki indikatör verileri alınarak yapılıyor, mesela x hissesinde kümelerden herhangi biri olmayabilir
#kümeleme yapılıyor, tek yöntemle belki başka yöntemler denenebilir
def clustering(scaled_train, actual_data, cluster_number ,sort=True, pca=True):
    hierarchical = AgglomerativeClustering(n_clusters=cluster_number)
    labels_train = hierarchical.fit_predict(scaled_train)
    
    if pca:
        actual_data = pd.concat([actual_data, scaled_train], axis=1)

    actual_data['category'] = labels_train

    if sort:
        list_for_sort = actual_data.columns[7:11]
        data_with_sorted_clusteres = sort_categorization(actual_data, list_for_sort)
    
    return data_with_sorted_clusteres

def new_plot(data, stock):

    import plotly.express as px
    viridis_colors = px.colors.sequential.Viridis[:10]  # Viridis paletinden ilk 10 renk

    fig = make_subplots(
    rows=1,
    cols=1,
    shared_xaxes=True,
    subplot_titles=["Original Scatter", "Lines by Category"])
    
    custom_colorscale = [
        [0, 'green'],  # 0, yeşil
        [0.33, 'yellow'],
        [0.66, 'purple'],
        [1, 'red']     # 1, kırmızı
    ]
    for i in range(10):
        # i = str(i)
        fig.add_trace(go.Scatter(
            x=data['Date'][data['new_category']==i],
            y=data['Close'][data['new_category']==i],
            mode='markers',
            marker=dict(
                color=viridis_colors[i],
                showscale=True
            ),
            name="Cluster-"+str(i))
            # visible=True if category == categories[0] else False,  # İlk kategori görünür
            # hovertemplate=(
            #     "<b>Date</b>: %{x}<br>" +     # X ekseni değeri
            #     "<b>Close</b>: %{y}<br>" +   # Y ekseni değeri
            #     "<b>Category</b>: %{marker.color}<extra></extra>"  # new_category değeri
            # )),
        ,row=1,col=1)

    # for i in range(7,11):
    #     fig.add_trace(go.Scatter(
    #     x=data['Date'],
    #     y=data[data.columns[i]],
    #     mode='markers',
    #     marker=dict(
    #         color=data['new_category'],
    #         colorscale=custom_colorscale,
    #         showscale=True
    #     ),
    #     name="Original Scatter",
    #     hovertemplate=(
    #         "<b>Date</b>: %{x}<br>" +     # X ekseni değeri
    #         "<b>Close</b>: %{y}<br>" +   # Y ekseni değeri
    #         "<b>Category</b>: %{marker.color}<extra></extra>"  # new_category değeri
    #     )),
    #     row=2, col=1)

           # for i in range(11,15):
    #         fig.add_trace(go.Scatter(
    #         x=data['Date'],
    #         y=data[data.columns[i]],
    #         mode='markers',
    #         marker=dict(
    #             color=data['new_category'],
    #             colorscale=custom_colorscale,
    #             showscale=True
    #         ),
    #         name="Original Scatter",
    #         hovertemplate=(
    #             "<b>Date</b>: %{x}<br>" +     # X ekseni değeri
    #             "<b>Close</b>: %{y}<br>" +   # Y ekseni değeri
    #             "<b>Category</b>: %{marker.color}<extra></extra>"  # new_category değeri
    #         )),
    #         row=3, col=1)

    #         layout = go.Layout(
    #             plot_bgcolor='#efefef',
    #             showlegend=False,
    #             # Font Families
    #             font_family='Monospace',
    #             font_color='#000000',
    #             font_size=20,
    #             xaxis=dict(
    #                 rangeslider=dict(
    #                     visible=False
    #                 ))
    #         )


    # Kategorilere göre butonları dinamik oluşturma
    

    fig.update_yaxes(type="log", row=1)

    fig.update_layout(title=stock,
            plot_bgcolor='#efefef',
            showlegend=True,
            font_family='Monospace',
            font_color='#000000',
            font_size=20,
            xaxis=dict(
                rangeslider=dict(visible=False)
            ))
       # Display plot in local browser window
    fig.show()

def create_transition_matrix(data, assetname_data, return_data,categories):
    # Geçiş sayım matrisini başlat
    transition_matrix = np.zeros((len(categories), len(categories)))
    after_transition_return_matrix = np.ones((len(categories), len(categories)))
    after_transition_same_state_matrix = np.zeros((len(categories), len(categories)))
    
    # Geçişleri say
    complete_change = False
    getiri_list = []
    for i in range(len(data) -2):
        current_asset = assetname_data[i]
        next_asset = assetname_data[i+2]

        if current_asset == next_asset:
            
            getiri = return_data[i+2]+1

            current_state = data[i]  # Kategoriler 0-indexli olacak
            next_state = data[i + 1]
            transition_matrix[current_state, next_state] += 1

            if current_state != next_state:
                previous_state = current_state
                after_transition_return_matrix[current_state, next_state] *= getiri
                complete_change = True

            elif current_state == next_state and complete_change:
                after_transition_return_matrix[previous_state, current_state] *= getiri
                after_transition_same_state_matrix[previous_state, current_state] += 1
        else:
            complete_change = False
            continue

    # Sütun bazında normalizasyon yaparak olasılık matrisi oluştur
    # transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    transition_df = pd.DataFrame(transition_matrix, index=categories, columns=categories)
    return_df = pd.DataFrame(after_transition_return_matrix, index=categories, columns=categories)
    stay_df = pd.DataFrame(after_transition_same_state_matrix, index=categories, columns=categories)


    return transition_df, return_df, stay_df

def plot_heatmap(data, categories):
    # custom_colorscale = [
    #     [0, 'green'],  # 0, yeşil
    #     [1, 'red']     # 1, kırmızı
    # ]



    fig1 = go.Figure(data=go.Heatmap(
    z=data.values,  # Olasılık matrisini kullan
    x=categories,          # Sütun isimleri
    y=categories,               # Satır isimleri
    colorscale=["green", "yellow", "purple", "red"],      # Renk skalası
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

def pca(original_data, data, component_number):
    pca = PCA(n_components=component_number)
    columns_list = ['PCA'+ str(i) for i in range(1,component_number+1)]

    X_train = pca.fit_transform(data)
    pca_df = pd.DataFrame(data=X_train, columns=columns_list)

    explained_variance = pca.explained_variance_ratio_
    pca_components = pca.components_

    return pca_df

def analysis_of_matrix(matrix):
    sorted_values = matrix.stack().sort_values(ascending=False)
    top_10_percent = sorted_values.head(int(len(sorted_values) * 0.1))
    row_col_indices_top = top_10_percent.index

    sorted_values = matrix.stack().sort_values(ascending=True)
    btm_10_percent = sorted_values.head(int(len(sorted_values) * 0.1))
    row_col_indices_btm = btm_10_percent.index


stock_data = PyTickerSymbols()
nasdaq_tickers = list(stock_data.get_stocks_by_index('NASDAQ 100'))
nasdaq_symbols = []

for stock in nasdaq_tickers:
    nasdaq_symbols.append(stock['symbol'])

bist30=["ULKER","AKBNK","ALARK","ASELS","ENKAI","EREGL","FROTO","GARAN","HEKTS","ISCTR","KCHOL","KRDMD","MGROS","PETKM","SAHOL","SASA","SISE","TCELL","THYAO","TOASO","TUPRS","YKBNK","DOAS","BIMAS","TTKOM","KOZAL","EKGYO","PGSUS","KONTR"]

#her bir hissenin verilerini alt alta birleştriyor.
#date ve close eşleşmesi korunuyor.
#her birinin indikatör değerleri alt alta ekleniyor.
#kümelemede topluca kullanılmak için
def data_gathering(symbols):
    x=0
    for i in symbols:
        if x == 0:
            data = main(i, "5y")
            x+=1
        else:
            df_1 = main(i, "5y")
            if df_1.empty: 
                continue
            else:
                data = pd.concat([data, df_1], axis=0, ignore_index=True)
                x+=1
    return data

data = data_gathering(nasdaq_symbols[:50])
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
data_train = train_determination(data)
data_train_scaled = scaling(data_train)
# data_train_pca = pca(data, data_train_scaled, 3)
clustered_data = clustering(data_train_scaled, data, 10, True, False)

#classification vs clustering results on test data
# count = int(len(clustered_data)*0.8)
# X_train = data_train_scaled[:count].values
# Y_train = clustered_data[['new_category']][:count].values

# X_test = data_train_scaled[count:].values
# Y_test = clustered_data[['new_category']][count:].values


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_train_scaled, clustered_data['new_category'], test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=20)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

stocks_list = clustered_data['assetname'].unique()

categories = list(range(0, 10))
# transition_matrix, return_matrix, expected_matrix = create_transition_matrix(clustered_data['new_category'].values,clustered_data['assetname'], clustered_data['Return'].values, categories)

for i in stocks_list:
    data_to_plot = cutting_each_stock(clustered_data, i)
    data_to_plot.reset_index(drop=True, inplace=True)
    transition_matrix, return_matrix, stay_matrix = create_transition_matrix(data_to_plot['new_category'].values,data_to_plot['assetname'], data_to_plot['Return'].values, categories)
    analysis_of_matrix(stay_matrix)
    analysis_of_matrix(return_matrix)
    plot_heatmap(transition_matrix, categories)
    plot_heatmap(stay_matrix, categories)
    plot_heatmap(return_matrix, categories)
    
    new_plot(data_to_plot, i)