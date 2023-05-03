# 라이브러리 로드 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# 파일 불러오기 
import zipfile

with zipfile.ZipFile('./KOSPOXDACON.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')
train = pd.read_csv('./data/train.csv')
train.head(2)
test = pd.read_csv('./data/test.csv')
test.head(2)
train.shape
train.info()
train.isnull().sum()
train['ID'].nunique()
# 전처리 및 EDA
## 컬럼 별 내용요약
ID : 샘플 고유 ID

User-ID : 유저 고유 ID

Book-ID : 도서 고유 ID

Book-Rating : 유저가 도서에 부여한 평점 (0점 ~ 10점), 
단, 0점인 경우에는 유저가 해당 도서에 관심이 없고 관련이 없는 경우

Age : 유저의 나이 . 

Location : 유저가 사는 지역 

 Book-Title : 도서명

 Book-Author : 저자 

 Year of publication :  도서 출판 년도 (-1일 경우 결측 혹은 알 수 없음)

 Publisher : 출판사 


# 전체 데이터 확인 
# from pandas_profiling import ProfileReport

# train.profile_report()
train.hist(bins = 100, figsize = (12,6))
컬럼간의 뚜렷한 상관관계가 없음
plt.figure(figsize=(10,10))
sns.heatmap(train.corr(), annot=True, annot_kws={'size': 20}) #annot 크기 조정
plt.tick_params(labelsize=25)
train['Age'] = train['Age'].astype(int)
train['Year-Of-Publication'] = train['Year-Of-Publication'].astype(int)
'Age', 'Year-Of-Publication', 'BookRating'을 박스플롯으로 시각화하여 이상치확인  
    - Age데이터에 불가능한 값이 있음을 확인  
    - Year-Of-Publication에도 오류로 보이는 값이 들어있음을 확인  
    
sns.boxplot(x='Age', data=train)
sns.boxplot(x='Year-Of-Publication', data=train)
sns.boxplot(x='Book-Rating', data=train)
# USER-ID 컬럼
train['User-ID'].value_counts()
전체 데이터중 고유 ID값의 갯수 시각화 
# "User-ID" 컬럼과 "ID" 컬럼의 갯수 구하기
user_count = len(train['User-ID'].unique())
id_count = len(train['ID'].unique())

# 데이터프레임 만들기
count_df = pd.DataFrame({'column': ['User-ID', 'total number of data rows'], 'count': [user_count, id_count]})

# 그래프 그리기
plt.figure(figsize=(4,3))
plt.bar(count_df['column'], count_df['count'])
plt.title('Number of unique values in train dataset')
plt.xlabel('Column')
plt.ylabel('Count')
plt.show()
평균적으로 USER-ID 별 책 10권에 대한 리뷰 남김 
train['User-ID'].value_counts().mean()
# AGE 컬럼 
데이터셋의 가장 작은 나이가 0, 가장 많은 나이가 244세로 잘못된 데이터입력이 있다는 것을 확인
max_age = train['Age'].max() # Age 열의 최대값 찾기
min_age = train['Age'].min() # Age 열의 최소값 찾기

print('Maximum Age:', max_age)
print('Minimum Age:', min_age)
0세부터 244까지의 분포를 더 자세하게 확인함
sorted_age = sorted(train['Age'].unique())
age_list = list(sorted_age)
print(age_list)
Age 분포 plotly를 통한 시각화
import plotly.graph_objs as go

# Create the data for the bar chart
data = [go.Bar(
            x=train['Age'].value_counts().index,
            y=train['Age'].value_counts().values,
            hovertext=['Age: ' + str(age) + '<br>Count: ' + str(count) for age, count in zip(train['Age'].value_counts().index, train['Age'].value_counts().values)],
            hoverinfo='text'
    )]

# Set the layout of the figure
layout = go.Layout(title='Age distribution',
                   xaxis=dict(title='Age'),
                   yaxis=dict(title='Count'))

# Create the figure object
fig = go.Figure(data=data, layout=layout)

# Show the figure
fig.show()
train['Age'].plot.hist(bins=30) # 히스토그램 그리기
plt.title('Age Distribution') # 그래프 제목 설정
plt.xlabel('Age') # x축 레이블 설정
plt.ylabel('Count') # y축 레이블 설정
plt.show() # 그래프 보이기
35세의 유저들이 가장 많았고, 유저들의 평균나이는 약 36.7세 이다. 
pd.DataFrame(train["Age"].value_counts())
train["Age"].mean()
# AGE 컬럼 전처리
전처리 논의   
* 의견1) 0세는 상위1% 값이 15세로 대체. 모두 하위 4%와 3% 구간 사이에서 비교적 급격한 나이 변화가 일어났으므로 급격한 나이변화가 일어나는 구간부터는 모두 이상치로 판단, 따라서 하위4% 이하인 80세 이상은 모두 80세로 대체  
* 의견2) 상위 1%인 0-15세를 모두 15세로 대체하며, 하위1%에 해당하는 67세-244세는 모두 67세로 대체
* 의견3) 나이별 사분위수 와 threshold사용
-> 모델 예측점수 결과가 가장 좋았던 의견 1)을 사용하기로 함 
#"Age" 컬럼의 1% 분위수
age_quantile_1pct = train["Age"].quantile(0.01)
age_quantile_1pct 
age_quantile_993pct = train["Age"].quantile(0.993)
age_quantile_993pct
age_quantile_996pct = train["Age"].quantile(0.996)
age_quantile_996pct
age_quantile_997pct = train["Age"].quantile(0.997)
age_quantile_997pct
age_quantile_998pct = train["Age"].quantile(0.998)
age_quantile_998pct
age_quantile_999pct  = train["Age"].quantile(0.999)
age_quantile_999pct
train['Age'] = np.where(train['Age'] == 0, 15, np.where(train['Age'] >= 80, 80, train['Age']))
test['Age'] = np.where(test['Age'] == 0, 15, np.where(test['Age'] >= 80, 80, test['Age']))
# Year-Of-Publication
import matplotlib.pyplot as plt

plt.hist(train['Year-Of-Publication'], bins=50)
plt.xlabel('Year of Publication')
plt.ylabel('Count')
plt.show()

출판년도에 -1값이 11515 개 있음
train["Year-Of-Publication"].median()
year_error = -1
train[train["Year-Of-Publication"] == year_error]
# train['Year-Of-Publication'] = np.where((train['Year-Of-Publication'] < 1950) | (train['Year-Of-Publication'] > 2010), np.nan, train['Year-Of-Publication'])

# 결측치를 평균값으로 대체하기
mean_Year = train['Year-Of-Publication'].mean()
train['Year-Of-Publication'] = train['Year-Of-Publication'].fillna(mean_Year)


# Year-Of-Publication 전처리논의 
* 의견1) -1값은 최빈값인 1997로 대체    
* 의견2)  이상치 확인, 1950년보다 작거나 2010년보다 큰 값을 가지는 셀에 대해, 해당 셀을 NaN으로 변환 후,해당 결측치를 평균값으로 대체  
* 의견3) 변동 없음  
# Book-Author
### 유저들이 가장 많이 찾은 상위 10위 작가
ds = train['Book-Author'].value_counts().reset_index()
ds.columns = ['author', 'count']
ds = ds.sort_values('count', ascending=False).head(10)

plt.figure(figsize=(8, 9))
sns.barplot(x='count', y='author', data=ds, orient='h')
plt.title('Top 10 Book-Author')
plt.show()
ds
평균 평점이 좋은 상위10위 작가
author = train['Book-Author'].value_counts().reset_index()
author.columns = ['Book-Author', 'author_evaluation_count']
df = pd.merge(train, author)

mean_df = df[df['author_evaluation_count']>100]
mean_df = mean_df.groupby('Book-Author')['Book-Rating'].mean().reset_index().sort_values('Book-Rating', ascending=False)

top_50_mean_df = mean_df.head(10)

plt.figure(figsize=(10,9))
sns.barplot(x='Book-Rating', y='Book-Author', data=top_50_mean_df, orient='h')
plt.title('Top 10 Book-Author with highest avarage Book-Rating')
plt.show()
# Publisher
### 평균평점이 높은 도서의 상위10위 출판사
books = train['Publisher'].value_counts().reset_index()
books.columns = ['Publisher', 'Publisher_evaluation_count']
df = pd.merge(train, books)
mean_df = df[df['Publisher_evaluation_count']>100]
mean_df = mean_df.groupby('Publisher')['Book-Rating'].mean().reset_index().sort_values('Book-Rating', ascending=False)

mean_df['Publisher'].nunique()
publisher = train['Publisher'].value_counts().reset_index()
publisher.columns = ['Publisher', 'Publisher_evaluation_count']
df = pd.merge(train, publisher)
df['Publisher'] = df['Publisher'].replace('TokyoPop', 'Tokyopop')  # 'TokyoPop'을 'Tokyopop'으로 변경
mean_df = df[df['Publisher_evaluation_count'] > 100]
mean_df = mean_df.groupby('Publisher')['Book-Rating'].mean().reset_index().sort_values('Book-Rating', ascending=False)

sns.set_style("whitegrid")
plt.figure(figsize=(10, 9))
sns.barplot(x='Book-Rating', y='Publisher', data=mean_df.head(10), orient='h')
plt.title('Top 10 Publishers with highest avarage Book-Rating', fontsize=16)
plt.show()

# 유저별 도서이용 횟수 및 평균 평점
# "Book-ID" string타입으로 변환
train['Book-ID'] = train['Book-ID'].astype(str)

# "Book-ID"에서 "BOOK_" 부분제거
train['Book-ID'] = train['Book-ID'].str.replace('BOOK_', '')

# "Book-ID" 정수로 변환 
train['Book-ID'] = train['Book-ID'].astype(int)
train['Book-ID']
train['User-ID']
user_groupby = train.groupby("User-ID")
book_groupby = train.groupby("Book-ID")
average_user_rating = user_groupby["Book-Rating"].mean() # 각 유저가 매기는 책 평점의 평균
number_of_rating_by_user = user_groupby["Book-Rating"].count() #각 유저별 책 평점 매긴 횟수 
average_book_rating = book_groupby["Book-Rating"].mean() # 책 별 평균 평점
number_of_book_ratings = book_groupby["Book-Rating"].count() # 각 책 별 평균 평점의 개수 
number_of_book_ratings
average_user_rating.name = "avg_rating"
number_of_rating_by_user.name = "N_ratings"
average_book_rating.name = "avg_rating"
number_of_book_ratings.name = "N_ratings"
users = train[["User-ID"]].drop_duplicates().merge(number_of_rating_by_user, on="User-ID")
users = users.join(average_user_rating, on="User-ID")
users
* 예를들어 User-ID가 USER_00000인 사람의 도서이용횟수는 8회이며 평균적으로 4.75 평점을 남김 
* 가장 많은 도서기록 횟수는 1145회이다. 
users['N_ratings'].max()
# 도서별 이용 횟수와 평균 평점  
도서의 정보 및 이용 횟수와 평균 평점 추가한 데이터셋 생성
books =  train[["Book-ID","Book-Title", "Book-Author", "Year-Of-Publication"]].drop_duplicates().merge(number_of_book_ratings, on="Book-ID")
books = books.join(average_book_rating, on="Book-ID")
books
books['N_ratings'].mean()
popbooks = books.sort_values(by="N_ratings", ascending=False).nlargest(10, 'N_ratings')
popbooks
가장 많이 이용한 상위 10개 도서
from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams['figure.figsize'] = 10, 9

# seaborn 패키지로 barplot 그리기
sns.barplot(x='N_ratings', y='Book-Title', data=popbooks,palette='Set1')

# x축 레이블 90도 회전
plt.xticks(rotation=50)

# 그래프 제목 추가
plt.title('Top10 Books most read by users', fontsize=16)

# 그래프 출력
plt.show()

도서당 평점의 갯수의 평균이 3.57이므로 N_ratings가 4 이상인 평점좋은 도서는 다음과 같다 
goodratingbooks = books[(books['N_ratings'] >= 4)].sort_values(by="avg_rating", ascending=False).nlargest(10, 'avg_rating')
goodratingbooks
* 예를들어 Harry Potter 단어가 포함된 단어를 찾으면 해당 정보를 아래와 같이 찾을 수 있음
* 같은 책이라도 에디션에 따라 나뉘는 종류가 다르게 경우가 있다
* 같은 저자라도 J.K Rowling, Joanne K,Rowling, Joanne K.Rowling과 같이 다른 방식으로 표기가 되어있는 경우가 있다 
books[books["Book-Title"].str.contains("Harry Potter") & books["Book-Author"].str.contains("Rowling")]
# Location
- city, state, country 로 구성.
국가 정보만 있는 새로운 열 생성
# Location 변수에서 "vermilion", "ohio", "usa" 추출하여 새로운 변수 생성
new_train = train.copy()
# new_trains = count_null_values(new_train, "")
# new_train['City'] = train['Location'].str.split(', ').str[0]
# new_train['State'] = train['Location'].str.split(', ').str[1]
# new_train['Country'] = train['Location'].str.split(', ').str[-1]
countries = []
cond = new_train['Location'].str.split(',')

for cont in cond:
    
    countries.append(cont[-1].strip().title())
countries = []
cond = new_train['Location'].str.split(',')

for cont in cond:
    
    countries.append(cont[-1].strip().title())
new_train["Country"] = countries
new_train.head()
"Usa", "United Sates"와 같이 동일 국가여도 다르게 표기된 데이터들이 있어 해당 정보 통일해주는 전처리 진행
new_train.loc[new_train["Country"] == "Usa", "Country"] = "United States"
new_train.loc[new_train["Country"] == "España", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "England", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "Trinidad And Tobago", "Country"] = "Trinidad and Tobago"
new_train.loc[new_train["Country"] == "Deutschland", "Country"] = "Germany"
new_train.loc[new_train["Country"] == "Tanzania", "Country"] = "Tanzania, United Republic of"
new_train.loc[new_train["Country"] == "Moldova", "Country"] = "Moldova, Republic of"
new_train.loc[new_train["Country"] == "Czech Republic", "Country"] = "Czechia"
new_train.loc[new_train["Country"] == "South Korea", "Country"] = "Korea, Republic of"
new_train.loc[new_train["Country"] == "Venezuela", "Country"] = "Venezuela, Bolivarian Republic of"
new_train.loc[new_train["Country"] == "Galiza", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "Taiwan", "Country"] = "Taiwan, Province of China"
new_train.loc[new_train["Country"] == "Scotland", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "Bolivia", "Country"] = "Bolivia, Plurinational State of"
new_train.loc[new_train["Country"] == "Iran", "Country"] = "Iran, Islamic Republic of"
new_train.loc[new_train["Country"] == "United Sates", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Maricopa", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Galiza Neghra", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "Richmond Country", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Catalunya", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "Vietnam", "Country"] = "Viet Nam"
new_train.loc[new_train["Country"] == "La Chine Éternelle", "Country"] = "China"
new_train.loc[new_train["Country"] == "Lleida", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "La Chine Éternelle !", "Country"] = "China"
new_train.loc[new_train["Country"] == "La Chine Éternelle!", "Country"] = "China"
new_train.loc[new_train["Country"] == "Framingham", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Usa (Currently Living In England)", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "Alderney", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "Saint Loius", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Laos", "Country"] = "LA"
new_train.loc[new_train["Country"] == "Collin", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Burma", "Country"] = "Myanmar"
new_train.loc[new_train["Country"] == "Shelby", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Worcester", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Los Estados Unidos De Norte America", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Russia", "Country"] = "Russian Federation"
new_train.loc[new_train["Country"] == "Polk", "Country"] = "United States"
new_train.loc[new_train["Country"] == "U.A.E", "Country"] = "United Arab Emirates"
new_train.loc[new_train["Country"] == "U.S.A.", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Cherokee", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Cananda", "Country"] = "Canada"
new_train.loc[new_train["Country"] == "Morgan", "Country"] = "France"
new_train.loc[new_train["Country"] == "Cape Verde", "Country"] = "Cabo Verde"
new_train.loc[new_train["Country"] == "Antigua And Barbuda", "Country"] = "Antigua and Barbuda"
new_train.loc[new_train["Country"] == "Us", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Euskal Herria", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "Bosnia And Herzegovina", "Country"] = "Bosnia and Herzegovina"
new_train.loc[new_train["Country"] == "Ventura County", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Csa", "Country"] = "Canada"
new_train.loc[new_train["Country"] == "Hernando", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Prince William", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "Onondaga Nation", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Tobago", "Country"] = "Trinidad and Tobago"
new_train.loc[new_train["Country"] == "Catalonia", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "Urugua", "Country"] = "Uruguay"
new_train.loc[new_train["Country"] == "Phillipines", "Country"] = "Philippines"
new_train.loc[new_train["Country"] == "San Mateo", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Nz", "Country"] = "New Zealand"
new_train.loc[new_train["Country"] == "Italia", "Country"] = "Italy"
new_train.loc[new_train["Country"] == "Berguedà", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "Ferrara", "Country"] = "Italy"
new_train.loc[new_train["Country"] == "L`Italia", "Country"] = "Italy"
new_train.loc[new_train["Country"] == "Wales", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "Macau", "Country"] = "China"
new_train.loc[new_train["Country"] == "Macedonia", "Country"] = "North Macedonia"
new_train.loc[new_train["Country"] == "Channel Islands", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "United Kindgonm", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "Brunei", "Country"] = "Brunei Darussalam"
new_train.loc[new_train["Country"] == "K1C7B1", "Country"] = "Canada"
new_train.loc[new_train["Country"] == "St.Thomasi", "Country"] = "Canada"
new_train.loc[new_train["Country"] == "Catalunya Spain", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "La Belgique", "Country"] = "Belgium"
new_train.loc[new_train["Country"] == "Aroostook", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Rutherford", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Fort Bend", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Syria", "Country"] = "Syrian Arab Republic"
new_train.loc[new_train["Country"] == "U.K.", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "Madrid", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "Orense", "Country"] = "Spain"
new_train.loc[new_train["Country"] == "St. Helena", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "La France", "Country"] = "France"
new_train.loc[new_train["Country"] == "U.S. Of A.", "Country"] = "United States"
new_train.loc[new_train["Country"] == "United Staes", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Ee.Uu", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Alachua", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Burlington", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Van Wert", "Country"] = "United States"
new_train.loc[new_train["Country"] == "Camden", "Country"] = "United Kingdom"
new_train.loc[new_train["Country"] == "U.S>", "Country"] = "United States"
new_train.loc[new_train["Country"] == "America", "Country"] = "United States"
new_train.loc[new_train["Country"] == "United State", "Country"] = "United States"
new_train
올바르지 않은 명칭으로 표기된 국가정보를 찾기 위해 pycountry 라이브러리 사용
#!pip install pycountry
import pycountry

countries = [country.name for country in pycountry.countries]
country_values = new_train["Country"].unique()
import pycountry

country = pycountry.countries.lookup('BE')
print(country.name)
해당 데이터는 사용자가 자유롭게 지역을 기입하는 방식으로 구성되어 있어 아래와 같이 국가정보 파악이 힘든 데이터가 있음을 발견 
invalid_countries = []

for country in country_values:
    if country not in countries:
        print(f"Invalid country name: {country}")
        invalid_countries.append(country)

위 국가 정보 삭제 
for country in invalid_countries:
    new_train = new_train[new_train['Country'] != country]
국가명(Country)을 ISO-3166-1 alpha-3 국가 코드로 변환
#!pip install country_converter

import country_converter as coco
converted_country=coco.convert(names=new_train["Country"], to="ISO3")
new_train["Country"]=converted_country
countriestop10=new_train['Country'].value_counts().reset_index().head(10)
countriestop10.set_index('index')
countriestop10pct=new_train['Country'].value_counts(1).reset_index().head(10)
countriestop10pct.set_index('index')
sns.barplot(x=countriestop10.index, y=countriestop10.Country)
* USA =  미국
* CAN =  캐나다
* GBR =  영국 
* DEU =  독일
* AUS =  호주
* ESP = 스페인
* FRA = 프랑스
* PRT = 포르투갈
* NZL = 뉴질랜드
* MYS =  말레이시아 
도서이용량이 많은 나라는 미국이 압도적으로 가장 많으며 캐나다, 영국, 독일, 호주 등으로 분포되어있다
#!pip install plotly-express
import plotly.express as px
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)


country=new_train["Country"].value_counts()
fig=px.choropleth(locations=country.index,
                  color=country.values,
                  color_continuous_scale=px.colors.sequential.dense,
                  template='plotly_dark',
                  title='Distribution of users by countries')

fig.update_layout(font=dict(size=17, family="Franklin Gothic"))
fig.show()
# 나라별 도서 평균 평점 
book_count = new_train.groupby('Book-ID')['Book-Rating'].count().reset_index(name='Count')
book_rating = new_train.groupby('Book-ID')['Book-Rating'].mean().reset_index(name='Avg-Rating')
book = pd.merge(book_count, book_rating, on='Book-ID')
book['Rating-Per-Count'] = book['Avg-Rating'] / book['Count']
country_rating = new_train.merge(book, on='Book-ID').groupby('Country')['Rating-Per-Count'].mean()
country_rating = country_rating.sort_values(ascending=False).head(10)
plt.figure(figsize=(15, 8))
sns.barplot(x=country_rating.index, y=country_rating.values)
plt.xticks(rotation=90)
plt.title('Average Rating per Book Count by Country')
plt.xlabel('Country')
plt.ylabel('Average Rating per Book Count')
plt.show()

- UGA = 우간다
- GAB = 가봉
- LTU = 리투아니아
- GTM = 과테말라
- MUS = 모리셔스공화국
- VNM = 베트남
- BGD = 방글라데스
- GNB = 기니비사우
- AND = 안다우스
- BRN = 브루나이
나라별 도서평균평점을 가장 높게 준 나라는 순서대로 우간다이며 가봉 리투아니아 과테말라가 뒤를 잇는다.
