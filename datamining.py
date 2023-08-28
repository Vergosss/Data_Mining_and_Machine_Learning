import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from IPython.display import display
from sklearn.cluster import KMeans
#read dataset
df = pd.read_csv('data.csv')
#general information about 
print('In the whole dataset null values of each attribute are:\n',df.isnull().sum())
print('------------------\nInformation of every attribute in the dataset:\n',df.info())
print('------------------\nDescribe every attribute in the dataset:\n',df.describe())
Countries = []
Countries = df['Entity']
Countries = list(set(Countries))
#df = df.groupby('Entity')#group data by country
print('--------------------\nDescribe deaths attribute for each group:\n',df.groupby('Entity').describe()['Deaths'])#omadopoihsh vash xoras kai ypologizei statistika gia kathe attribute mono gia thn xora ayth
#Countries = list(df['Entity'])
stats = df.groupby('Entity')[['Daily tests','Cases','Deaths']].mean()#gia kathe xora athrisma/mesos kok ton deaths
#
print('---------------------\nFor each attribute per group print its mean:\n',stats)
print('---------------------\nCountries are:\n',Countries)
###
missing_deaths_per_country = df.groupby('Entity')['Deaths'].apply(lambda x: x.isnull().sum())
missing_cases_per_country = df.groupby('Entity')['Cases'].apply(lambda x: x.isnull().sum())#poses times leipoun apo kathe xora
missing_daily_tests_per_country = df.groupby('Entity')['Daily tests'].apply(lambda x: x.isnull().sum())
###

print('-------------------\nMissing deaths per country-Missing cases per country-Missing Daily Tests per country:\n',missing_deaths_per_country,'\n',missing_cases_per_country,'\n',missing_daily_tests_per_country)
#fill missing values
df['Deaths'] = df.groupby('Entity')['Deaths'].transform(lambda x: x.fillna(x.ffill(axis=0)))
df['Deaths'] = df.groupby('Entity')['Deaths'].transform(lambda x: x.fillna(x.bfill(axis=0)))
#df['Deaths'].fillna('2.1',inplace=True)
df['Cases'] = df.groupby('Entity')['Cases'].transform(lambda x: x.fillna(x.ffill(axis=0)))
df['Cases'] = df.groupby('Entity')['Cases'].transform(lambda x: x.fillna(x.bfill(axis=0)))
df['Daily tests'] = df.groupby('Entity')['Daily tests'].transform(lambda x: x.fillna(x.ffill(axis=0)))
df['Daily tests'] = df.groupby('Entity')['Daily tests'].transform(lambda x: x.fillna(x.bfill(axis=0)))
#check missing values now
missing_deaths_per_country = df.groupby('Entity')['Deaths'].apply(lambda x: x.isnull().sum())
missing_cases_per_country = df.groupby('Entity')['Cases'].apply(lambda x: x.isnull().sum())
missing_daily_tests_per_country = df.groupby('Entity')['Daily tests'].apply(lambda x: x.isnull().sum())
print('-------------------\nMissing values now per country:\n',missing_deaths_per_country,missing_cases_per_country,missing_daily_tests_per_country)
print('-------------------\n',df)

columns_to_drop = ['Date','Daily tests','Cases','Deaths','Continent','Entity']

death_rate_per_country = df.groupby('Entity').apply(lambda x: 100*(x['Deaths'].max()/x['Population'].min().astype(float)))
#
print('-----------------------\nDeath rate per country:\n',death_rate_per_country)
total_cases_to_date = df.groupby('Entity')['Cases'].agg('max')
print('-------------------------\nTotal cases to date: \n',total_cases_to_date)
#
#print('-----------------------\nUnique Latitude values: \n',df['Latitude'].unique().round(3))

##attributes that we will boxplot
boxplot_attributes = ['Latitude','Longitude','Average temperature per year','Hospital beds per 1000 people','Medical doctors per 1000 people','GDP/Capita','Population','Median age','Population aged 65 and over (%)']

#####
print('----------------------------\n')
#print('25% Q1 quantile of latitude with linear interpolation is : \n',df['Latitude'].quantile(0.25,interpolation='linear'))#h python kai sygkekrimena h describe kanei linear interpolation gi ayto vgenoun 'lathos'. Gia na paro ton klasiko orismo ton quantiles vazo orisma to midpoint. Ta count,min,max,median,mean DEN EPIREAZONTAI
#an de valo tipota h valo interpolation = 'linear' vgenei to idio 
#ftiaxno boxplots ton epanalamvanomenon timon xaraktistikon
#

for column in df.drop(['Entity','Continent','Date','Daily tests','Cases','Deaths'],axis=1):#rixno aytes pou leo kato giati den tis vazo
	plt.figure()
	df.boxplot([column])#h prasini grammi deixnei th diameso ton timon.
#den exei simasia an einai grouped h oxi giati kathe xora logo imerominion exei ton idio arithmo pleiadon kai kathe xoras timi einai idia px lat
#to kouti einai to interquantile range (Q3-Q1). Oi koukides einai oi outliers(akraies times) eno oi mayres grammes ta oria ton whisker
	plt.show()
#o logos pou kano boxplot se ayta ta attributes einai giati(imerominies kai entity kai continent den exei noima) 
#eno ta deaths cases daily tests allazoun
#ta ypoloipa einai peripou idia ston arithmo se kathe xora opote eykola analiontai se boxplot
#prosoxi: den exoun oles oi xores apo 25-02 eos 28-02 opote den einai ENTELOS isaposes oi pleiades kathe xoras alla einai poli konta opote
#logika den tha epireasoun toso tragika ta boxplots(de tha apokleinoun para poli).


#ftiaxnei histogrammata vgazontas ta idia attributes afairontas tis diples+ times. gi ayto kai sta perisotera vgainoun counts=1
#an ta hist einai poli 'pykna' times pou einai kontines px 20 me 22 tha vgainoun 'mazi' sto oliko count
df.drop(['Entity','Continent','Date','Daily tests','Cases','Deaths'],axis=1).drop_duplicates().hist(bins=15,figsize=(16,9),rwidth=0.8)
plt.show()
df.drop(['Entity','Continent','Date','Daily tests','Cases','Deaths'],axis=1).hist(bins=15,figsize=(16,9),rwidth=0.8)
plt.show()#gi ayto kai kano drop duplicates th mia tail(1) thn allh klp
#an den kano drop duplicates tha ayksithoun oi times tous ston aksona y(ta counts tous-to plithos tous).
##Sysxetisi xaraktiristikon apeikonismeno se heatmap

df_corr = df.groupby('Entity').tail(1).drop(['Entity','Continent','Date'],axis=1) #rikse tis mi arithmitikes stiles
#gia kathe xora pare th teleytaia kataxorisi-pleiada
#print(df.groupby('Entity').tail(1))#teleytaia pleiada kathe group

#perno 1 pleiada ana group esto teleytaia. ayth gia ta gnorismata latitude-population tha vgazei idio r.
#gia ta attributes deaths klp tha vgazei alla epeidh einai diaforetika se kathe grammh.
#antistixa kai prin me ta boxplot [...poles..]=[..liges..]
#pezei rolo h seira logo ton zeygarion. ta athrismata kai ta athrismata tetragonon den epireazontai.
#prin eixa df_corr = df.drop(...) gia ta idia attributes evgenan idia ta r. logw isou plithous
#to mitroo sysxetisis ypologizetai vash tou pearson 
sns.heatmap(df_corr.corr(),annot=True,cmap=plt.cm.Reds)
plt.show()

##

#coefficient,colormap=kokkino,annot=True oi times ton r grafontai pano sta kelia.

#o coefficient pernei timi apo -1 eos +1 . r=0 kamia sxesh . px ipsos anthropou me polemo oukranias
#timi pros to +1 deixnei isxyrh thetiki sysxetisi px oso pio psilos toso pio varys(ipsos-varos)
#timi pros to -1 deixnei isxyrh arnitiki sysxetisi px oso anevainei to ipsos h piesh mionetai(ipsos-piesi) 

df_temp = df.loc[(df['Date']>'2020-02-25') & (df['Date']<= '2020-10-13')] 
countries = ['Greece','Cyprus','Slovenia','Croatia']#countries to test the plot
#an plotaro deaths kai cases epeidh ayta exoun mia 'syberifora' - mia ekselksi ana imera ta perno ola mazi kai oxi ena opos pano me ta alla attrs
#plotaro ta krousmata kai tous thanatous 
for output_variable in ['Cases', 'Deaths']:#kai gia ta krousmata kai tous thanatous
    fig, ax = plt.subplots(figsize=(10, 6))#h subplot epistrefei ena Figure kai enan Axes. Ousiastika dio orismata rows-columns-10x6
    #fig: to layout olo to subplot,axes o orizontios axonas
    #key: kathe monadiki xora sto grouped dataset,grp: to group me tis eggrafes px key=Greece gia to grp=eggrafes tou greece
    #me label=key diaforopoiountai xromatika sto idio plot oi xores(afou to key einai xora)
    for key, grp in df_temp[df_temp['Entity'].isin(countries)].groupby(['Entity']):
        ax = grp.plot(ax=ax, kind='line', x='Date', y=output_variable, label=key)#kind: line -to eidos tis grafikis me grammes
        #x,y oi aksones(os times). ax=ax os aksonas tou current figure orizo ton ax apo pano(proigoumeni epanalipsi) ara oles oi xores
        #gia to dothen attribute einai sto idio axes ara kai grafima
        #grp afou kaloume ti plot einai groupedby dataframe
        #pernei ton axes tou proigoumenou tou oste na emfanizontai ola sto idio figure
        #epomeno attribute px cases dimiourgo allo figure ara kai allo axes eksou kai to neo parathyro kai outo kathe eksis.
    plt.legend(loc='best')#to legend sth pyplot perigrafei se ena mikro plaisio kathe diaforetiko xroma/info kathe grammhs
    #to location=best vazei to legend ekei pou yperkalypttei ligotero th grafikh parastash 
    plt.xticks(rotation=90)#gyrizo 90 mires tis times tou aksona x(einai imerominies opote na mhn peftoun i mia sthn allh)
    plt.ylabel(output_variable)#o aksonas y einai eite oi thanatoi eite ta krousmata
    plt.show()#emfanisi ti grafiki parastasi
    
#print(df_temp[df_temp['Entity'].isin(countries)])# apo tis eggrafes pano apo tin tade imeromhnia typose aytes mono pou exoun xora sth pano lista
#h plot pernei ta orismata:
#print(df_temp[df_temp['Entity'].isin(countries)].groupby('Entity'))
df_last = df.groupby('Entity').tail(1).drop(['Entity','Date'],axis=1) #rikse tis mi arithmitikes stiles
continents = df_last['Continent'].unique()#afou to df_last einai 1 pleiada ana xora tote apla pernoume tin ipiro kathe xoras(exei duplicates)
print(continents)


for column in df_last.columns:
    fig, ax = plt.subplots(ncols=2, figsize=(14, 4))# ftiakse ena figure kai axes 14x4-gia cols=2 afou 2 einai ta subplots ara tha paei aristera-			#>deksia. an evaza nrows=2 tha tane pano pros kato. ara epistrefei 2 axes
    df_last.plot.scatter(x=column, y='Cases', ax=ax[0])#kane plot scatter me orizontio aksona tin antistixi stili-kai katakoryfo Cases-1os axes
    df_last.plot.scatter(x=column, y='Deaths', ax=ax[1])#kane plot scatter me orizontio aksona tin antistixi stili-kai katakoryfo deaths-2os axes
    #se grouped by dataframe kano .plot.scatter
    plt.show()
    if column == 'Continent':
        fig.autofmt_xdate(rotation=90)#an eisai stin hpeiro gyrna tis times tou katheta epeidh einai onomata gia na xorane
