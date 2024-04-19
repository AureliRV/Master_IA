import pandas
import seaborn as sns

data = pandas.read_csv('C:/Users/Aureli/Dropbox/PC (2)/Documents/Git_works_MASTER_AI/Master_IA/EDA/nba.csv', index_col='Team')
utah = data.loc['Utah Jazz']
#sns.lineplot(utah['Age'], utah['Weight'])
#sns.lineplot(data['Age'], data['Weight'])
sns.scatterplot(data['Age'], data['Weight'], hue= data['Position'])

#print(data)
#print(utah)