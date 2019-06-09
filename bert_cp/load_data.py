import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()

df = pd.read_csv("data/train.csv", sep=';', encoding='utf8') #En cas d'erreur essayez avec d'autres encodings

# Crée les DataFrames train et dev dont BERT aura besoin, en ventillant 1 % des données dans test
df_bert = pd.DataFrame({'user_id': df['ID'], 'label': le.fit_transform(df['intention']), 'alpha': ['a']*df.shape[0], 'text':df['question'].replace(r'\n',' ',regex=True)})
df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.25)

# Crée la DataFrame test dont BERT aura besoin
df_test = pd.read_csv("data/test.csv", sep=',', encoding='utf8') #En cas d'erreur essayez avec d'autres encodings
df_bert_test = pd.DataFrame({'user_id': df_test['ID'], 'text': df_test['question'].replace(r'\n',' ',regex=True)})

print(df_bert_train.dtypes, df_bert_test.dtypes)
df_bert_train['label'] = df_bert_train['label'].astype(int)
print(df_bert_train.dtypes, df_bert_test.dtypes)
# Enregistre les DataFrames au format .tsv (tab separated values) comme BERT en a besoin
df_bert_train.to_csv('data/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
df_bert_test.to_csv('data/test.tsv', sep='\t', index=False, header=True)

