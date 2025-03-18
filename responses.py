import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# V241110 - PRE: What is it that R likes about Democratic Presidential candidate [text]
df_likes_dcan = pd.read_excel('/Users/yipho/anes/anes_open2025.xlsx', sheet_name='V241110')
df_likes_dcan['cleaned_response'] = df_likes_dcan['V241110 - PRE: What is it that R likes about Democratic Presidential candidate [text]'].fillna('').astype(str)

# V241112 - PRE: What is it that R dislikes about Democratic Presidential candidate [text]
df_dislikes_dcan = pd.read_excel('/Users/yipho/anes/anes_open2025.xlsx', sheet_name='V241112')
df_dislikes_dcan['cleaned_response'] = df_dislikes_dcan['V241112 - PRE: What is it that R dislikes about Democratic Presidential candidate [text]'].fillna('').astype(str)

# V241114 - PRE: What is it that R likes about Republican Presidential candidate [text]
df_likes_rcan = pd.read_excel('/Users/yipho/anes/anes_open2025.xlsx', sheet_name='V241114')
df_likes_rcan['cleaned_response'] = df_likes_rcan['V241114 - PRE: What is it that R likes about Republican Presidential candidate [text]'].fillna('').astype(str)

# V241116 - PRE: What is it that R dislikes about Republican Presidential candidate [text]
df_dislikes_rcan = pd.read_excel('/Users/yipho/anes/anes_open2025.xlsx', sheet_name='V241116')
df_dislikes_rcan['cleaned_response'] = df_dislikes_rcan['V241116 - PRE: What is it that R dislikes about Republican Presidential candidate [text]'].fillna('').astype(str)

# V241170 - PRE: What does R like about Democratic party [text]
df_likes_dparty = pd.read_excel('/Users/yipho/anes/anes_open2025.xlsx', sheet_name='V241170')
df_likes_dparty['cleaned_response'] = df_likes_dparty['V241170 - PRE: What does R like about Democratic party [text]'].fillna('').astype(str)

# V241172 - PRE: What does R dislike about the Democratic party [text]
df_dislikes_dparty = pd.read_excel('/Users/yipho/anes/anes_open2025.xlsx', sheet_name='V241172')
df_dislikes_dparty['cleaned_response'] = df_dislikes_dparty['V241172 - PRE: What does R dislike about the Democratic party [text]'].fillna('').astype(str)

# V241174 - PRE: What does R like about Republican party [text]
df_likes_rparty = pd.read_excel('/Users/yipho/anes/anes_open2025.xlsx', sheet_name='V241174')
df_likes_rparty['cleaned_response'] = df_likes_rparty['V241174 - PRE: What does R like about Republican party [text]'].fillna('').astype(str)

# V241176 - PRE: What does R dislike about the Republican party [text]
df_dislikes_rparty = pd.read_excel('/Users/yipho/anes/anes_open2025.xlsx', sheet_name='V241176')
df_dislikes_rparty['cleaned_response'] = df_dislikes_rparty['V241176 - PRE: What does R dislike about the Republican party [text]'].fillna('').astype(str)

# # bow EDA
vectorizer = CountVectorizer(stop_words='english')
# print(vectorizer.get_stop_words())
#can I change these stop words?

X_dislikes_dcan = vectorizer.fit_transform(df_dislikes_dcan['cleaned_response'])
bow_df_dl_dcan = pd.DataFrame(X_dislikes_dcan.toarray(), columns=vectorizer.get_feature_names_out())


X_likes_dcan = vectorizer.fit_transform(df_likes_dcan['cleaned_response'])
bow_df_l_dcan = pd.DataFrame(X_likes_dcan.toarray(), columns=vectorizer.get_feature_names_out())

dislikes_sum = bow_df_dl_dcan.sum().sort_values(ascending=False)
likes_sum = bow_df_l_dcan.sum().sort_values(ascending=False)
