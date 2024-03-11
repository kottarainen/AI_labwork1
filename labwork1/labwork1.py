import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv("news_articles.csv", encoding='latin1')  

# Remove null values, if any
data.dropna(inplace=True)

# Lowercase the text
data['processed_text'] = data['Article'].apply(lambda x: x.lower())

# Tokenization
data['processed_text'] = data['processed_text'].apply(word_tokenize)

# Remove punctuation
data['processed_text'] = data['processed_text'].apply(lambda x: [word for word in x if word.isalnum()])

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['processed_text'] = data['processed_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Stemming
stemmer = PorterStemmer()
data['processed_text'] = data['processed_text'].apply(lambda x: [stemmer.stem(word) for word in x])

# Join the tokens back into text
data['processed_text'] = data['processed_text'].apply(lambda x: ' '.join(x))

# Display the preprocessed data
#print(data['processed_text'].head())

# TF-IDF keyword extraction
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_text'])

# Get feature names (terms)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Extract top keywords for each document
num_keywords = 5  # Number of keywords to extract for each document

top_keywords_per_document = []
for i in range(len(data)):
    document_vector = tfidf_matrix[i]
    # Sort indices by TF-IDF scores in descending order
    top_indices = document_vector.toarray().argsort()[0][::-1][:num_keywords]
    # Get top keywords based on indices
    top_keywords = [feature_names[idx] for idx in top_indices]
    top_keywords_per_document.append(top_keywords)

# Add top keywords to DataFrame
data['top_keywords'] = top_keywords_per_document

# Display preprocessed data with top keywords
#print(data[['processed_text', 'top_keywords']].head())

# Convert top keywords to TF-IDF vectors
tfidf_vectorizer_keywords = TfidfVectorizer(max_features=1000)
tfidf_matrix_keywords = tfidf_vectorizer_keywords.fit_transform(data['top_keywords'].apply(' '.join))

# Perform K-means clustering
num_clusters = 5  # Number of clusters/groups
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(tfidf_matrix_keywords)

# Display articles grouped by cluster
# for cluster_id in range(num_clusters):
#     print(f"Cluster {cluster_id}:")
#     cluster_articles = data[data['cluster'] == cluster_id]['Article'].head()
#     for article in cluster_articles:
#         print(f"- {article}")
#     print()
    
def search_articles(query, data, tfidf_vectorizer, tfidf_matrix):
    # Preprocess the query
    query = query.lower()
    query_tokens = nltk.word_tokenize(query)
    query_tokens = [stemmer.stem(word) for word in query_tokens if word.isalnum() and word not in stop_words]
    query_text = ' '.join(query_tokens)

    # Convert the query to a TF-IDF vector
    query_vector = tfidf_vectorizer.transform([query_text])

    # Calculate the similarity between query vector and article vectors using dot product
    similarities = tfidf_matrix.dot(query_vector.T).toarray().flatten()

    # Sort articles by similarity and get the top results
    top_indices = similarities.argsort()[::-1][:5]  # Get top 5 most similar articles
    top_articles = data.iloc[top_indices]

    return top_articles

#Example usage:
# query = "oil"
# top_articles = search_articles(query, data, tfidf_vectorizer, tfidf_matrix)
# print(top_articles[['Article', 'cluster']])

def search_interface(data, tfidf_vectorizer, tfidf_matrix):
    while True:
        # Get user input
        query = input("Enter your search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            print("Exiting search interface...")
            break

        # Perform search
        top_articles = search_articles(query, data, tfidf_vectorizer, tfidf_matrix)

        # Display search results
        if len(top_articles) > 0:
            print(f"Search Results for query '{query}':")
            for idx, article in top_articles.iterrows():
                print(f"- {article['Article']}")
        else:
            print("No matching articles found.")
        print()

# Example usage:
search_interface(data, tfidf_vectorizer, tfidf_matrix)



