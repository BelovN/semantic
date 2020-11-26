import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from pymystem3 import Mystem
from settings import DATA_DIR
from string import punctuation
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

mpl.use('Qt5Agg')


SW = stopwords.words('russian')

def stem_words(tokenized_data):
    stemmer = PorterStemmer()
    spell = Speller(lang='en')

    for i in range(len(tokenized_data)):
        tokenized_data[i] = stemmer.stem(tokenized_data[i])

    return tokenized_data


def load_data(path=DATA_DIR):
    data = []
    with open(DATA_DIR, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.replace('\n', '')
            data.append(line)

    return data


def clean_text(text):
    cleaned_text = ''
    for char in text:
        if char not in punctuation and not char.isdigit():
            cleaned_text += char
    return cleaned_text


def prepare_data(data):
    prepared_data = []
    mystem = Mystem()
    for text in data:
        text = text.lower()
        text = text.replace('\n', '')
        text = clean_text(text)
        tokens = mystem.lemmatize(text.lower())
        all_tokens = []
        for token in tokens:
            if token.replace(' ', '') and token not in SW and token != '\n':
                all_tokens.append(token)
        prepared_data.append(all_tokens)

    return prepared_data


def get_uniq_words(data):
    uniq_words = []
    for text in data:
        for word in text:
            if word not in uniq_words:
                uniq_words.append(word)
    return uniq_words


def get_matrix(uniq_words, data):

    count_texts = len(data)
    count_words = len(uniq_words)
    dict_data = {}
    for i in range(1, count_texts+1):
        dict_data['T'+str(i)] = [0]*count_words

    df = pd.DataFrame.from_dict(dict_data)
    df.index = uniq_words

    for word in uniq_words:
        for i in range(1, count_texts+1):
            if word in data[i-1]:
                df['T'+str(i)][word] += 1

    return df


def main():
    data = load_data()
    data_p = prepare_data(data)
    uniq_words = get_uniq_words(data_p)
    X = get_matrix(uniq_words, data_p)
    X.to_csv('matrix.csv', index=True, sep=';')

    svd = TruncatedSVD()
    points = svd.fit_transform(X.T)


    plt.scatter(points[:,0], points[:,1])

    for i in range(len(data)):
        plt.annotate(text="T"+str(i), xy=(points[i,0], points[i,1]))
    plt.show()
    # for i in range(len(data)):
    #     plt.annotate(text=uniq_words[i], xy=(points[i,0], points[i,1]))
    # plt.show()


    kmeans = KMeans(n_clusters=3)
    kmeans.fit(points)
    print(kmeans.cluster_centers_)
    y_km = kmeans.fit_predict(points)
    print(y_km)
    d = {
        'text': data,
        'cluster': y_km,
    }

    df = pd.DataFrame(data=d)
    df.to_csv('output.csv', index=False, sep=';')

    # plt.scatter(points[y_km ==0,0], points[y_km == 0,1], s=100, c='red')
    # plt.scatter(points[y_km ==1,0], points[y_km == 1,1], s=100, c='black')
    # plt.scatter(points[y_km ==2,0], points[y_km == 2,1], s=100, c='blue')
    # plt.show()

if __name__ == '__main__':
    main()
