import pandas as pd
import altair as alt
import nltk  # ← new
from nltk.corpus import stopwords as stop
from nltk.stem import PorterStemmer as stemmer
from nltk.stem import WordNetLemmatizer as lemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")


def download_models():
    """call this at the beginning of a session to install dependencies"""
    nltk.download("punkt")  # necessary for tokenization
    nltk.download("wordnet")  # necessary for lemmatization
    nltk.download("stopwords")  # necessary for removal of stop words
    nltk.download("averaged_perceptron_tagger")  # necessary for POS tagging
    nltk.download("maxent_ne_chunker")  # necessary for entity extraction
    nltk.download("words")
    nltk.download("punkt_tab")
    nltk.download("averaged_perceptron_tagger_eng")
    alt.data_transformers.enable("vegafusion")


def ingest(text_name: str) -> str:
    with open(text_name, "r") as f:
        story = f.read()
    return story


def define_stopwords() -> list:
    stopwords = stop.words("english")
    stopwords.append("chapter")
    return stopwords


def tokenize(story: str, stopwords: list) -> list:
    tokens = nltk.word_tokenize(story.lower())
    words = [word for word in tokens if word.isalpha()]
    without_stopwords = [word for word in words if word not in stopwords]
    return words, without_stopwords


def bag_of_words(words: list[str]) -> list:
    bow = {}
    for word in words:
        bow[word] = words.count(word)
    words_frequency = sorted(bow.items(), key=lambda x: x[1], reverse=True)
    return words_frequency


def create_wordcloud(story: str):
    wc = WordCloud(width=500, height=500, background_color="white").generate(story)
    # display the generated image:
    my_dpi = 72
    plt.figure(figsize=(500 / my_dpi, 500 / my_dpi), dpi=my_dpi)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def get_common_words(words_frequency: list) -> pd.DataFrame:
    # first we create a dataframe from the word frequencies
    df = pd.DataFrame(words_frequency, columns=["word", "count"])
    return df


def plot_top_words(df: pd.DataFrame):
    # we want to focus just on the top 20 words
    df_top = df[:50]
    # df_top_100 = df[:100] # use this later

    # draw horizontal barchart
    alt.Chart(df_top).mark_bar().encode(x="count:Q", y=alt.Y("word:N", sort="-x"))


def main():
    # download_models()
    stopwords = define_stopwords()
    story = ingest("docs/alice.txt")
    _, without_stopwords = tokenize(story, stopwords)
    words_frequency = bag_of_words(without_stopwords)
    create_wordcloud(story)
    df = get_common_words(words_frequency)
    df.head()
    plot_top_words(df)


main()
