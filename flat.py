import pandas as pd
import altair as alt
import nltk  # ← new
from nltk.corpus import stopwords as stop

# from nltk.stem import PorterStemmer as stemmer
# from nltk.stem import WordNetLemmatizer as lemmatizer
# from nltk.corpus import wordnet

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
    df_100 = df[:100]
    # df_top_100 = df[:100] # use this later

    # draw horizontal barchart
    alt.Chart(df_top).mark_bar().encode(x="count:Q", y=alt.Y("word:N", sort="-x"))
    return df_100


def words_by_type(story: str, stopwords: list) -> str:
    # first we extract all words and their types (a.k.a. parts-of-speech or POS)
    pos = pos_tag(word_tokenize(story))

    # we will be collecting words and types in lists of the same length
    words = []
    types = []

    # iterate over all entries in the pos list (generated above)
    for p in pos:
        # get the word and turn it into lowercase
        word = p[0].lower()
        # get the word's type
        tag = p[1]

        # for this analysis we remove entries that contain punctuation or numbers
        # and we also ignore the stopwords (sorry: the, and, or, etc!)
        if word.isalpha() and word not in stopwords:
            # first we add this word to the words list
            words.append(word)
            # then we add its word type to types list, based on the 1st letter of the pos tag
            # note that we access letters in a string, like entries in a list
            if tag[0] == "J":
                types.append("Adjective")
            elif tag[0] == "N":
                types.append("Noun")
            elif tag[0] == "R":
                types.append("Adverb")
            elif tag[0] == "V":
                types.append("Verb")
            # there are many more word types, we simply subsume them under 'other'
            else:
                types.append("Other")
    return words, types


def create_pos_dataframe(
    words: list, types: list, df_top: pd.DataFrame
) -> pd.DataFrame:
    # create a dataframe from the words and types lists
    df = pd.DataFrame({"word": words, "type": types})
    index = df["word"].isin(df_top["word"])
    df_pared = df[index].reset_index(drop=True)
    return df_pared


def plot_pos_words(df_pared: pd.DataFrame):
    # along the type column, we want to support a filter selection
    selection = alt.selection(type="multi", fields=["type"])

    # we create a composite chart consisting of two sub-charts
    # the base holds it together and acts as the concierge taking care of the data
    base = alt.Chart(df_pared)

    # this shows the types, note that we rely on Altair's aggregation prowess
    chart1 = (
        base.mark_bar()
        .encode(
            x=alt.Y("type:N"),
            y=alt.X("count()"),
            # when a bar is selected, the others are displayed with reduced opacity
            opacity=alt.condition(selection, alt.value(1), alt.value(0.25)),
        )
        .add_selection(selection)
    )

    # this chart reacts to the selection made in the left/above chart
    chart2 = (
        base.mark_bar(width=5)
        .encode(
            x="word:N",
            y=alt.Y("count()"),
        )
        .transform_filter(selection)
    )

    chart1 | chart2


# def main():
#     # download_models()
#     stopwords = define_stopwords()
#     story = ingest("docs/alice.txt")
#     _, without_stopwords = tokenize(story, stopwords)
#     words_frequency = bag_of_words(without_stopwords)
#     create_wordcloud(story)
#     df = get_common_words(words_frequency)
#     df.head()
#     df = plot_top_words(df)
#     words, types = words_by_type(story, stopwords)
#     df_pared = create_pos_dataframe(words, types, df)


# main()
