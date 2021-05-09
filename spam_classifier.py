import random
import spacy
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import string
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from spacy.util import minibatch
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

dataset = "spamham_dataset/spam.csv"

stopwords = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

def data_clean(text):
    text_without_sw = []
    remove_punct = text.translate(str.maketrans('', '', punctuation))
    tokenize = nltk.tokenize.word_tokenize(remove_punct)
    for word in tokenize:
        if word not in stopwords:
            text_without_sw.append(word)

    without_sw = ' '.join(text_without_sw)
    return without_sw



def train_model(model, train_data, optimizer, batch_size, epochs=10):
    losses = {}
    random.seed(1)

    for epoch in range(epochs):
        random.shuffle(train_data)

        batches = minibatch(train_data, size=batch_size)
        for batch in batches:
            texts, labels = zip(*batch)
            #print(texts)
            model.update(texts, labels, sgd=optimizer, losses=losses)

        print("Loss: {}".format(losses['textcat']))

    return losses['textcat']


def predict_label(model, texts):
    docs = [model.tokenizer(text) for text in texts]

    textcat = model.get_pipe('textcat')
    scores, _ = textcat.predict(docs)

    predicted_labels = scores.argmax(axis=1)
    predicted_class = [textcat.labels[label] for label in predicted_labels]

    return predicted_class


def main():
    data = pd.read_csv(dataset, usecols=[0, 1], names=('label', 'text'), header=0)
    print(data.head())
    rows = len(data.index)
    print("DataFrame size: {}".format(rows))
    print(data.label.value_counts())
    print(data.label.value_counts() / rows * 100.0)

    data['cleaned_text'] = data['text'].apply(lambda x: data_clean(x))

    nlp = spacy.blank("en")

    text_class = nlp.create_pipe(
        "textcat",
        config={
            "exclusive_classes": True,
            "architecture": "bow"})

    nlp.add_pipe(text_class)

    text_class.add_label("ham")
    text_class.add_label("spam")

    x_train, x_test, y_train, y_test = train_test_split(data['cleaned_text'], data['label'], test_size=0.33, random_state=7)

    train_labels = [{'cats': {'ham': label == 'ham',
                              'spam': label == 'spam'}} for label in y_train]

    test_labels = [{'cats': {'ham': label == 'ham',
                             'spam': label == 'spam'}} for label in y_test]

    train_data = list(zip(x_train, train_labels))
    test_data = list(zip(x_test, test_labels))

    optimizer = nlp.begin_training()
    batch_size = 5
    epochs = 10

    train_model(nlp, train_data, optimizer, batch_size, epochs)

    train_predictions = predict_label(nlp, x_train)
    test_predictions = predict_label(nlp, x_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print("Accuracy of Train data: {}".format(train_accuracy))
    print("Accuracy of Test data: {}".format(test_accuracy))

    train_cf = confusion_matrix(y_train, train_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(train_cf, annot=True, fmt='d')

    test_cf = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cf, annot=True, fmt='d')


if __name__ == "__main__":
    main()
