import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle
import csv
import altair as alt
import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re, string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

# Load models
word_tfidf = joblib.load("models/word_tfidf_vectorizer.pkl")
char_tfidf = joblib.load("models/char_tfidf_vectorizer.pkl")
lr_toxic = joblib.load("models/toxic.pkl")
lr_severe = joblib.load("models/severe_toxic.pkl")
lr_obscene = joblib.load("models/obscene.pkl")
lr_threat = joblib.load("models/threat.pkl")
lr_insult = joblib.load("models/insult.pkl")
lr_identity = joblib.load("models/identity_hate.pkl")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join(text.split())
    return text

# Function to lemmatize text
def word_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in words])
from googletrans import Translator
def translate_text(text, dest_language='en'):
    translator = Translator()
    try:
        translated = translator.translate(text, dest=dest_language)
        return translated.text
    except Exception as e:
        return text
# Function to predict toxicity
def predict_toxicity(comment):
    # Translate comment to English
    comment = translate_text(comment, dest_language='en')

    cleaned_text = clean_text(comment)
    processed_text = word_lemmatizer(cleaned_text)
    
    word_features = word_tfidf.transform([processed_text])
    char_features = char_tfidf.transform([processed_text])
    all_features = hstack([word_features, char_features])
    
    pred_toxic = np.round(lr_toxic.predict_proba(all_features)[:,1], 2)*100
    pred_severe_toxic = np.round(lr_severe.predict_proba(all_features)[:,1], 2)*100
    pred_obscene = np.round(lr_obscene.predict_proba(all_features)[:,1], 2)*100
    pred_threat = np.round(lr_threat.predict_proba(all_features)[:,1], 2)*100
    pred_insult = np.round(lr_insult.predict_proba(all_features)[:,1], 2)*100
    pred_identity = np.round(lr_identity.predict_proba(all_features)[:,1], 2)*100
    
    toxicity_scores = [pred_toxic, pred_severe_toxic, pred_obscene, pred_threat, pred_insult, pred_identity]
    max_score = max(toxicity_scores)
    if max_score < 20:
        return "Non-Toxic"
    else:
        max_index = toxicity_scores.index(max_score)
        #getting the label of the max score
        labels = ["Toxic", "Severe Toxic", "Obscene", "Threat", "Insult", "Identity Hate"]
        toxicity_label = labels[max_index]
        return toxicity_label
import emoji
import os
from googleapiclient.discovery import build
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import numpy as np
# api key
api_key = 'AIzaSyDYEeSTrT7pPpVzpmaJ491gxogVxfWwpvM'
#load model and tokenizer
from tensorflow.keras.models import load_model
model1=load_model('sentiment_model_lstm.h5')
import pickle
with open('tokenizer_lstm.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)
def video_comments(video_id):
    # Create CSV file for storing comments and attributes
    with open('video_comments.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Author', 'Comment', 'Likes', 'Timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header row to CSV file
        writer.writeheader()

        # creating youtube resource object
        youtube = build('youtube', 'v3', developerKey=api_key)

        # retrieve youtube video results
        video_response = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id
        ).execute()

        # iterate video response
        while video_response:
            # extracting required info from each result object
            for item in video_response['items']:
                # Extracting comments
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                # Counting number of likes for the comment
                likes = item['snippet']['topLevelComment']['snippet']['likeCount']
                # Timestamp of the comment
                timestamp = item['snippet']['topLevelComment']['snippet']['publishedAt']
                # Author name
                author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']

                # Write data to CSV file
                writer.writerow({'Author': author, 'Comment': comment, 'Likes': likes, 'Timestamp': timestamp})

                # Empty reply list
                replies = []

            # Again repeat
            if 'nextPageToken' in video_response:
                video_response = youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    pageToken=video_response['nextPageToken']
                ).execute()
            else:
                break
#test loaded model
def SentimentAnalysis1(text):
    sentece = [text]
    tokenized_sentence = tokenizer1.texts_to_sequences(sentece)
    input_sequence = pad_sequences(tokenized_sentence, maxlen=32, padding='pre')
    prediction_ = model1.predict(input_sequence)
    print(prediction_)
    prediction = prediction_.argmax()
    print(prediction)
    if prediction == 0:
        return "Negative"
    elif prediction == 1:
        return "Neutral"
    else:
        return "Positive"

with st.sidebar:
    st.image('https://static.vecteezy.com/system/resources/thumbnails/019/787/023/small/skull-and-bones-warning-sign-on-transparent-background-free-png.png', use_column_width=True)
    select = option_menu(
        "",
        ['Home',"Toxic Analysis"],
        icons=['house-door','youtube'],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "0", "background-color": "white"}, 
            "icon": {"color": "black", "font-size": "20px"},    
            "nav-link": {
                "font-size": "16px",
                "margin": "0px",
                "color": "black",                                          
            },   
            "nav-link-selected": {
                "background-color": "orange", 
                "color": "white",                           
            },
        },
    )
if select == 'Home':
    st.markdown(
    """
    <style>
    /* Apply background image to the main content area */
    .main {
        background-image: url('https://png.pngtree.com/thumb_back/fh260/background/20240919/pngtree-a-background-of-orange-blue-and-yellow-gradients-with-gritty-appearance-image_16233934.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    col1,col2,col3=st.columns([2,6,2])
    col2.markdown('<h1 style="text-align: center; color: red; font-size: 100px;">TOXIC COMMENT 𝐀𝐧𝐚𝐥𝐲𝐬𝐢𝐬</h1>', unsafe_allow_html=True)
elif select == 'Toxic Analysis':
    st.markdown(f"<h1 style='text-align: center; color:green;'>𝓽𝓸𝔁𝓲𝓬 𝓒𝓸𝓶𝓶𝓮𝓷𝓽 𝓐𝓷𝓪𝓵𝔂𝓼𝓲𝓼</h1>", unsafe_allow_html=True)
    url=st.text_input('Enter the URL of the video',placeholder='https://www.youtube.com/watch?v=video_id')
    if url:
        try:
            col1,col2,col3=st.columns([2.5,3,1])
            if col2.button('Submit',type='primary'):
                video_id = url.split('=')[1] if '=' in url else url.split('/')[-1]
                video_comments(video_id)
                df=pd.read_csv('video_comments.csv')
                data = []
                with open('video_comments.csv', 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        # Check if 'Likes', 'Timestamp', and 'Author' are not null
                        if row['Likes'] and row['Timestamp'] and row['Author']:
                            data.append(row)
                
                # Remove duplicates based on 'Author', 'Comment', and 'Timestamp'
                unique_data = [dict(tupleized) for tupleized in set(tuple(item.items()) for item in data)]

                # Save the cleaned data to a new CSV file
                fieldnames = ['Author', 'Comment', 'Likes', 'Timestamp']
                with open('cleaned_video_comments.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(unique_data)
                data = pd.read_csv("cleaned_video_comments.csv")
                data["Toxicity"] = data["Comment"].apply(predict_toxicity)

                # Display table in Streamlit
                st.write(data)
                col1,col2=st.columns([1,1])
                # Display pie chart
                toxicity_counts = data['Toxicity'].value_counts()
                pie_chart = alt.Chart(toxicity_counts.reset_index()).mark_arc().encode(
                    theta=alt.Theta(field='Toxicity', type='quantitative'),
                    color=alt.Color(field='index', type='nominal'),
                    tooltip=['index', 'Toxicity']
                ).properties(
                    title="Toxicity Distribution"
                )
                col1.altair_chart(pie_chart, use_container_width=True)

                # Display word cloud of comments in col2
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt

                from PIL import Image
                from io import BytesIO
                from wordcloud import WordCloud
                from wordcloud import STOPWORDS
                import matplotlib.pyplot as plt
                from io import BytesIO
                from PIL import Image

                # Create a word cloud from the comments
                all_comments = " ".join(data["Comment"].astype(str).tolist())
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=200).generate(all_comments)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.tight_layout()
                # Save the word cloud image to a BytesIO object
                wordcloud_image = BytesIO()
                plt.savefig(wordcloud_image, format='png')
                plt.close()
                wordcloud_image.seek(0)
                # Display the word cloud image in Streamlit
                col2.image(wordcloud_image, caption='Word Cloud of Comments', use_column_width=True)
                
                # year wise distribution of comments
                data['Timestamp'] = pd.to_datetime(data['Timestamp'])
                data['Year'] = data['Timestamp'].dt.year
                year_counts = data['Year'].value_counts().reset_index()
                year_counts.columns = ['Year', 'Count']
                year_counts = year_counts.sort_values(by='Year')
                # Create a bar chart for year-wise distribution
                bar_chart = alt.Chart(year_counts).mark_bar().encode(
                    x=alt.X('Year:O', title='Year'),
                    y=alt.Y('Count:Q', title='Number of Comments'),
                    tooltip=['Year', 'Count'],
                    color=alt.Color('Year:N', scale=alt.Scale(domain=year_counts['Year'].tolist(), range=alt.Undefined))
                ).properties(
                    title="Year-wise Distribution of Comments"
                )
                col1.altair_chart(bar_chart, use_container_width=True)

                # Display the sentiment analysis results
                sentiment_results = []
                for comment in data['Comment']:
                    sentiment = SentimentAnalysis1(comment)
                    sentiment_results.append(sentiment)
                data['Sentiment'] = sentiment_results
                sentiment_counts = data['Sentiment'].value_counts()
                sentiment_counts = sentiment_counts.reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                sentiment_counts = sentiment_counts.sort_values(by='Count', ascending=False)
                # Create a bar chart for sentiment analysis results
                sentiment_chart = alt.Chart(sentiment_counts).mark_bar().encode(
                    x=alt.X('Sentiment:O', title='Sentiment'),
                    y=alt.Y('Count:Q', title='Number of Comments'),
                    tooltip=['Sentiment', 'Count'],
                    color=alt.Color('Sentiment:N', scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['green', 'orange', 'red']))
                ).properties(
                    title="Sentiment Analysis Results"
                )
                col2.altair_chart(sentiment_chart, use_container_width=True)
                # Display the sentiment analysis results in a table



                
        except Exception as e:
            st.write(e)
            pass
            
    else:
        st.image('https://blog.happyfox.com/wp-content/uploads/2020/08/Sentiment-Analysis-option-3.png',use_column_width=True)