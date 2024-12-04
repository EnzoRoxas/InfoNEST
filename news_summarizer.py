import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen
from newspaper import Article
import io
import nltk
import spacy
import re
import streamlit.components.v1 as components
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import pytextrank
import torch
import streamlit as st
import speech_recognition as sr
from googlenewsdecoder import new_decoderv1
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import os
import time
from spacy.cli import download
from pathlib import Path
#import spacy_streamlit
#from spacy_streamlit import load_model

# Load spaCy's English model and add PyTextRank
# Ensure the model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


nlp.add_pipe("textrank", last=True)



# Page config
st.set_page_config(page_title='INFONest🇵🇭: Get the News!📰', page_icon='./Meta/newspaper1.ico')

#Import the Youtube summarizer
from video_summarizer import run_youtube_summarizer

@st.cache_resource
def get_actual_article_link(google_news_url):
    interval_time = 5  # Specify an interval to prevent rate-limiting issues
    try:
        decoded_url = new_decoderv1(google_news_url, interval=interval_time)
        if decoded_url.get("status"):
            return decoded_url["decoded_url"]
        else:
            st.warning("Could not decode URL.")
            return None
    except Exception as e:
        st.error(f"Error occurred while retrieving the article link: {e}")
        return None

# Load essential resources with caching
@st.cache_resource
def punkt_load():
    return nltk.download('punkt')
punkt = punkt_load()


@st.cache_resource
def stopwords_load():
    nltk.download('stopwords')
    stop_words = nltk.corpus.stopwords.words("english")
    stop_words = stop_words + ['hi', 'im', 'hey']
    return stop_words

stop_words = stopwords_load()


@st.cache_resource
def bart_tokenizer_load():
    #bart_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    bart_tokenizer =AutoTokenizer.from_pretrained("Angel0J/distilbart-multi_news-12-6")
    return bart_tokenizer


#Load the bart model to GPU
@st.cache_resource
def bart_model_load():
    bart_model = AutoModelForSeq2SeqLM.from_pretrained("Angel0J/distilbart-multi_news-12-6")#, use_safetensors= True)
    #bart_model = BartForConditionalGeneration.from_pretrained("Angel0J/distilbart-multi_news-12-6", use_safetensors= True)
    return bart_model


# Load the models
loading_message = st.empty()  # Container for the "Now Loading..." message
progress = st.progress(0)  # Initialize progress bar
loading_message.markdown("**Now Loading...**", unsafe_allow_html=True)
with st.empty():  # To prevent the spinner from blocking progress updates
    # Load the BART model
    progress.progress(50)  # Set progress to 66% after loading BART model
    bart_model = bart_model_load()

    # Load the BART tokenizer
    progress.progress(100)  # Set progress to 100% after loading BART tokenizer
    bart_tokenizer = bart_tokenizer_load()
    
# Once the models are loaded, remove the progress bar
progress.empty()  # Remove the progress bar from the screen
loading_message.empty()  # Remove the "Now Loading..." message
time.sleep(1)

# Fetch news function
@st.cache_resource
def fetch_news_from_rss(url):
    op = urlopen(url)
    rd = op.read()
    op.close()
    return soup(rd, 'xml').find_all('item')

@st.cache_resource
def fetch_news_search_topic(topic):
    site = f'https://news.google.com/news/rss/search/section/q/{topic}?hl=en&gl=PH&ceid=PH%3Aen'
    return fetch_news_from_rss(site)

@st.cache_resource
def fetch_category_news(category):
    site = f'https://news.google.com/news/rss/headlines/section/topic/{category}?hl=en&gl=PH&ceid=PH%3Aen'
    return fetch_news_from_rss(site)

# Utility functions
@st.cache_data(ttl=None, max_entries=80)
def clean_text(text, stop_words):
    cleanT = re.sub(r"(@\[A-Za-z0-9]+)|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    cleanT = re.sub(r'[^a-zA-Z0-9\s.,\r\n-]+', '', cleanT)
    cleanT = re.sub(r'\s+', ' ', cleanT).strip()
    sentences = sent_tokenize(cleanT)
    return ' '.join([w for w in sentences if w.lower() not in stop_words and w])



def fetch_news_poster(poster_link):
    try:
        u = urlopen(poster_link)
        raw_data = u.read()
        image = Image.open(io.BytesIO(raw_data))
    except:
        image = Image.open('./Meta/no_image.jpg')
    st.image(image, use_column_width=True)

@st.cache_data(ttl=None, max_entries=80)
def extract_entities(text):
    if text:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    return []

def prioritize_sentences_with_entities(text, entities, top_n=5):
    # Tokenize sentences
    sentences = sent_tokenize(text)
    
    # Create a dictionary to store scores
    sentence_scores = {}
    
    for sentence in sentences:
        score = 0
        for entity, _ in entities:
            # Boost score for each entity found in the sentence
            if entity in sentence:
                score += 1
        sentence_scores[sentence] = score
    
    # Sort sentences by score
    prioritized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    
    # Return the top N sentences with the highest scores
    return prioritized_sentences[:top_n]

@st.cache_data(ttl=None, max_entries=80)
def enhanced_textrank_summarize(text, num_sentences=5):
    if text:
        # Extract entities
        entities = extract_entities(text)  # No num_sentences argument here
        
        # Prioritize sentences with entities
        prioritized_sentences = prioritize_sentences_with_entities(text, entities, top_n=num_sentences)
        
        # Use TextRank for summarization
        doc = nlp(' '.join(prioritized_sentences))
        summary = ' '.join([str(sent) for sent in doc._.textrank.summary(limit_sentences=num_sentences)])
        return summary if summary else "Summary is Not Available..."
    return "Summary is not Available..."



def count_sentences(text):
    return len(sent_tokenize(text))

@st.cache_data(ttl=None, max_entries=80)
def bart_summarize(_bart_tokenizer, text, _bart_model, num_sentences=5):
    if text:
        # Pre-clean the input text
        text = re.sub(r"[{}:\"']", "", text)  # Remove unnecessary punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces

        inputs = _bart_tokenizer([text], return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(next(_bart_model.parameters()).device) for k, v in inputs.items()}
        
        # Define max_length and min_length based on the user-specified number of sentences
        max_length = 50 * num_sentences  # Estimate 50 tokens per sentence
        min_length = 20 * num_sentences  # Estimate 20 tokens per sentence
        
        with torch.no_grad():
            summary_ids = _bart_model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                num_beams=3,
                no_repeat_ngram_size=3
            )
        
        decoded_summary = _bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        final_summary = re.sub(r"([.!?])[^.!?]*$", r"\1", decoded_summary)  # Ensure it ends on a full sentence
        return final_summary
    return "Summary is Not Available..."



# Custom extraction function
def extract_main_content(html):
    soup_obj = soup(html, 'html.parser')
    # Modify the selector based on the actual HTML structure
    main_content = soup_obj.find('div', class_='article-content')  # Adjust this selector
    return main_content.get_text(strip=True) if main_content else ""

    
# Main display function
def display_news(list_of_news, news_quantity, stop_words, bart_tokenizer, bart_model):
    for c, news in enumerate(list_of_news[:news_quantity], start=1):
        st.write(f'**({c}) {news.title.text}**')

        rss_link = news.link.text
        news_link = get_actual_article_link(rss_link)

        if not news_link:
            st.warning("Could not retrieve the article link.")
            continue

        news_data = Article(news_link)

        try:
            news_data.download()
            news_data.parse()
            raw_text = extract_main_content(news_data.html) or news_data.text
            clean_txt = clean_text(raw_text, stop_words)
        except Exception as e:
            st.error(f"Error fetching article: {e}")
            continue

        fetch_news_poster(news_data.top_image)

        with st.expander(news.title.text):
            
            num_sentences = 5

            textrank_summary = enhanced_textrank_summarize(clean_txt, num_sentences)
            bart_summary = bart_summarize(bart_tokenizer, textrank_summary, bart_model, num_sentences)

            st.markdown(f'<h6 style="text-align: justify;">{bart_summary}</h6>', unsafe_allow_html=True)
            st.markdown(f"[Read more at source]({news_link})")

        st.success(f"Published Date: {news.pubDate.text}")


def speech_to_text():
    recognizer = sr.Recognizer()
    duration = 5  # Recording duration in seconds
    sample_rate = 44100  # Sampling rate

    # Inform the user to start speaking
    mic_Msg = st.empty()
    mic_Msg.write("Please Say Your Topic...")

    # Record audio using sounddevice
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is complete
    mic_Msg.empty()

    # Convert the recorded NumPy array to AudioData for speech_recognition
    audio_bytes = audio_data.tobytes()
    audio = sr.AudioData(audio_bytes, sample_rate, 2)  # 2 is the width in bytes (16-bit audio)

    # Process the audio with speech recognition
    try:
        text = recognizer.recognize_google(audio)
        message = st.empty()
        message.write(f"Recognized: {text}")
        time.sleep(3)
        message.empty()
        return text
    except sr.UnknownValueError:
        st.write("Sorry, I could not understand the audio.")
        return ""
    except sr.RequestError:
        st.write("Could not request results from Google Speech Recognition service.")
        return

# Function to convert image to base64 format to use in the HTML img tag
def image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def run():
    stop_words = stopwords_load()

    st.title("INFONest🇵🇭: Get The News!📰")
    image = Image.open('./Meta/newspaper4.png')
    
    # Use st.empty() and markdown to center the image with a fixed width
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{}" width="400"/>
        </div>
        """.format(image_to_base64(image)),
        unsafe_allow_html=True,
    )

    # Track category selection in session state
    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = None



    # Reset session state if category changes
    category = ['--Select--', 'Top News', 'Hot Topics', 'Search', 'Video News']
    cat_op = st.selectbox('Please Select:', category)

    if st.session_state["selected_category"] != cat_op:
        st.session_state["selected_category"] = cat_op


    if cat_op in category[0:4]:  # Show for 'Top News', 'Hot Topics', and 'Search'
        with st.expander("INSTRUCTIONS: How to use INFONest!"):
            st.write("""
                NOTE: Some articles may not be loaded at all as not all websites allow for the scraping of data.
            
                1. Select a category of your choice! (i.e. Top News!, Hot Topics, Search, and Video News)
                
                2. If you pick Top News, Hot Topics, or Search, the application will load 5 of the recent and newest articles 
                based on the category chosen. (NOTE: Please check out the Video News Category to know more about it)
                
                3. The articles loaded will have their own summaries. (NOTE: Please wait as it may take time to load the articles and 
                summaries!)  

                4. Use the summaries as overview for what each of the news is about!            
            """)
    

    # If category is not selected yet, show a warning
    if cat_op == category[0]:
        st.warning('Please Select a Category!')
    elif cat_op == category[1]:
        with st.expander("PLEASE READ! : What is Top News?"):
            st.write("""
                NOTE: Please wait as the loading of the articles and summaries may take some time!

                - Top News are recent and relevant news about the Philippines gathered from different sources!

                - What it covers will be the recent developments or topics that are currently trending in the country. 

                
             """)
        st.subheader("Here Are the Top News For You!")
        no_of_news = 5  #st.slider('Number of News:', 5, 25, 10)
        news_list = fetch_news_from_rss('https://news.google.com/news/rss?hl=en&gl=PH&ceid=PH%3Aen')
        display_news(news_list, no_of_news, stop_words, bart_tokenizer, bart_model)
    elif cat_op == category[2]:
        with st.expander("PLEASE READ! : What is Hot Topics?"):
            st.write("""
                  NOTE: Please wait as the loading of the articles and summaries may take some time!

                - Hot Topics offers a selection of topics from which the user can choose from. These news can have articles 
                from different countries not just the Philippines. 

                - The topics are : WORLD, NATION, BUSINESS, TECHNOLOGY, ENTERTAINMENT, SPORTS, SCIENCE, and HEALTH. This will
                provide news articles that are about what the currently selected topic is along with summaries of each of the article.
                 
             """)
        av_topics = ['--Please Select A Topic!--', 'WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SPORTS', 'SCIENCE', 'HEALTH']
        chosen_topic = st.selectbox("Choose a Topic:", av_topics)
        
        # Initialize news_list to avoid UnboundLocalError
        news_list = []

        if chosen_topic == av_topics[0]:
            st.warning("Please select a valid topic to proceed.")
        else:
            no_of_news = 5 #st.slider('Number of News:', 5, 25, 10)
            news_list = fetch_category_news(chosen_topic)
            
        if news_list:
            st.subheader("Here are the {} News for you".format(chosen_topic))
            display_news(news_list, no_of_news, stop_words, bart_tokenizer, bart_model)
    
        
    elif cat_op == category[3]:
        with st.expander("PLEASE READ!: Instructions for Search"):
            st.write(""" 
                 NOTE: Please wait as the loading of the articles and summaries may take some time!

                 1. The Search Category allows the user for direct searching if they are looking for a specific topic.

                 2. The user can either input in the Search Bar or use the Voice button for speech to text input.

                 3. The user must have a microphone (i.e from earphones or headphones) in order to use the Voice button for 
                 the speech to text input.

                 4. To use it, simply press the Voice button and wait for the text "Please Say Your Topic..." to appear. The button
                 can be used again by pressing and speaking again. Once your input has been recognize, it will automatically search for
                 the news related to your topic.
            """)
        # Voice input button
        if st.button("Voice"):
            user_topic = speech_to_text()
            st.session_state['user_topic'] = user_topic.strip() if user_topic else ""

            # Automatically trigger search if a valid topic is recognized
            if st.session_state['user_topic']:
                news_list = fetch_news_search_topic(st.session_state['user_topic'].replace(' ', ''))
                st.session_state['search_news_list'] = news_list

        # Text input for manual topic entry
        user_topic_input = st.text_input(
            "Enter a Topic",
            value=st.session_state.get('user_topic', "")
        )

        # Slider for selecting the number of news items
        no_of_news = 5 #st.slider('Number of News:', 5, 15, 10)

        # Automatically search if the user enters text in the input box
        if user_topic_input.strip():
            st.session_state['user_topic'] = user_topic_input.strip()
            news_list = fetch_news_search_topic(st.session_state['user_topic'].replace(' ', ''))
            st.session_state['search_news_list'] = news_list
        else:
            news_list = st.session_state.get('search_news_list', [])

        # Display news or show warnings
        if news_list:
            user_topic = st.session_state.get('user_topic', "Your Topic")
            st.subheader(f"Here are some {user_topic.capitalize()} News for you")
            display_news(news_list, no_of_news, stop_words, bart_tokenizer, bart_model)
        else:
            st.warning("Please enter a topic to search.")

    elif cat_op == category[4]:  # video_summarizer
        run_youtube_summarizer()  # Call the function from youtube_summarizer.py

run()
