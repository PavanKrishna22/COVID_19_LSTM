import pickle
from PIL import Image
import streamlit as st
import streamlit.components.v1 as stc



def home():
    
    st.title('Covid-19 India ') 
    st.write("Project developed to depict the COVID-19 condition in INDIA")
    # st.sidebar.title("Conditions")
    image = Image.open("images\img_1.png")
    st.image(image, use_column_width=True)
    st.markdown('<style>body{background-color: lightblue;}</style>', unsafe_allow_html=True)
    st.write('''The primary goal of this project is to predict the number of COVID-19 death cases and
estimate the death rate if the pandemic continues. To achieve this, we will employ machine
learning algorithms and techniques to analyze and model the available data. By utilizing
these predictive models, we can estimate future death rates based on the available data and
project them into potential future scenarios. These projections can assist healthcare
professionals and policymakers in understanding the potential impact of the pandemic on
healthcare systems, aiding in resource allocation and planning for the future.
To accomplish our objectives, we have structured the project into several modules. The data
collection and preprocessing module involves gathering relevant COVID-19 data from
reliable sources and ensuring its quality and consistency. The subsequent exploratory data
analysis module helps us gain insights into the data and identify underlying trends and
patterns. We then move on to the feature engineering module, where we extract
meaningful features from the data to enhance the predictive power of our models. These
features may include temporal factors, demographic information, and relevant health
indicators.
In the model selection and training module, we select appropriate machine learning
algorithms and train them on the prepared dataset. To ensure the reliability and
generalizability of our models, we fine-tune and evaluate them using various performance
metrics and cross-validation techniques. Once the models are trained and evaluated, we can
proceed to estimate future death rates based on the available data and provide projections
for potential future scenarios.''')
    st.write("")
    st.write("")
    st.write("")
    st.write("")


