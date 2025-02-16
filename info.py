import pickle
from PIL import Image
import streamlit as st
import streamlit.components.v1 as stc

def info():
    st.title('What is COVID-19 ?') 
    
    #image = Image.open("images\info1.png")
    #st.image(image, use_column_width=True)

    video_file = open('C:/Users/krish/Desktop/RUBY_DA-2-3/images/info_vid.mp4', 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes)

    st.markdown('<style>body{background-color: lightblue;}</style>', unsafe_allow_html=True)
    st.markdown("""<style>.big-font {font-size:25px !important;}</style>""", unsafe_allow_html=True)

    st.markdown('<p class="big-font"><b>Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus</b></p>', unsafe_allow_html=True)
    st.write('''
Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age. 

The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Protect yourself and others from infection by staying at least 1 metre apart from others, wearing a properly fitted mask, and washing your hands or using an alcohol-based rub frequently. Get vaccinated when it’s your turn and follow local guidance.

The virus can spread from an infected person’s mouth or nose in small liquid particles when they cough, sneeze, speak, sing or breathe. These particles range from larger respiratory droplets to smaller aerosols. It is important to practice respiratory etiquette, for example by coughing into a flexed elbow, and to stay home and self-isolate until you recover if you feel unwell.''')
    
    st.title(" Symptoms ")
    st.write(''' 
Signs and symptoms of coronavirus disease 2019 (COVID-19) may appear 2 to 14 days after exposure. This time after exposure and before having symptoms is called the incubation period. You can still spread COVID-19 before you have symptoms (presymptomatic transmission). Common signs and symptoms can include:
\n
Fever
\n
Cough
\n
Tiredness
\n
\n
Early symptoms of COVID-19 may include a loss of taste or smell.

Other symptoms can include:
\n
Shortness of breath or difficulty breathing
\n
Muscle aches \n
Chills \n
Sore throat \n
Runny nose \n
Headache \n
Chest pain \n
Pink eye (conjunctivitis) \n
Nausea \n
Vomiting \n
Diarrhea \n
Rash \n
This list isn't complete. Children have similar symptoms to adults and generally have mild illness.
\n
The severity of COVID-19 symptoms can range from very mild to severe. Some people may have only a few symptoms. Some people may have no symptoms at all, but can still spread it (asymptomatic transmission). Some people may experience worsened symptoms, such as worsened shortness of breath and pneumonia, about a week after symptoms start.

Some people experience COVID-19 symptoms for more than four weeks after they're diagnosed. These health issues are sometimes called post-COVID-19 conditions. Some children experience multisystem inflammatory syndrome, a syndrome that can affect some organs and tissues, several weeks after having COVID-19. Rarely, some adults experience the syndrome too.

People who are older have a higher risk of serious illness from COVID-19, and the risk increases with age. People who have existing medical conditions also may have a higher risk of serious illness. Certain medical conditions that may increase the risk of serious illness from COVID-19 include:
\n
Serious heart diseases, such as heart failure, coronary artery disease or cardiomyopathy
\n Cancer
\n Chronic obstructive pulmonary disease (COPD)
\n Type 1 or type 2 diabetes
\n Overweight, obesity or severe obesity
\n High blood pressure
\n Smoking
\n Chronic kidney disease
\n Sickle cell disease or thalassemia
\n Weakened immune system from solid organ transplants or bone marrow transplants
\n Pregnancy
\n Asthma
\n Chronic lung diseases such as cystic fibrosis or pulmonary hypertension
\n Liver disease
\n Dementia
\n Down syndrome
\n Weakened immune system from bone marrow transplant, HIV or some medications
\n Brain and nervous system conditions, such as strokes
\n Substance use disorders
\n
This list is not complete. Other medical conditions may increase your risk of serious illness from COVID-19. ''')

    # adding gif ---------- st.image("my_logo.gif)
    image = Image.open("images\info2.png")
    st.image(image, use_column_width=True)