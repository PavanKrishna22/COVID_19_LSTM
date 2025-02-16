import pickle
from PIL import Image
import streamlit as st
import streamlit.components.v1 as stc

def precaution():
    st.title('How Can You Protect Yourself From COVID-19 ?') 
    
    image = Image.open("images\prec1.png")
    st.image(image, use_column_width=True)
    st.write('\n\n\n\n\n\n')
    st.write(''' **Protecting ourselves from COVID-19 is crucial for several reasons:**

1. Personal Health: COVID-19 is a highly contagious respiratory illness that can cause a range of symptoms from mild to severe, and in some cases, it can be fatal. By protecting ourselves from COVID-19, we reduce the risk of contracting the virus and experiencing the associated health complications.

2. Preventing Transmission: Even if we do not develop severe symptoms, we can still transmit the virus to others, including those who are more vulnerable, such as older adults and individuals with underlying health conditions. By protecting ourselves, we also help protect those around us and contribute to preventing the further spread of the virus.

3. Overwhelming Healthcare Systems: The rapid spread of COVID-19 can put a strain on healthcare systems, leading to overcrowded hospitals and limited resources. By taking measures to protect ourselves, such as wearing masks and practicing good hygiene, we can help alleviate the burden on healthcare systems and ensure that those in need receive proper care.

4. Emergence of Variants: The virus that causes COVID-19 has the potential to mutate and give rise to new variants. Some variants have shown increased transmissibility or resistance to certain treatments. By minimizing the spread of the virus through protective measures, we can also reduce the likelihood of new variants emerging.

5. Public Health and Global Control: Controlling the spread of COVID-19 is essential for public health and the overall well-being of communities. By following guidelines and taking precautions, we contribute to the collective effort in mitigating the impact of the pandemic and working towards its eventual control.

It's important to note that protecting ourselves from COVID-19 should be done in conjunction with other recommended measures, such as vaccination, social distancing, and following local health guidelines. ''')
    st.write('\n\n\n\n\n\n')
    st.write('\n\n\n\n\n\n')

    st.subheader(''' Some Precautions That Can Be Taken Are: \n\n''')
    col1, mid, col2 = st.columns([20,15,20])
    with col1:
        st.image('images/1.png', width=200)
    with col2:
        st.image('images/2.png', width=200)
    st.write('\n\n\n\n\n\n')
    col1, mid, col2 = st.columns([20,15,20])
    with col1:
        st.image('images/3.png', width=200)
    with col2:
        st.image('images/4.png', width=200)
    st.write('\n\n\n\n\n\n')   
    col1, mid, col2 = st.columns([20,15,20])
    with col1:
        st.image('images/5.png', width=200)
    with col2:
        st.image('images/6.png', width=200)


