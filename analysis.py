# libraries
import streamlit as st
import pickle
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def analysis():
    st.title('Analysis Of Covid-19')

        # importing of the DATASET
    @st.cache_data
    def load_data():
        df = pd.read_csv("covid_19_state_wise_data.csv")
        return df

    df = load_data()

    state_select = st.sidebar.selectbox('state', df['state'].unique())
    visualization = st.sidebar.selectbox('Chart type', ('Bar Chart', 'Pie Chart', 'Line Chart'))
    status_select = st.sidebar.radio('Covid-19 status', ('confirmed_cases', 'active_cases', 'recovered_cases', 'death_cases'))
    # select = st.sidebar.selectbox('Covid-19 patient status',('confirmed_cases','active_cases','recovered_cases','death_cases'))
    selected_state = df[df['state'] == state_select]
    st.markdown("## **State level analysis**")


    # Visualization Part
    def get_total_dataframe(df):
        total_dataframe = pd.DataFrame({
        'Status': ['Confirmed', 'Recovered', 'Deaths', 'Active'],
        'Number of cases': (df.iloc[0]['confirmed_cases'],
                            df.iloc[0]['active_cases'],
                            df.iloc[0]['recovered_cases'], df.iloc[0]['death_cases'])})
        return total_dataframe


    state_total = get_total_dataframe(selected_state)
    if visualization == 'Bar Chart':
        state_total_graph = px.bar(state_total, x='Status', y='Number of cases',
                                   labels={'Number of cases': 'Number of cases in %s' % (state_select)}, color='Status')
        st.plotly_chart(state_total_graph)
    elif visualization == 'Pie Chart':
        if status_select == 'confirmed_cases':
            st.title("Total Confirmed Cases ")
            fig = px.pie(df, values=df['confirmed_cases'], names=df['state'])
            st.plotly_chart(fig)
        elif status_select == 'active_cases':
            st.title("Total Active Cases ")
            fig = px.pie(df, values=df['active_cases'], names=df['state'])
            st.plotly_chart(fig)
        elif status_select == 'death_cases':
            st.title("Total Death Cases ")
            fig = px.pie(df, values=df['death_cases'], names=df['state'])
            st.plotly_chart(fig)
        else:
            st.title("Total Recovered Cases ")
            fig = px.pie(df, values=df['recovered_cases'], names=df['state'])
            st.plotly_chart(fig)
    elif visualization == 'Line Chart':
        if status_select == 'death_cases':
            st.title("Total Death Cases Among states")
            fig = px.line(df, x='state', y=df['death_cases'])
            st.plotly_chart(fig)
        elif status_select == 'confirmed_cases':
            st.title("Total Confirmed Cases Among states")
            fig = px.line(df, x='state', y=df['confirmed_cases'])
            st.plotly_chart(fig)
        elif status_select == 'recovered_cases':
            st.title("Total Recovered Cases Among states")
            fig = px.line(df, x='state', y=df['recovered_cases'])
            st.plotly_chart(fig)
        else:
            st.title("Total Active Cases Among states")
            fig = px.line(df, x='state', y=df['active_cases'])
            st.plotly_chart(fig)



    # Specify the path to your dataset.csv file
    data_path = 'WHO-COVID-19-global-data.csv'

    # Read the dataset.csv file
    dff = pd.read_csv(data_path)

    # Convert the 'Date_reported' column to datetime
    dff['Date_reported'] = pd.to_datetime(dff['Date_reported'])

    # Create a Streamlit app
    st.title("COVID-19 Timeline Graph")
    st.write(dff)

    # Plot the timeline graph
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dff['Date_reported'], dff['Cumulative_cases'], marker='o', linestyle='-', color='b', label='Cumulative Cases')
    ax.plot(dff['Date_reported'], dff['Cumulative_deaths'], marker='o', linestyle='-', color='r', label='Cumulative Deaths')
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title('COVID-19 Cases and Deaths Over Time')
    ax.legend()
    plt.xticks(rotation=45)

    # Display the graph in Streamlit
    st.pyplot(fig)


    # dataset in table 
    def get_table():
        datatable = df[['state', 'confirmed_cases', 'recovered_cases', 'death_cases', 'active_cases']].sort_values(by=['confirmed_cases'], ascending=False)
        return datatable


    datatable = get_table()
    st.dataframe(datatable)


