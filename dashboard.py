import streamlit as st
import plotly.express as px
import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

### Control Variables ###
beam_width = 12
outtage = 230

st.set_page_config(page_title="Dashboard", page_icon=":bar_chart:", layout="wide")

# Title
with st.container():
    st.title("5G Beam Prediction with Machine Learning on High Speed Train with Machine Learning Dashboard")

# Define the choices for the train tracks
tracks = list(range(1, 9))

# Create the selectbox for selecting a train track
track = st.sidebar.selectbox("Pilih Lintasan", options=tracks, index=0)

ml_method = st.sidebar.radio(
    "Select the Beam Index Prediction Method",
    ('KNN', 'Neural Networks', 'Lookup Table', 'Naive Bayes', 'Support Vector Machine', 'Random Forest')
)

# Define the choices for the multiselect box
choices = ['KNN', 'Neural Networks', 'Lookup Table', 'Naive Bayes', 'Support Vector Machine', 'Random Forest']

# Create the multiselect box in the sidebar
method = st.sidebar.multiselect('Select machine learning method to compare', choices)

# Display the selected track and methods
st.write("---")
st.markdown(f"<h2 style='text-align: center; color: black;'>High Speed Train Track {track}</h2>", unsafe_allow_html=True)

# Determine the folder path for the selected track
selected_track = f'https://github.com/KimSsamu-dev/Streamlit-Dashboard/blob/main/down_lokasi_kereta_{track}.xlsx'
df_track = pd.read_excel(selected_track, engine='openpyxl')

# Column
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)  # Generate 2 columns

    with left_column:
        st.header("Beam Index")
        selected_index = track
        # Read train coordinates and time data from Excel file
        file_path_train = f'https://github.com/KimSsamu-dev/Streamlit-Dashboard/blob/main/down_lokasi_kereta_{track}.xlsx'
        df_train = pd.read_excel(file_path_train)

        # Read Index
        file_azimuth = f'https://github.com/KimSsamu-dev/Streamlit-Dashboard/blob/main/sum_{track}.xlsx'
        df_azimuth = pd.read_excel(file_azimuth)

        # Define base station coordinates
        base_station_x = 0
        base_station_y = 0

        # Define beam length
        beam_length = 1000  # Example length

        # Create Plotly figure
        fig = px.scatter(df_train, x=f'lokasi_{track}_y', y=f'lokasi_{track}_x', title='Train with Beam Pattern')

        # Add base station
        fig.add_trace(go.Scatter(x=[base_station_x], y=[base_station_y], mode='markers', marker=dict(color='red', size=10), name='Base Station'))

        # Add initial beam pattern based on azimuth angle
        initial_azimuth = df_azimuth.loc[0, f'azimuth_{ml_method}']
        angle1 = np.radians(initial_azimuth + beam_width)  # Convert to radians and add/subtract 30 degrees for beam pattern
        angle2 = np.radians(initial_azimuth - beam_width)
        tip1_x = base_station_x + beam_length * np.cos(angle1)
        tip1_y = base_station_y + beam_length * np.sin(angle1)
        tip2_x = base_station_x + beam_length * np.cos(angle2)
        tip2_y = base_station_y + beam_length * np.sin(angle2)
        beam_shape_x = [base_station_x, tip1_x, tip2_x, base_station_x]
        beam_shape_y = [base_station_y, tip1_y, tip2_y, base_station_y]
        fig.add_trace(go.Scatter(x=beam_shape_x, y=beam_shape_y, mode='lines', fill='toself', fillcolor='rgba(0, 0, 255, 0.2)', line=dict(color='blue'), name='Beam Pattern'))

        # Update layout
        fig.update_layout(
            xaxis_title='X',
            yaxis_title='Y',
            xaxis=dict(range=[-400, 400], title_font=dict(size=16)),
            yaxis=dict(range=[-400, 400], title_font=dict(size=16)),
            showlegend=True,
            width=800,
            height=500,
            font=dict(size=14),
            updatemenus=[{
                "buttons": [{
                    "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True, "mode": "immediate"}],
                    "label": "Play",
                    "method": "animate"
                }],
                "direction": "left",
                "pad": {"r": -100, "t": 20},
                "showactive": False,
                "type": "buttons",
                "x": 1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )

        # Define frames for animation
        frames = []
        for i, row in df_train.iterrows():
            closest_train_x = row[f'lokasi_{track}_y']
            closest_train_y = row[f'lokasi_{track}_x']
            azimuth_angle = df_azimuth.loc[i, f'azimuth_{ml_method}']
            angle1 = np.radians(azimuth_angle + beam_width)
            angle2 = np.radians(azimuth_angle - beam_width)
            tip1_x = base_station_x + beam_length * np.cos(angle1)
            tip1_y = base_station_y + beam_length * np.sin(angle1)
            tip2_x = base_station_x + beam_length * np.cos(angle2)
            tip2_y = base_station_y + beam_length * np.sin(angle2)
            beam_shape_x = [base_station_x, tip1_x, tip2_x, base_station_x]
            beam_shape_y = [base_station_y, tip1_y, tip2_y, base_station_y]

            frames.append(go.Frame(data=[
                go.Scatter(x=[base_station_x], y=[base_station_y], mode='markers', marker=dict(color='red', size=10), name='Base Station'),
                go.Scatter(x=[closest_train_x], y=[closest_train_y], mode='markers', marker=dict(color='blue'), name='Train'),
                go.Scatter(x=beam_shape_x, y=beam_shape_y, mode='lines', fill='toself', fillcolor='rgba(0, 0, 255, 0.2)', line=dict(color='blue'), name='Beam Pattern')],
                name=str(i)
            ))

        # Add frames to the figure
        fig.frames = frames

        # Display the plot using Streamlit
        st.plotly_chart(fig)
        st.write(f'Initial azimuth angle: {initial_azimuth} degrees')

    with right_column:
        st.header("Data Rate")
        st.subheader("")

        # Read data from Excel file
        file_datarate = f'https://github.com/KimSsamu-dev/Streamlit-Dashboard/blob/main/sum_{track}.xlsx'
        datas = pd.read_excel(file_datarate)

        # Plot
        fig_rate = px.bar(datas, x='point', y=f'convert_{ml_method}', barmode='group', color_discrete_sequence=['green']*len(datas))

        # Add a red horizontal line
        fig_rate.add_shape(
            type='line',
            x0=0,
            x1=61,
            y0=outtage,  # Change this to the y-value where you want the line
            y1=outtage,  # Same y-value as above to make the line horizontal
            line=dict(color='red', width=5)
        )

        # Update y-axis label
        fig_rate.update_yaxes(title_text='Datarate', title_font={"size": 18})
        fig_rate.update_xaxes(title_text='Point', title_font={"size": 18})
        # Display the plot in Streamlit
        st.plotly_chart(fig_rate)

    st.write("---")
    # Create a 3x2 grid layout
    st.subheader("Accuracy and Prediction Time")
    # Function to display comparison
    def display_comparison(method_name, accuracy, pred_time, outtage, column):
        with column:
            # Display the method name in a larger font size using HTML <h1> tag
            st.markdown(f"<h1>{method_name}</h1>", unsafe_allow_html=True)
            # Display accuracy with larger font size
            st.markdown(f"**Accuracy:** <h2>{accuracy}%</h2>", unsafe_allow_html=True)
            st.markdown(f"**Prediction Time:**<h2>{pred_time}s</h2>", unsafe_allow_html=True)
            st.markdown(f"**Outtage Percentage:**<h2>{outtage}%</h2>", unsafe_allow_html=True)

    # Read the Excel file once
    file_acc = f'https://github.com/KimSsamu-dev/Streamlit-Dashboard/blob/main/Acc_Track1.xlsx'
    df_method = pd.read_excel(file_acc)

    if method:
        cols = st.columns(3)
        for i, method_name in enumerate(method):
            if i < 6:  # Ensure we don't exceed 6 methods
                # Check if the method exists in the DataFrame
                if method_name in df_method["method"].values:
                    # Get the accuracy and prediction time for the current method
                    accuracy = df_method.loc[df_method["method"] == method_name, "Accuracy"].values[0]
                    pred_time = df_method.loc[df_method["method"] == method_name, "Prediction Time"].values[0]
                    outtage = df_method.loc[df_method["method"] == method_name, "Outtage Percentage"].values[0]

                    # Display the method information in the appropriate column
                    display_comparison(method_name, accuracy, pred_time, outtage, cols[i % 3])

                    # Reset columns after every 3 items
                    if (i + 1) % 3 == 0:
                        cols = st.columns(3)
                else:
                    st.write(f"Method {method_name} not found in the data.")
    else:
        st.write("No methods selected")

