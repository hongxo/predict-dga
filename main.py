import streamlit as st
import pandas as pd
import numpy as np
from dga_diag import *
from dga_neural_network import *
import logging 
# Title
st.title("Transformer Diagnostic based on ANN-DGA")

# Input section
st.sidebar.subheader('Input section')
content_name = st.sidebar.text_input('Name:')
content_H2 = st.sidebar.text_input('H2 (ppm):')
content_CH4 = st.sidebar.text_input('CH4 (ppm):') 
content_C2H6 = st.sidebar.text_input('C2H6 (ppm):') 
content_C2H4 = st.sidebar.text_input('C2H4 (ppm):') 
content_C2H2 = st.sidebar.text_input('C2H2 (ppm):') 
content_CO = st.sidebar.text_input('CO (ppm):') 
content_CO2 = st.sidebar.text_input('CO2 (ppm):')
content_submit = st.sidebar.button("Analyze")

# Dataframe initialize
if 'full_data' not in st.session_state:
    st.session_state.full_data = pd.DataFrame(columns = ['Sample', 'H2', 'CH4', 'C2H4', 'C2H6', 'C2H2','CO', 'CO2', 'IEC', 'ROGERS','DUVAL TRI', 'DUVAL PENT', 'NEURAL NETWORK'])
if 'temp_data' not in st.session_state:
    st.session_state.temp_data = pd.DataFrame(columns = ['Sample', 'H2', 'CH4', 'C2H4', 'C2H6', 'C2H2','CO', 'CO2'])

# Perform analysis when the analyze button is hit
if content_submit:
    try:
    # Get the input array as dictionary and transfer to a temporary data slot
        gas_content_temp = {
        'H2' : [float(content_H2)],
        'CH4' : [float(content_CH4)],
        'C2H4' : [float(content_C2H4)],
        'C2H6' : [float(content_C2H6)],
        'C2H2' : [float(content_C2H2)],
        'CO' : [float(content_CO)],
        'CO2': [float(content_CO2)]
        }
        gas_content_temp.update({'Sample': content_name})
        temp_data = pd.DataFrame.from_dict(gas_content_temp)
    # Perform analysis by several methods
        iec_temp = iec(temp_data.iloc[0])
        rogers_temp = rogers(temp_data.iloc[0])
        [duval_triangle_temp, duval_triangle_fig] = duval_triangle(temp_data.iloc[0])
        [duval_pentagon_temp, duval_pentagon_fig] = duval_pentagon(temp_data.iloc[0])
        [mlp_temp, mlp_fig, mlp_per] = dga_multi_layer_perceptron(temp_data.iloc[0])
        
    # Add the analysis result to the temporary dictionary
        gas_content_temp.update({'IEC': iec_temp})
        gas_content_temp.update({'ROGERS': rogers_temp})
        gas_content_temp.update({'DUVAL TRI': duval_triangle_temp})
        gas_content_temp.update({'DUVAL PENT': duval_pentagon_temp})
        gas_content_temp.update({'NEURAL NETWORK': mlp_temp})

    # Convert the temporary dictionary to temporary dataframe
        temp_data = pd.DataFrame.from_dict(gas_content_temp)
        # Add the current data to the existing dataframe
        st.session_state.full_data = pd.concat([st.session_state.full_data,temp_data],ignore_index=True) #st.session_state.full_data.append(temp_data,ignore_index=True)
        
    # Throw exception if an error is encountered        
    except:
        st.error("Please check the gas content again!")
        logging.exception('')
        st.stop()
    
# Display the analysis table
st.table(st.session_state.full_data)

# Display the figures in the expandable section
try:
    with st.expander("See full explanation"):
        if st.session_state.full_data.empty:
            st.warning("Analysis has not been performed yet!")
        else:
            # Create a selectbox for the end-user to choose which sample to display
            sample = st.selectbox('Select sample to show the in-depth analysis',st.session_state.full_data)   
            
            # Select existing sample and transfer to a temporary data slot
            temp_data = st.session_state.full_data.loc[st.session_state.full_data['Sample']==sample]
            
            # Perform analysis by several methods
            iec_temp = iec(temp_data.iloc[0])
            rogers_temp = rogers(temp_data.iloc[0])
            [duval_triangle_temp, duval_triangle_fig] = duval_triangle(temp_data.iloc[0])
            [duval_pentagon_temp, duval_pentagon_fig] = duval_pentagon(temp_data.iloc[0])
            [key_gas_temp, key_gas_fig] = key_gas(temp_data.iloc[0])
            [mlp_temp, mlp_fig, mlp_per] = dga_multi_layer_perceptron(temp_data.iloc[0])
            
            # Display an in-depth analysis
            col1, col2 = st.columns(2)
            # Key gas
            col1.write('Key gas indication: ') 
            col1.write(key_gas_temp)
            col1.pyplot(key_gas_fig)
            # Neural network
            tt = mlp_per.max(axis=1)
            col2.write('The neural network predicts: ') 
            col2.write(mlp_temp + ' (' + str(tt.iloc[0]) + '%)')
            col2.pyplot(mlp_fig)
            # IEC
            col1.write('IEC code: '+ iec_temp)
            # ROGERS
            col2.write('Rogers code: '+ rogers_temp)
            # DUVAL TRIANGLE
            col1.write('Duval triangle #1: ' + duval_triangle_temp)
            col1.pyplot(duval_triangle_fig)
            #DUVAL PENTAGON
            col2.write('Duval pentagon #1: ' + duval_pentagon_temp)
            col2.pyplot(duval_pentagon_fig)
            
            
# Throw exception if an error is encountered                  
except:
    st.error("An error has occured! Please make sure you had filled the inputs!")
    logging.exception('')
    st.stop()
