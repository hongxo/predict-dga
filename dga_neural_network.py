from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def dga_multi_layer_perceptron(gas_content):
    # Get the input
    H2 = gas_content["H2"]
    CH4 = gas_content["CH4"]
    C2H6 = gas_content["C2H6"]
    C2H4 = gas_content["C2H4"]
    C2H2 = gas_content["C2H2"]

    # Define the input vector (X_input)
    X_input = [[H2,CH4,C2H6,C2H4,C2H2]] #np.array([[H2,CH4,C2H6,C2H4,C2H2]])
    scaler = StandardScaler()
    df = pd.read_csv("X_ann.csv")
    #   X = df[:1]
    scaled_X = scaler.fit_transform(df)
    # Load the pre-trained model
    filename = 'dga_neuralnet_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    class_names = ["PD","D1", "D2", "T1", "T2", "T3","N"]
    x_test1 = scaler.transform(X_input)
    # Make prediction
    y_output = loaded_model.predict(x_test1)
    # if y_output == 1:
    #     Fault_code = "PD"
    # elif y_output == 2:
    #     Fault_code = "D1"
    # elif y_output == 3:
    #     Fault_code = "D2"
    # elif y_output == 4:
    #     Fault_code = "T1"
    # elif y_output == 5:
    #     Fault_code = "T2"
    # elif y_output == 6:
    #     Fault_code = "T3"
    # else:
    #     Fault_code = "N/A"            
    Fault_code = class_names[y_output[0]-1]
    # For plotting
    Fault_vector = loaded_model.predict_proba(x_test1)
    Fault_vector = np.ravel(Fault_vector)
    labels = ['PD','D1','D2','T1','T2','T3','N']
    Fault_label = np.arange(len(labels))  # the label locations
    width = 0.7  # the width of the bars
    mlp_figure, ax = plt.subplots()
    rects1 = ax.barh(y= Fault_label, height = 0.7, width = np.floor(100*Fault_vector), color=['grey', 'green', 'magenta', 'lightgrey', 'yellow','red','blue'])
    ax.set_yticks(Fault_label)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Fault percentage')
    ax.bar_label(rects1, padding=3)
    mlp_figure.tight_layout()
    mlp_percentage = np.floor((loaded_model.predict_proba(X_input))*100)
    mlp_percentage = pd.DataFrame(mlp_percentage,columns=labels)

    return [Fault_code, mlp_figure, mlp_percentage]