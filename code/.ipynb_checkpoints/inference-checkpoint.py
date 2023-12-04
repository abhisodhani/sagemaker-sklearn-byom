import os
import joblib
import numpy as np
from io import StringIO

def convert_to_numpy_array(string):
    # Split the string into rows
    rows = string.split('\n')

    # Split each row into columns and convert to float
    array = [[float(num) for num in row.split(',')] for row in rows]

    # Convert the list of lists to a numpy array
    numpy_array = np.array(array)

    return numpy_array

def input_fn(request_body, request_content_type):
    """An input_fn that load and transform numpy array"""
    
    print(request_content_type)
    
    array = convert_to_numpy_array(request_body)
    
    #test data capture by removing a row from input request
    trans_array= np.delete(array,0,0) 

    return trans_array
    
def predict_fn(input_object, model):
    ###########################################
    # Do your custom preprocessing logic here #
    ###########################################

    print("********calling model*********")
    predictions = model.predict(input_object)
    return predictions


def model_fn(model_dir):
    print("loading model.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return loaded_model
