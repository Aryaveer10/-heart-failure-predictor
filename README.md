# Heart-Failure-Prediction
This is a deep learning model that is trained to predict the likelihood of heart failure in a patient, given certain clinical characteristics. The model was created using TensorFlow and is deployed using Streamlit.

### To build the model on your local machine:

Clone the repository to your local machine using git. The dataset used to train this model can be found here: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Make sure you have dependencies listed in 'requirements.txt' installed in your Python environment, and the csv file in the same repository.

Run each cell of model.ipynb notebook sequentially to train and save the model.

### To use the model:

Navigate to the directory where you cloned the repository in a terminal and run streamlit run app.py to start the app. This will open a new tab in your web browser with the model's interface.

Follow the prompts on the page to enter the relevant clinical characteristics for the patient.

The model will output a prediction of the likelihood of heart failure for the patient.

Please note that this is just a model and should not be used as a definitive diagnostic tool. It is intended to be used as a tool to assist in the diagnostic process, but final decisions should always be made by a trained medical professional.

I hope you find this model helpful in your work. If you have any questions or run into any issues while using it, don't hesitate to ask for help!