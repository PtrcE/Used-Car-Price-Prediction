import pandas as pd
import numpy as np
import gradio as gr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


#Loading the dataset
dataset = pd.read_csv('used_cars.csv')

#To drop irrelevant columns
dataset.drop(['clean_title', 'ext_col', 'int_col'], axis=1, inplace=True)

print(dataset.head())
print(dataset.info())
print(dataset.describe())

#To Handle missing values
dataset.fillna(dataset.select_dtypes(include=np.number).mean(), inplace=True)

#Converts features to numerical
encodedFeatures = pd.get_dummies(dataset[['brand', 'model']])
dataset['milage'] = dataset['milage'].str.replace('mi.', '', regex=True).str.replace(',', '', regex=True).astype(int)
dataset['price'] = dataset['price'].str.replace('$', '', regex=True).str.replace(',', '', regex=True).astype(int)

#Defines features and targets variables
x =pd.concat([encodedFeatures, dataset[['milage']]], axis=1) #Feature
y = dataset['price'] #Target

#Train Random Forest Regressor model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rModel = RandomForestRegressor(n_estimators=100, random_state=42)
rModel.fit(x_train, y_train)

# Make predictions
y_pred = rModel.predict(x_test)

# Evaluate the model
MAE = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {MAE}")
print(f"R2 Score: {r2}")

#GUI
def predictPrice(brand, model, milage):
    inputFeatures = pd.DataFrame(columns = x.columns)
    inputFeatures.loc[0] = 0
    
    inputFeatures['milage'] = int(milage)
    
    brandColumn = f'brand_{brand}'
    modelColumn = f'brand_{model}'
    
    if brandColumn in inputFeatures.columns:
        inputFeatures[brandColumn] = 1
    
    if modelColumn in inputFeatures.columns:
        inputFeatures[modelColumn] = 1
        
    
    predictedPrice = rModel.predict(inputFeatures)[0]
    return f"Predicted Price: ${predictedPrice:,.2f}"


interface = gr.Interface(
    fn = predictPrice, 
    inputs = [
        gr.Textbox(label="Enter a Car Brand"), 
        gr.Textbox(label="Enter the Model"), 
        gr.Textbox(label="Enter the Mileage")
    ], 
    outputs = gr.Textbox(label="Predicted Price")
)

interface.launch()