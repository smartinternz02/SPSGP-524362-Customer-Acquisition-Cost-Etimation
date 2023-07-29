import streamlit as st
import numpy as np
import pandas as pd
import graphviz
from sklearn import tree 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import requests
import json


df=pd.read_csv("media prediction and its cost.csv")

features_to_drop = ['avg_cars_at home(approx).1', 'net_weight', 'meat_sqft', 'salad_bar', 'food_category', 'food_department', 'food_family', 'sales_country', 'marital_status', 'education', 'member_card', 'houseowner', 'brand_name']
df.drop(columns=features_to_drop, inplace=True)

class output:
    def __init__(self,mse,r2,pred,graphs) -> None:
        self.mse = mse
        self.r2 = r2
        self.pred = pred
        self.graphs = graphs
    
if __name__ == "__main__":


    api = 'http://localhost:5000/test'

    st.title("Customer Acquisition Cost Predictor")

    st.header("Team No. 265")

    st.markdown(""" ### Members: 
    Anusha Garg    
    Bhavya Nagpal   
    Saatvik Gupta    
    Gursehaj Singh
    """)

    st.divider()

    st.markdown(""" # Working of the project """)

    st.markdown(""" ## Enter Customer Data """)

    new_inp=pd.DataFrame(columns=df.columns[:-1:],index = [((len(df)))])

    new_inp['store_sales(in millions)'].iloc[0] = st.number_input('Enter estimated Store Sales in months')
    
    new_inp['store_cost(in millions)'].iloc[0] = st.number_input('Enter estimated Store Cost (In Millions)')
    
    new_inp['unit_sales(in millions)'].iloc[0] = st.slider('Enter Unit Sales (In Millions)',1,5,3)
    
    new_inp['promotion_name'].iloc[0] = st.selectbox('Enter The Promotion Name',(df.promotion_name.unique()))

    new_inp['gender'].iloc[0] = st.selectbox('Gender of customer',('M','F'))
    
    new_inp['total_children'].iloc[0] = st.number_input('No. of children of the customer')
    
    new_inp['occupation'].iloc[0] = st.selectbox('Customer Occupation:',(df.occupation.unique()))
    
    new_inp['avg_cars_at home(approx)'].iloc[0] = st.slider('No. of Cars at home',0,6,1)

    new_inp['avg. yearly_income'].iloc[0]= st.selectbox('Average Yearly income',(df['avg. yearly_income'].unique()))
    
    new_inp['num_children_at_home'].iloc[0] = st.slider('No. of children at home',0,6,2)
    
    new_inp['SRP'].iloc[0] = st.number_input('SRP of product bought by the customer')
    
    new_inp['gross_weight'].iloc[0] = st.number_input('Gross weight of the product bought')
    
    new_inp['recyclable_package'].iloc[0] = st.selectbox('Recyclable Package or Not',(0,1))
    
    new_inp['low_fat'].iloc[0] = st.selectbox('Low fat or Not',(0,1))
    
    new_inp['units_per_case'].iloc[0] = st.number_input('Units per case')
    
    new_inp['store_type'].iloc[0]= st.selectbox('Store Type',(df.store_type.unique()))
    
    new_inp['store_city'].iloc[0]= st.selectbox('Store City',(df.store_city.unique()))

    new_inp['store_state'].iloc[0]= st.selectbox('Store State',(df.store_state.unique()))
    
    new_inp['store_sqft'].iloc[0]= st.selectbox('Store SqFt',(df.store_sqft.unique()))
    
    new_inp['grocery_sqft'].iloc[0]= st.selectbox('grocery_sqft',(df.grocery_sqft.unique()))
    
    new_inp['frozen_sqft'].iloc[0]= st.selectbox('frozen_sqft',(df.frozen_sqft.unique()))
    
    new_inp['coffee_bar'].iloc[0]= st.selectbox('coffee_bar',(df.coffee_bar.unique()))
    
    new_inp['video_store'].iloc[0]= st.selectbox('video_store',(df.video_store.unique()))
    
    new_inp['prepared_food'].iloc[0]= st.selectbox('prepared_food',(df.prepared_food.unique()))
    
    new_inp['florist'].iloc[0]= st.selectbox('florist',(df.florist.unique()))
    
    new_inp['media_type'].iloc[0]= st.selectbox('media_type',(df.media_type.unique()))

    st.divider()

    df1=pd.concat([df,new_inp])
    

    categorical_cols = df1.select_dtypes(include='object').columns
    from sklearn.preprocessing import LabelEncoder
    df1[categorical_cols] = df1[categorical_cols].apply(LabelEncoder().fit_transform)
    
        
    split=st.slider('Split the test and train data',0.0,1.0,0.7)

    model = st.radio(
        "Select the model on which you want to input",
        (['Random Forests']))

    if st.button('Train Data'):
        match model:
            case 'Random Forests':
                data_dict=(df1.to_dict())
                data_dict["train_ratio"]=split
                payload=json.dumps(data_dict)
                headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
                response = requests.post(api, data=payload, headers=headers)
                print(response)
                response_dict=response.json()
                out=output(response_dict["mse"],response_dict["r2"],response_dict["pred"],response_dict["graphs"])
                
    st.divider()
    st.markdown(""" ### The Predicted CAC for the customer is """)
    st.write(out.pred)
    st.divider()
    st.markdown(""" #### The MSE is """)
    st.write(out.mse)
    st.markdown(""" #### The R^2 is """)
    st.write(out.r2)
    
    dot_data = tree.export_graphviz(out.graphs[2], out_file=None,filled=True)
    print(type(dot_data))
    st.graphviz_chart(dot_data,True)

