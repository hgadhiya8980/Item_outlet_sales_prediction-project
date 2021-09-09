from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("main_dataframe_item.csv")

X=df.drop("Item_Outlet_Sales", axis=1)
y=df["Item_Outlet_Sales"]

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=123)

sc=StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# 'Item_Weight', 'Item_Visibility', 'Item_MRP',
#        'Outlet_Establishment_Year', 'Item_Outlet_Sales',
#        'Item_Identifier_FDW13', 'Item_Identifier_FDG33',
#        'Item_Identifier_NCY18', 'Item_Identifier_FDD38',
#        'Item_Identifier_DRE49', 'Item_Identifier_FDV60',
#        'Item_Identifier_NCQ06', 'Item_Identifier_FDF52',
#        'Item_Identifier_FDX04', 'Item_Identifier_NCJ30',
#        'Item_Identifier_FDV38', 'Item_Identifier_NCF42',
#        'Item_Identifier_FDT07', 'Item_Identifier_FDW26',
#        'Item_Identifier_NCL31', 'Item_Identifier_FDU12',
#        'Item_Identifier_FDG09', 'Item_Identifier_FDQ40',
#        'Item_Identifier_FDX20', 'Item_Identifier_NCI54',
#        'Item_Identifier_FDX31', 'Item_Identifier_FDP25',
#        'Item_Identifier_FDW49', 'Item_Identifier_FDF56',
#        'Item_Identifier_FDO19', 'Item_Identifier_DRN47',
#        'Item_Identifier_NCB18', 'Item_Fat_Content_LF',
#        'Item_Fat_Content_Low Fat', 'Item_Fat_Content_Regular',
#        'Item_Fat_Content_low fat', 'Item_Fat_Content_reg',
#        'Item_Type_Baking Goods', 'Item_Type_Breads', 'Item_Type_Breakfast',
#        'Item_Type_Canned', 'Item_Type_Dairy', 'Item_Type_Frozen Foods',
#        'Item_Type_Fruits and Vegetables', 'Item_Type_Hard Drinks',
#        'Item_Type_Health and Hygiene', 'Item_Type_Household', 'Item_Type_Meat',
#        'Item_Type_Others', 'Item_Type_Seafood', 'Item_Type_Snack Foods',
#        'Item_Type_Soft Drinks', 'Item_Type_Starchy Foods',
#        'Outlet_Identifier_OUT010', 'Outlet_Identifier_OUT013',
#        'Outlet_Identifier_OUT017', 'Outlet_Identifier_OUT018',
#        'Outlet_Identifier_OUT019', 'Outlet_Identifier_OUT027',
#        'Outlet_Identifier_OUT035', 'Outlet_Identifier_OUT045',
#        'Outlet_Identifier_OUT046', 'Outlet_Identifier_OUT049',
#        'Outlet_Location_Type_Tier 1', 'Outlet_Location_Type_Tier 2',
#        'Outlet_Location_Type_Tier 3', 'Outlet_Type_Grocery Store',
#        'Outlet_Type_Supermarket Type1', 'Outlet_Type_Supermarket Type2',
#        'Outlet_Type_Supermarket Type3'

model = joblib.load("Item_outlet_sales_prediction.pkl")

def house_price_prediction(model, Item_Weight, Item_Visibility, Item_MRP, Outlet_Establishment_Year, Item_Identifier, 
                            Item_Fat_Content, Item_Type, Outlet_Identifier, Outlet_Location_Type, Outlet_Type):
    x=np.zeros(len(X.columns))
    x[0]= Item_Weight
    x[1]=Item_Visibility
    x[2]=Item_MRP
    x[3]=Outlet_Establishment_Year
    
    if "Item_Identifier_"+Item_Identifier in X.columns:
        Item_Identifier_index = np.where(X.columns=="Item_Identifier_"+Item_Identifier)[0][0]
        x[Item_Identifier_index]=1
        
    if "Item_Fat_Content_"+Item_Fat_Content in X.columns:
        Item_Fat_Content_index = np.where(X.columns=="Item_Fat_Content_"+Item_Fat_Content)[0][0]
        x[Item_Fat_Content_index]=1

    if "Item_Type_"+Item_Type in X.columns:
        Item_Type_index = np.where(X.columns=="Item_Type_"+Item_Type)[0][0]
        x[Item_Type_index]=1
        
    if "Outlet_Identifier_"+Outlet_Identifier in X.columns:
        Outlet_Identifier_index = np.where(X.columns=="Outlet_Identifier_"+Outlet_Identifier)[0][0]
        x[Outlet_Identifier_index]=1

    if "Outlet_Location_Type_"+Outlet_Location_Type in X.columns:
        Outlet_Location_Type_index = np.where(X.columns=="Outlet_Location_Type_"+Outlet_Location_Type)[0][0]
        x[Outlet_Location_Type_index]=1
        
    if "Outlet_Type_"+Outlet_Type in X.columns:
        Outlet_Type_index = np.where(X.columns=="Outlet_Type_"+Outlet_Type)[0][0]
        x[Outlet_Type_index]=1
        
    x = sc.transform([x])[0]
    
    return model.predict([x])[0]

app=Flask(__name__)
# model, Item_Weight, Item_Visibility, Item_MRP, Outlet_Establishment_Year, Item_Identifier, 
#                             Item_Fat_Content, Item_Type, Outlet_Identifier, Outlet_Location_Type, Outlet_Type

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Item_Weight = request.form["Item_Weight"]
    Item_Visibility = request.form["Item_Visibility"]
    Item_MRP = request.form["Item_MRP"]
    Outlet_Establishment_Year = request.form["Outlet_Establishment_Year"]
    Item_Identifier = request.form["Item_Identifier"]
    Item_Fat_Content = request.form["Item_Fat_Content"]
    Item_Type = request.form["Item_Type"]
    Outlet_Identifier = request.form["Outlet_Identifier"]
    Outlet_Location_Type = request.form["Outlet_Location_Type"]
    Outlet_Type = request.form["Outlet_Type"]
    
    predicated_price1 =house_price_prediction(model, Item_Weight, Item_Visibility, Item_MRP, Outlet_Establishment_Year, Item_Identifier, 
                                            Item_Fat_Content, Item_Type, Outlet_Identifier, Outlet_Location_Type, Outlet_Type)
    predicated_price = round(predicated_price1, 2)

    return render_template("index.html", prediction_text="Predicated price of bangalore House is {} RS".format(predicated_price))


if __name__ == "__main__":
    app.run()    
    