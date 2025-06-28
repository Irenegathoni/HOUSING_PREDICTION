import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("C:\\Users\\hp\\Downloads\\INDIA HOUSING\\archive (1)\\House_Rent_Dataset.csv")
df.drop(['Posted On', 'Tenant Preferred', 'Area Locality'], axis=1, inplace=True)
# Define features and target
features = ['BHK', 'Size', 'City', 'Furnishing Status', 'Bathroom']
target = 'Rent'

X = df[features]
y = df[target]

# Categorical and numeric
categorical_features = ['City', 'Furnishing Status']
numeric_features = ['BHK', 'Size', 'Bathroom']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'
)

# Full model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split and fit model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI starts here
st.title(" House Rent Predictor (India Dataset)")

# Inputs from user
city = st.selectbox("City", df['City'].unique())
furnishing = st.selectbox("Furnishing Status", df['Furnishing Status'].unique())
size = st.number_input("Size (in Sqft)", min_value=200, max_value=5000, step=50)
bhk = st.slider("Number of BHK", 1, 10, 2)
bathroom = st.slider("Number of Bathrooms", 1, 10, 1)

# Predict button
if st.button("Predict Rent"):
    input_data = pd.DataFrame({
        'BHK': [bedroom,hallway,kitchen],
        'Size': [size],
        'City': [city],
        'Furnishing Status': [furnishing],
        'Bathroom': [bathroom]
    })

    rent_prediction = model.predict(input_data)
    st.success(f"Estimated Rent: ₹{int(rent_prediction[0]):,}")