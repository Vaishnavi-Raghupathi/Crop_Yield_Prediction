import pandas as pd

# Load CSV files
soil_data = pd.read_csv('/content/soil_analysis_data.csv')
water_data = pd.read_csv('/content/water_usage_data.csv')
crop_data = pd.read_csv('/content/crop_production_data.csv')

# Display the first few rows
print(soil_data.head(), water_data.head(), crop_data.head())

# Check for missing values
print(soil_data.isnull().sum())
print(water_data.isnull().sum())
print(crop_data.isnull().sum())

# Clean the data
soil_data.dropna(inplace=True)  # Remove rows with missing data
water_data.dropna(inplace=True)
crop_data.dropna(inplace=True)

# Check for dublicates
print(soil_data.duplicated().sum())
print(water_data.duplicated().sum())
print(crop_data.duplicated().sum())

# Remove duplicates
soil_data.drop_duplicates(inplace=True)
water_data.drop_duplicates(inplace=True)
crop_data.drop_duplicates(inplace=True)

# Check the cleaned data
print(soil_data.head())
print(water_data.head())
print(crop_data.head())

print(soil_data.columns)
print(water_data.columns)
print(crop_data.columns)

# Merge soil and water data on 'District'
merged_data = pd.merge(soil_data, water_data, on='District')

# Merge with crop production data
final_data = pd.merge(merged_data, crop_data, on=['District', 'Crop'])

print(final_data.head())
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le = LabelEncoder()
final_data['Soil Type'] = le.fit_transform(final_data['Soil Type'])
final_data['Irrigation Method'] = le.fit_transform(final_data['Irrigation Method'])
final_data['Season'] = le.fit_transform(final_data['Season'])
print(final_data.columns)
from sklearn.model_selection import train_test_split

# Define features and target
X = final_data[['pH Level', 'Organic Matter (%)', 'Nitrogen Content (kg/ha)', 'Phosphorus Content (kg/ha)',
                'Potassium Content (kg/ha)', 'Water Consumption (liters/hectare)', 'Water Availability (liters/hectare)',
                'Area (hectares)', 'Soil Type', 'Irrigation Method', 'Season']]
y = final_data['Yield (quintals)']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Linear Regression - MSE: {mse}, R2: {r2}')
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest - MSE: {mse_rf}, R2: {r2_rf}')
# Import necessary widgets
from IPython.display import display
import ipywidgets as widgets

# Create input widgets for all 11 features
ph_level = widgets.FloatText(description="pH Level:")
organic_matter = widgets.FloatText(description="Organic Matter (%):")
nitrogen_content = widgets.FloatText(description="Nitrogen (kg/ha):")
phosphorus_content = widgets.FloatText(description="Phosphorus (kg/ha):")
potassium_content = widgets.FloatText(description="Potassium (kg/ha):")
water_consumption = widgets.FloatText(description="Water Consumption (liters/hectare):")
water_availability = widgets.FloatText(description="Water Availability (liters/hectare):")
area_hectares = widgets.FloatText(description="Area (hectares):")
soil_type = widgets.IntText(description="Soil Type (encoded):")
irrigation_method = widgets.IntText(description="Irrigation Method (encoded):")
season = widgets.IntText(description="Season (encoded):")

# Display input form
display(ph_level, organic_matter, nitrogen_content, phosphorus_content, potassium_content,
        water_consumption, water_availability, area_hectares, soil_type, irrigation_method, season)

# Function to make predictions
def make_prediction():
    # Create the feature array with all 11 features in the correct order
    features = [ph_level.value, organic_matter.value, nitrogen_content.value,
                phosphorus_content.value, potassium_content.value,
                water_consumption.value, water_availability.value,
                area_hectares.value, soil_type.value, irrigation_method.value, season.value]

    # Perform prediction using the trained model
    prediction = rf_model.predict([features])
    print(f"Predicted Crop Yield: {prediction[0]} quintals")

# Button to trigger prediction
predict_button = widgets.Button(description="Predict Yield")
predict_button.on_click(lambda _: make_prediction())

display(predict_button)


