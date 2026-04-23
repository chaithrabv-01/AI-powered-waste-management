import streamlit as st
import pandas as pd
import numpy as np
import math
import cv2
import random
from PIL import Image
from datetime import datetime, timedelta


def classify_waste(image_file):
    try:
        # Convert the uploaded file to an OpenCV image
        image = Image.open(image_file)
        image = np.array(image)
        
        # Convert RGB to BGR (OpenCV format)
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_cv = image # Grayscale or other
            
        # Simulate processing time and feature extraction
        height, width = img_cv.shape[:2]
        mean_color = cv2.mean(img_cv)[:3]
        
        # Simulated classification logic based on mock probabilities
        waste_classes = ["Plastic", "Organic", "Metal", "Paper", "Glass"]
        
        # We add some deterministic behavior based on image size/color for the mock
        seed_value = int(mean_color[0] + width + height)
        random.seed(seed_value)
        
        detected_class = random.choice(waste_classes)
        confidence = round(random.uniform(75.0, 99.9), 2)
        
        return detected_class, confidence
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Unknown", 0.0


DEPOT_LOCATION = {"id": "Depot", "lat": 12.9716, "lon": 77.5946}

BIN_LOCATIONS = {
    "Bin-001": {"lat": 12.9750, "lon": 77.5900},
    "Bin-002": {"lat": 12.9800, "lon": 77.6000},
    "Bin-003": {"lat": 12.9650, "lon": 77.5850},
    "Bin-004": {"lat": 12.9600, "lon": 77.6100},
    "Bin-005": {"lat": 12.9850, "lon": 77.5800},
}

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

def optimize_route(full_bins):
    if not full_bins:
        return [], 0.0

    unvisited = []
    for bin_id in full_bins:
        if bin_id in BIN_LOCATIONS:
            unvisited.append({
                "id": bin_id,
                "lat": BIN_LOCATIONS[bin_id]["lat"],
                "lon": BIN_LOCATIONS[bin_id]["lon"]
            })

    if not unvisited:
        return [], 0.0

    current_location = DEPOT_LOCATION
    optimized_route = [DEPOT_LOCATION]
    total_distance = 0.0

    while unvisited:
        nearest_bin = None
        min_dist = float('inf')

        for target_bin in unvisited:
            dist = haversine_distance(
                current_location["lat"], current_location["lon"],
                target_bin["lat"], target_bin["lon"]
            )
            if dist < min_dist:
                min_dist = dist
                nearest_bin = target_bin

        optimized_route.append(nearest_bin)
        total_distance += min_dist
        current_location = nearest_bin
        unvisited.remove(nearest_bin)

    return_dist = haversine_distance(
        current_location["lat"], current_location["lon"],
        DEPOT_LOCATION["lat"], DEPOT_LOCATION["lon"]
    )
    total_distance += return_dist
    optimized_route.append(DEPOT_LOCATION)

    return optimized_route, round(total_distance, 2)

def get_map_data(route):
    if not route:
        return pd.DataFrame()
    df = pd.DataFrame(route)
    return df[['lat', 'lon']]



def get_initial_dataset():
    today = datetime.today()
    dates = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(14, 0, -1)]
    base_waste = np.linspace(100, 150, 14)
    noise = np.random.normal(0, 15, 14)
    waste_amounts = np.maximum(base_waste + noise, 0).round(2)
    data = {
        'Date': dates,
        'Bin ID': ['Bin-001'] * 14,
        'Waste Amount (kg)': waste_amounts,
        'Primary Type': np.random.choice(['Plastic', 'Organic', 'Mixed'], 14)
    }
    return pd.DataFrame(data)

def predict_future_waste(df, days_ahead=7):
    if df.empty or len(df) < 2:
        return pd.DataFrame()
        
    df['Date'] = pd.to_datetime(df['Date'])
    df_sorted = df.sort_values(by='Date')
    
    y = df_sorted['Waste Amount (kg)'].values
    x = np.arange(len(y))
    
    if len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
    else:
        p = lambda val: y[0]
        
    last_date = df_sorted['Date'].iloc[-1]
    future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days_ahead + 1)]
    future_x = np.arange(len(y), len(y) + days_ahead)
    predicted_y = np.maximum(p(future_x), 0).round(2) 
    
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Waste (kg)': predicted_y
    })
    return pred_df


st.set_page_config(page_title="Smart AI Waste Management", layout="wide")
st.title("♻️ Smart AI Waste Management System")

# 1. AI Waste Classifier
st.header("1. AI Waste Classifier")
st.write("Upload an image of waste to classify it into categories like Plastic, Organic, Metal, etc.")
uploaded_file = st.file_uploader("Upload an image of waste", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    with st.spinner('Classifying image...'):
        detected_class, confidence = classify_waste(uploaded_file)
    st.success(f"**Detected Class:** {detected_class} | **Confidence:** {confidence}%")

st.divider()

# 2. Real-time Bin Monitoring & Route Optimization
st.header("2. Bin Monitoring & Route Optimization")
st.write("Simulate full bins to generate an optimized collection route.")
st.write("Select which bins are currently FULL:")

full_bins = []
cols = st.columns(len(BIN_LOCATIONS))
for i, bin_id in enumerate(BIN_LOCATIONS.keys()):
    with cols[i]:
        if st.checkbox(bin_id):
            full_bins.append(bin_id)

if st.button("Calculate Optimized Route"):
    if full_bins:
        route, total_distance = optimize_route(full_bins)
        st.write(f"### Optimized Route (Total Distance: {total_distance} km)")
        route_steps = " ➡️ ".join([step['id'] for step in route])
        st.info(route_steps)
            
        map_data = get_map_data(route)
        if not map_data.empty:
            st.map(map_data, zoom=12)
    else:
        st.warning("Please select at least one full bin to optimize a route.")

st.divider()

# 3. Predictive Analytics
st.header("3. Predictive Analytics")
st.write("Analyze past waste generation and predict future trends.")

if st.button("Load Historical Data & Predict"):
    df = get_initial_dataset()
    st.subheader("Historical Data (Last 14 Days)")
    st.dataframe(df, use_container_width=True)
    
    st.write("**Waste Generation Trend:**")
    chart_data = df[['Date', 'Waste Amount (kg)']].set_index('Date')
    st.line_chart(chart_data)
    
    pred_df = predict_future_waste(df, days_ahead=7)
    st.subheader("Future Predictions (Next 7 Days)")
    st.dataframe(pred_df, use_container_width=True)
    
    st.write("**Predicted Waste Generation Trend:**")
    pred_chart_data = pred_df[['Date', 'Predicted Waste (kg)']].set_index('Date')
    st.line_chart(pred_chart_data)
