import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown('''
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .price-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
''', unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model_and_info():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model_info.pkl', 'rb') as file:
        model_info = pickle.load(file)
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)
    return model, model_info, feature_names

@st.cache_data
def load_original_data():
    return pd.read_csv('cars_cleaned_encoded.csv')

# Load everything
try:
    model, model_info, feature_names = load_model_and_info()
    df_original = load_original_data()
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict used car prices using Machine Learning</p>', unsafe_allow_html=True)
    
    # Model info in expander
    with st.expander("ℹ️ Model Information"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", model_info['model_name'])
        with col2:
            st.metric("Accuracy", f"{model_info['test_accuracy']:.2f}%")
        with col3:
            st.metric("R² Score", f"{model_info['test_r2']:.4f}")
        with col4:
            st.metric("RMSE", f"{model_info['test_rmse']:.0f}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Data Explorer", "📈 Model Insights"])
    
    # ===================================================================
    # TAB 1: PREDICT PRICE
    # ===================================================================
    with tab1:
        st.markdown("### Enter Car Details")
        
        col1, col2, col3 = st.columns(3)
        
        # Create input fields based on your features
        user_input = {}
        
        with col1:
            st.markdown("#### 🚙 Basic Information")
            user_input['Kilometers_Driven'] = st.number_input(
                "Kilometers Driven", 
                min_value=0, 
                max_value=500000, 
                value=50000, 
                step=1000
            )
            user_input['Age'] = st.number_input(
                "Car Age (years)", 
                min_value=0, 
                max_value=25, 
                value=5, 
                step=1
            )
            user_input['Seats'] = st.selectbox(
                "Number of Seats", 
                options=[2, 4, 5, 6, 7, 8, 9, 10],
                index=2
            )
        
        with col2:
            st.markdown("#### ⚙️ Specifications")
            user_input['Engine'] = st.number_input(
                "Engine (CC)", 
                min_value=500, 
                max_value=5000, 
                value=1500, 
                step=100
            )
            user_input['Power'] = st.number_input(
                "Power (bhp)", 
                min_value=30, 
                max_value=500, 
                value=100, 
                step=10
            )
            user_input['Mileage(kmpl)'] = st.number_input(
                "Mileage (km/l)", 
                min_value=5.0, 
                max_value=35.0, 
                value=15.0, 
                step=0.5
            )
        
        with col3:
            st.markdown("#### 🏷️ Category")
            
            # For encoded features, you'll need to handle them based on your encoding
            # This is a simplified version - adjust based on your actual encoded columns
            
            # Brand (if label encoded)
            if 'Brand' in feature_names:
                brand_options = sorted(df_original['Brand'].unique()) if 'Brand' in df_original.columns else list(range(31))
                user_input['Brand'] = st.selectbox("Brand", options=brand_options)
            
            # Location (if label encoded)
            if 'Location' in feature_names:
                location_options = sorted(df_original['Location'].unique()) if 'Location' in df_original.columns else list(range(11))
                user_input['Location'] = st.selectbox("Location", options=location_options)
            
            # Fuel Type (one-hot encoded - you need to set all fuel type columns)
            fuel_types = [col for col in feature_names if col.startswith('Fuel_Type_')]
            if fuel_types:
                fuel_display = st.selectbox("Fuel Type", 
                    options=['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])
                for fuel_col in fuel_types:
                    user_input[fuel_col] = 0
                # Set the selected fuel type to 1
                if fuel_display != 'Diesel':  # Diesel is the reference (drop_first=True)
                    matching_col = f"Fuel_Type_{fuel_display}"
                    if matching_col in user_input:
                        user_input[matching_col] = 1
            
            # Transmission (one-hot encoded)
            transmission_cols = [col for col in feature_names if col.startswith('Transmission_')]
            if transmission_cols:
                transmission = st.selectbox("Transmission", options=['Manual', 'Automatic'])
                for trans_col in transmission_cols:
                    user_input[trans_col] = 0
                if transmission == 'Automatic':
                    user_input['Transmission_Manual'] = 1
            
            # Owner Type (one-hot encoded)
            owner_cols = [col for col in feature_names if col.startswith('Owner_Type_')]
            if owner_cols:
                owner = st.selectbox("Owner Type", 
                    options=['First', 'Second', 'Third', 'Fourth & Above'])
                for owner_col in owner_cols:
                    user_input[owner_col] = 0
                if owner != 'First':
                    matching_col = f"Owner_Type_{owner}"
                    if matching_col in user_input:
                        user_input[matching_col] = 1
        
        # Set default values for any missing features
        for feature in feature_names:
            if feature not in user_input:
                user_input[feature] = 0
        
        st.markdown("---")
        
        # Predict button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            predict_button = st.button("🔮 Predict Price", use_container_width=True, type="primary")
        
        if predict_button:
            # Prepare input dataframe
            input_df = pd.DataFrame([user_input])[feature_names]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Display prediction
            st.markdown("---")
            st.markdown(f'''
            <div class="prediction-box">
                <h2>Predicted Car Price</h2>
                <div class="price-value">₹ {prediction:,.2f}</div>
                <p>Estimated market value based on the provided specifications</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Additional insights
            st.markdown("### 📊 Price Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Confidence", f"{model_info['test_accuracy']:.1f}%")
                st.caption("Model accuracy on test data")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                avg_price = df_original['Price'].mean()
                diff = ((prediction - avg_price) / avg_price) * 100
                st.metric("vs Average", f"{diff:+.1f}%")
                st.caption(f"Average market price: ₹{avg_price:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                price_range = (prediction * 0.9, prediction * 1.1)
                st.metric("Price Range", f"₹{price_range[0]:,.0f} - ₹{price_range[1]:,.0f}")
                st.caption("±10% confidence interval")
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ===================================================================
    # TAB 2: DATA EXPLORER
    # ===================================================================
    with tab2:
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cars", f"{len(df_original):,}")
        with col2:
            st.metric("Features", len(feature_names))
        with col3:
            st.metric("Avg Price", f"₹{df_original['Price'].mean():,.0f}")
        with col4:
            st.metric("Price Range", f"₹{df_original['Price'].min():,.0f} - ₹{df_original['Price'].max():,.0f}")
        
        st.markdown("---")
        
        # Price distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df_original, x='Price', nbins=50,
                             title='Price Distribution',
                             labels={'Price': 'Price (₹)', 'count': 'Frequency'})
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df_original, y='Price',
                        title='Price Box Plot',
                        labels={'Price': 'Price (₹)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Show sample data
        st.markdown("### 📋 Sample Data")
        st.dataframe(df_original.head(100), use_container_width=True)
    
    # ===================================================================
    # TAB 3: MODEL INSIGHTS
    # ===================================================================
    with tab3:
        st.markdown("### 🎯 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Model metrics
            metrics_data = {
                'Metric': ['R² Score', 'RMSE', 'MAE', 'Accuracy'],
                'Value': [
                    f"{model_info['test_r2']:.4f}",
                    f"₹{model_info['test_rmse']:,.0f}",
                    f"₹{model_info['test_mae']:,.0f}",
                    f"{model_info['test_accuracy']:.2f}%"
                ]
            }
            st.table(pd.DataFrame(metrics_data))
        
        with col2:
            st.markdown("#### 📝 Model Details")
            st.write(f"**Algorithm:** {model_info['model_name']}")
            st.write(f"**Total Features:** {model_info['n_features']}")
            st.write(f"**Training Samples:** {len(df_original)}")
            
        st.markdown("---")
        st.info("💡 **Tip:** Higher R² score means better predictions. RMSE shows average prediction error.")

except FileNotFoundError as e:
    st.error(f"❌ Error: Required files not found. Please ensure you have saved the model files.")
    st.write(f"Missing file: {e.filename}")
except Exception as e:
    st.error(f"❌ An error occurred: {str(e)}")