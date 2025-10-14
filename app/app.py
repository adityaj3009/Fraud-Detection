# app/streamlit_app.py
# Streamlit Web Application for Credit Card Fraud Detection
# FIXED VERSION - Works properly with error handling

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import sys
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION - MUST BE FIRST STREAMLIT COMMAND
# ============================================================================

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #666;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_model_path():
    """Get correct path to model file"""
    # Try multiple paths
    possible_paths = [
        'model/fraud_model.pkl',
        '../model/fraud_model.pkl',
        Path(__file__).parent.parent / 'model' / 'fraud_model.pkl',
        'fraud_detection/model/fraud_model.pkl'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def get_scaler_path():
    """Get correct path to scaler file"""
    # Try multiple paths
    possible_paths = [
        'model/scaler.pkl',
        '../model/scaler.pkl',
        Path(__file__).parent.parent / 'model' / 'scaler.pkl',
        'fraud_detection/model/scaler.pkl'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def validate_csv_structure(df):
    """Validate that CSV has required columns"""
    required_columns = ['Time', 'Amount']
    v_columns = [f'V{i}' for i in range(1, 29)]
    required_columns.extend(v_columns)
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"
    
    if len(df.columns) != 30:
        return False, f"Expected 30 columns, found {len(df.columns)}"
    
    return True, "CSV structure is valid"

def prepare_features(df):
    """Prepare features in correct order for model prediction"""
    # Ensure correct column order: Time, V1-V28, Amount
    base_columns = ['Time']
    v_columns = [f'V{i}' for i in range(1, 29)]
    amount_column = ['Amount']
    
    required_order = base_columns + v_columns + amount_column
    
    # Reorder columns if they exist
    existing_columns = [col for col in required_order if col in df.columns]
    
    if len(existing_columns) != 30:
        raise ValueError(f"Missing some required columns. Found {len(existing_columns)}/30 columns")
    
    return df[existing_columns]

# ============================================================================
# LOAD MODELS AND SCALER
# ============================================================================

@st.cache_resource
def load_models():
    """Load pre-trained model and scaler with better error handling"""
    try:
        model_path = get_model_path()
        scaler_path = get_scaler_path()
        
        if model_path is None or scaler_path is None:
            return None, None, False
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler, True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, False

# Load models
model, scaler, models_loaded = load_models()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Select a page:", 
    ["Home", "Single Transaction", "Batch Analysis", "Model Info"]
)

# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == "Home":
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("---")
    
    # Status check
    if not models_loaded:
        st.error("⚠️ Models not loaded! Please ensure model files are in the correct location.")
    else:
        st.success("✅ Models loaded successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 Overview")
        st.write("""
        This application uses machine learning to detect fraudulent credit card transactions
        in real-time with high accuracy.
        
        **Key Features:**
        - 🎯 Single transaction fraud detection
        - 📈 Batch transaction analysis
        - 📊 Model performance metrics
        - 🔐 Secure and reliable predictions
        
        **Dataset:**
        - 284,807 transactions
        - 30 anonymized features
        - Binary classification: Genuine vs Fraud
        """)
    
    with col2:
        st.header("📈 Model Performance")
        st.info("""
        **Best Model: Random Forest**
        
        - **Accuracy:** ~99.9%
        - **Precision:** ~88%
        - **Recall:** ~80%
        - **AUC-ROC:** ~0.98
        - **F1-Score:** ~0.84
        
        *These metrics indicate the model's ability to correctly identify fraud
        while minimizing false alarms.*
        """)
    
    st.markdown("---")
    
    st.header("🚀 How to Use")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1️⃣ Single Transaction")
        st.write("Check if a single transaction is fraudulent by entering details")
    
    with col2:
        st.subheader("2️⃣ Batch Analysis")
        st.write("Upload a CSV file with multiple transactions for analysis")
    
    with col3:
        st.subheader("3️⃣ Model Info")
        st.write("View detailed model information and performance metrics")
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="footer">Made by Aditya Jalgaonkar</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 2: SINGLE TRANSACTION PREDICTION
# ============================================================================

elif page == "Single Transaction":
    st.title("🔍 Single Transaction Fraud Detection")
    st.markdown("---")
    
    if not models_loaded:
        st.error("❌ Models not loaded. Please check the model files.")
    else:
        st.write("Enter transaction details to check if it's fraudulent:")
        
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Information:**")
                time = st.number_input(
                    "Time (seconds from start)", 
                    min_value=0, 
                    max_value=172792, 
                    value=0,
                    step=1
                )
                amount = st.number_input(
                    "Transaction Amount ($)", 
                    min_value=0.0, 
                    value=100.0,
                    step=0.01
                )
            
            with col2:
                st.write("**Principal Component Features (V1-V28):**")
                st.caption("These are anonymized features from PCA")
            
            st.markdown("---")
            st.write("**Feature Values (V1-V28):** Drag sliders to adjust")
            
            # Generate 28 V features in a more compact way
            v_features = {}
            cols = st.columns(4)
            
            for i in range(1, 29):
                col_idx = (i - 1) % 4
                with cols[col_idx]:
                    v_features[f'V{i}'] = st.slider(
                        f"V{i}", 
                        -5.0, 
                        5.0, 
                        0.0, 
                        step=0.1,
                        label_visibility="collapsed"
                    )
            
            # Submit button
            submit_button = st.form_submit_button(
                "🔮 Check Transaction",
                use_container_width=True
            )
            
            if submit_button:
                try:
                    # Prepare input data
                    input_data = [time] + [v_features[f'V{i}'] for i in range(1, 29)] + [amount]
                    
                    # Scale input
                    input_array = np.array(input_data).reshape(1, -1)
                    scaled_input = scaler.transform(input_array)
                    
                    # Make prediction
                    prediction = model.predict(scaled_input)[0]
                    probability = model.predict_proba(scaled_input)[0]
                    fraud_prob = float(probability[1])
                    genuine_prob = float(probability[0])
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🎯 Prediction Result")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if prediction == 1:
                            st.error("⚠️ FRAUDULENT TRANSACTION")
                        else:
                            st.success("✅ GENUINE TRANSACTION")
                    
                    with col2:
                        st.metric("Fraud Probability", f"{fraud_prob:.2%}")
                    
                    with col3:
                        st.metric("Genuine Probability", f"{genuine_prob:.2%}")
                    
                    # Probability visualization
                    st.markdown("---")
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    categories = ['Genuine', 'Fraudulent']
                    probabilities = [genuine_prob, fraud_prob]
                    colors = ['#09ab3b', '#d73449']
                    
                    bars = ax.bar(categories, probabilities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                    ax.set_ylabel('Probability', fontsize=12)
                    ax.set_title('Transaction Classification Probability', fontsize=14, fontweight='bold')
                    ax.set_ylim([0, 1])
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, probabilities):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{prob:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="footer">Made by Aditya Jalgaonkar</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 3: BATCH ANALYSIS
# ============================================================================

elif page == "Batch Analysis":
    st.title("📊 Batch Transaction Analysis")
    st.markdown("---")
    
    if not models_loaded:
        st.error("❌ Models not loaded. Please check the model files.")
    else:
        st.write("Upload a CSV file with multiple transactions for batch analysis")
        
        # CSV template information
        with st.expander("📋 CSV File Requirements"):
            st.write("""
            Your CSV file must contain exactly 30 columns with these names:
            
            **Required Columns (in any order):**
            - `Time`: Seconds elapsed between transaction and first transaction
            - `V1` to `V28`: 28 principal component features (anonymized)
            - `Amount`: Transaction amount
            
            **Example CSV structure:**
            ```csv
            Time,V1,V2,V3,...,V28,Amount
            0,-1.359807,-0.072781,2.536347,...,0.133558,149.62
            0,1.191857,0.266151,0.166480,...,-0.008983,2.69
            ```
            
            **Note:** The model expects all 30 features. Column order doesn't matter.
            """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read CSV
                df = pd.read_csv(uploaded_file)
                
                st.write(f"✅ **File uploaded successfully!** ({len(df)} transactions)")
                st.write("**First few rows:**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Validate CSV structure
                is_valid, message = validate_csv_structure(df)
                
                if not is_valid:
                    st.error(f"❌ Invalid CSV structure: {message}")
                    st.info("Please check the CSV requirements above and upload a properly formatted file.")
                else:
                    if st.button("🔮 Analyze Transactions", use_container_width=True):
                        with st.spinner("Analyzing transactions..."):
                            # Prepare features in correct order
                            processed_df = prepare_features(df)
                            
                            # Make predictions
                            scaled_data = scaler.transform(processed_df)
                            predictions = model.predict(scaled_data)
                            probabilities = model.predict_proba(scaled_data)[:, 1]
                            
                            # Add predictions to dataframe
                            results_df = df.copy()
                            results_df['Prediction'] = predictions
                            results_df['Fraud_Probability'] = probabilities
                            results_df['Classification'] = results_df['Prediction'].apply(
                                lambda x: 'Fraud' if x == 1 else 'Genuine'
                            )
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("📈 Analysis Results")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            total_transactions = len(results_df)
                            frauds = (results_df['Prediction'] == 1).sum()
                            genuine = (results_df['Prediction'] == 0).sum()
                            fraud_rate = (frauds / total_transactions * 100) if total_transactions > 0 else 0
                            
                            with col1:
                                st.metric("Total Transactions", total_transactions)
                            
                            with col2:
                                st.metric("Fraudulent", frauds, delta=f"{fraud_rate:.2f}%")
                            
                            with col3:
                                st.metric("Genuine", genuine)
                            
                            with col4:
                                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                            
                            # Visualization
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Pie chart
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sizes = [genuine, frauds]
                                labels = ['Genuine', 'Fraudulent']
                                colors = ['#09ab3b', '#d73449']
                                
                                if genuine > 0 or frauds > 0:
                                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
                                          startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
                                ax.set_title('Transaction Distribution', fontsize=12, fontweight='bold')
                                st.pyplot(fig)
                            
                            with col2:
                                # Fraud probability distribution
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.hist(results_df['Fraud_Probability'], bins=30, color='steelblue',
                                       edgecolor='black', alpha=0.7)
                                ax.set_xlabel('Fraud Probability', fontsize=11)
                                ax.set_ylabel('Number of Transactions', fontsize=11)
                                ax.set_title('Fraud Probability Distribution', fontsize=12, fontweight='bold')
                                ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
                                ax.legend()
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            # Display detailed results
                            st.markdown("---")
                            st.subheader("📋 Detailed Results")
                            
                            # Show fraudulent transactions
                            fraud_transactions = results_df[results_df['Classification'] == 'Fraud'].sort_values(
                                'Fraud_Probability', ascending=False
                            )
                            
                            if len(fraud_transactions) > 0:
                                st.write(f"**⚠️ Detected {len(fraud_transactions)} Fraudulent Transactions:**")
                                display_cols = [col for col in ['Time', 'Amount', 'Classification', 'Fraud_Probability'] 
                                              if col in fraud_transactions.columns]
                                st.dataframe(fraud_transactions[display_cols], use_container_width=True)
                            else:
                                st.success("✅ No fraudulent transactions detected!")
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="fraud_detection_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Make sure your CSV has all 30 features: Time, V1-V28, Amount")
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="footer">Made by Aditya Jalgaonkar</div>', unsafe_allow_html=True)

# ============================================================================
# PAGE 4: MODEL INFORMATION
# ============================================================================

elif page == "Model Info":
    st.title("📊 Model Information & Metrics")
    st.markdown("---")
    
    st.header("🤖 Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        st.write("""
        **Model Type:** Random Forest Classifier
        
        **Parameters:**
        - n_estimators: 100
        - random_state: 42
        - n_jobs: -1 (parallel processing)
        
        **Training Data:**
        - Total samples: 227,846
        - After SMOTE (balanced): 340,719
        - Features: 30
        - Classes: 2 (Genuine, Fraud)
        """)
    
    with col2:
        st.subheader("Data Preprocessing")
        st.write("""
        **Steps Applied:**
        
        1. **Scaling:** StandardScaler
           - Fit on training data
           - Applied to all features
        
        2. **Class Imbalance:** SMOTE
           - Oversampling minority class
           - k_neighbors: 5
        
        3. **Train-Test Split:** 80-20
           - Stratified split
           - Random state: 42
        """)
    
    st.markdown("---")
    st.header("📈 Performance Metrics")
    
    # Create metrics comparison
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity'],
        'Value': [0.9991, 0.8815, 0.8045, 0.8411, 0.9798, 0.9993],
        'Description': [
            'Overall correctness of predictions',
            'Accuracy of fraud predictions',
            'Ability to catch all frauds',
            'Harmonic mean of precision and recall',
            'Area under ROC curve',
            'Ability to identify genuine transactions'
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Visualize metrics
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity']
    values = [0.9991, 0.8815, 0.8045, 0.8411, 0.9798, 0.9993]
    colors_list = ['#09ab3b' if v >= 0.9 else '#ff9800' if v >= 0.8 else '#d73449' for v in values]
    
    bars = ax.barh(metrics, values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1.05])
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.02, i, f'{val:.4f}', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    st.header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Strengths:**
        - ✅ Very high accuracy (99.91%)
        - ✅ Excellent specificity (99.93%)
        - ✅ Strong AUC-ROC (0.9798)
        - ✅ Good precision (88.15%)
        """)
    
    with col2:
        st.warning("""
        **Considerations:**
        - ⚠️ Recall is 80.45% (misses ~20% of fraud)
        - ⚠️ Trade-off: Better recall might increase false alarms
        - ⚠️ Consider business requirements when adjusting threshold
        - ⚠️ Model tuning can improve recall if needed
        """)
    
    # Footer
    st.markdown("---")
    st.markdown('<div class="footer">Made by Aditya Jalgaonkar</div>', unsafe_allow_html=True)