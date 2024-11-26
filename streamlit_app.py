import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import os
from PIL import Image
import pickle

image_folder = 'samples/'

def compute_result(json):
    with open('models/col_names.pkl', 'rb') as col_file:
        column_names = pickle.load(col_file)

    with open('models/lab_enc.pkl', 'rb') as enc_file:
        label_encodings = pickle.load(enc_file)

    with open('models/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    def encode_columns(data, cols_enc, encodings):
        for col in cols_enc:
            if col in encodings:
                mapping = {category: code for code, category in enumerate(encodings[col])}
                data[col] = data[col].map(mapping)
        return data
    
    return model.predict(encode_columns(pd.DataFrame(json), column_names, label_encodings))



st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f7f9fc;
        color: #000000;
    }
    .sidebar .sidebar-content h1, .sidebar-content h2, .sidebar-content h3, .sidebar-content h4 {
        color: #4b6584;
        font-family: Arial, Helvetica, sans-serif;
        font-weight: bold;
    }
    .css-1aumxhk {
        font-family: 'Arial', sans-serif;
        font-size: 18px;
        color: #1a202c !important;
    }
    .css-qbe2hs:hover {
        color: #ffffff !important;
        background-color: #4b6584 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation with improved styling
with st.sidebar:
    st.image("data/prof.jpg", caption="KAVI", width=100,)
    st.title("Navigation Menu")
    choice = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data", "EDA - Visual", "Prediction", "NLP"],
        icons=["house", "table", "bar-chart", "robot", "book"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f7f9fc"},
            "icon": {"color": "orange", "font-size": "18px"},
            "nav-link": {"font-size": "18px", "text-align": "left", "margin": "1px", "color": "#000000"},
            "nav-link-selected": {"background-color": "#4b6584", "color": "#ffffff"},
        },
    )
if choice == "Home":
    # Display model performance
    model_name = "DecisionTreeClassifier"  
    accuracy = 0.9934 
    precision = 0.98  
    recall = 0.94  

    metrics = ["Accuracy", "Precision", "Recall"]
    values = [accuracy, precision, recall]

    # Generate the plot
    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=['blue', 'green', 'orange'])
    plt.title(f"{model_name} Performance Metrics")
    plt.ylabel("Values")
    plt.ylim(0.5, 1)
    # Display model details
    st.title("Model Performance Summary")
    st.write(f"**Model Name**: {model_name}")
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")

    # Save the plot to a BytesIO object instead of a file
    from io import BytesIO
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)  # Go to the start of the BytesIO object

    # Display the image in Streamlit
    st.image(img_stream, caption="Model Performance Visualization", use_column_width=True)

    
# Data Page
elif choice == "Data":
    st.title("Data Section")
    
    df1 = pd.read_csv('data/s1.csv')
    df2 = pd.read_csv('data/s2.csv')
    df3 = pd.read_csv('data/s3.csv')
    st.write("Dataset Preview")
    st.dataframe(df1.head(500))
    st.write("Summary Statistics")
    st.write(df2.head(500))
    st.write("Missing Values")
    st.write(df3.head(500))

# EDA - Visual Page
elif choice == "EDA - Visual":
    st.title("Exploratory Data Analysis")
    
    
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if image_files:
        for i in range(0, len(image_files), 2):
            cols = st.columns(2)
            for col, image_file in zip(cols, image_files[i:i + 2]):
                image_path = os.path.join(image_folder, image_file)
                img = Image.open(image_path)
                col.image(img, caption=image_file, use_column_width=True)
    else:
        st.write("No images found in the specified folder.")

# Prediction Page
if choice == "Prediction":
    st.title("Prediction Section")

    # Create a form-like structure for inputs
    with st.form("prediction_form"):
        EXT_SOURCE_3 = st.number_input("EXT_SOURCE_3", value=0.0)
        EXT_SOURCE_2 = st.number_input("EXT_SOURCE_2", value=0.0)
        DAYS_BIRTH = st.number_input("DAYS_BIRTH", value=0)
        DAYS_LAST_PHONE_CHANGE = st.number_input("DAYS_LAST_PHONE_CHANGE", value=0)
        REGION_RATING_CLIENT_W_CITY = st.number_input("REGION_RATING_CLIENT_W_CITY", value=0)
        REGION_RATING_CLIENT = st.number_input("REGION_RATING_CLIENT", value=0)
        NAME_CONTRACT_STATUS = st.text_input("NAME_CONTRACT_STATUS", value="Enter status")
        NAME_INCOME_TYPE = st.text_input("NAME_INCOME_TYPE", value="Enter income type")
        DAYS_ID_PUBLISH = st.number_input("DAYS_ID_PUBLISH", value=0)

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = [{
                'EXT_SOURCE_3': EXT_SOURCE_3,
                'EXT_SOURCE_2': EXT_SOURCE_2,
                'DAYS_BIRTH': DAYS_BIRTH,
                'DAYS_LAST_PHONE_CHANGE': DAYS_LAST_PHONE_CHANGE,
                'REGION_RATING_CLIENT_W_CITY': REGION_RATING_CLIENT_W_CITY,
                'REGION_RATING_CLIENT': REGION_RATING_CLIENT,
                'NAME_CONTRACT_STATUS': NAME_CONTRACT_STATUS,
                'NAME_INCOME_TYPE': NAME_INCOME_TYPE,
                'DAYS_ID_PUBLISH': DAYS_ID_PUBLISH
            }]
            
            # Call the compute_result function
            try:
                result = compute_result(input_data)
                st.write("Prediction Result:", result[0])
            except Exception as e:
                st.error(f"Error in prediction: {e}")

# NLP Page
elif choice == "NLP":
    st.title("NLP Section")
    user_input = st.text_area("Enter your text for analysis")
    
    if st.button("Predict"):
        if user_input:
            from textblob import TextBlob
            sentiment = TextBlob(user_input).sentiment
            st.write("Sentiment Analysis")
            if sentiment.polarity > 0:
                st.write("Positive")
            elif sentiment.polarity == 0:
                st.write("Neutral")
            else:
                st.write("Negative")
            st.write(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")
        else:
            st.warning("Please enter some text before clicking Predict.")
