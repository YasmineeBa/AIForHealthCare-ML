import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, LabelEncoder
from sklearn.metrics import  mean_squared_error, classification_report, r2_score, roc_curve, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from imblearn.over_sampling import SMOTE
from ydata_profiling import ProfileReport
import plotly.express as px
from streamlit_pandas_profiling import st_profile_report


def file_to_df(file):
    name = file.name
    extension = name.split(".")[-1]
    
    if extension == "csv":
        df = pd.read_csv(file)
    elif extension == "tsv":
        df = pd.read_csv(file, sep="\t")
    elif extension == "xml":
        df = pd.read_xml(file)
    elif extension == "xlsx":
        df = pd.read_excel(file)
    elif extension == "json":
        df = pd.read_json(file)
    
    return df

def CategoricalToNumerical(df):
   le = LabelEncoder()

# Encode categorical columns
   for col in df.select_dtypes(include='object').columns:
     df[col] = le.fit_transform(df[col].astype(str))
   return df


def fillmissingvalue(data):
    for label, content in data.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Fill missing numeric values with median since it's more robust than the mean
                data[label] = content.fillna(content.median())

    return data

def plot_roc_curve(fpr, tpr):
    """
    Plots a ROC curve given the false positve rate (fpr) and 
    true postive rate (tpr) of a classifier.
    """
    plt.figure()
    # Plot ROC curve
    plt.plot(fpr, tpr, color='orange', label='ROC')
    # Plot line with no predictive power (baseline)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label='Guessing')
    # Customize the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    st.pyplot(plt)
    
#setup the layout of the app
st.set_page_config("ML app", layout='wide')
st.title("AI For Health Care")
st.text("Use the menu at left to select machine learning type.")

st.sidebar.title('Main Menu')
option = st.sidebar.radio('1- Select the machine learning type :', options=["Classification", "Regression"])

if option == "Classification":

    st.title('🫀 Heart Desease Predictor')
    st.markdown("<h4>Learn how to build a machine learning-powered classification model to predict whether a patient has heart disease or not based on their health parameters and heart measurements.</h4>", unsafe_allow_html=True)

    st.markdown("""
        <h4>1. Problem Definition</h4>
        <p>In our case, the problem we will be exploring is binary classification (a sample can only be one of two things).</p>
        <p>This is because we're going to be using a number of different features (pieces of information) about a person to predict whether they have heart disease or not.</p>
        <p>In a statement:</p>
        <blockquote>
            Given clinical parameters about a patient, can we predict whether or not they have heart disease?
        </blockquote>
        """, unsafe_allow_html=True)
    
    st.subheader("2. Data")

    uploaded_data = st.file_uploader('Select the machine learning task then Upload your dataset :', type=["csv","tsv","xlsx","json","xml"])

    if uploaded_data is not None:
        
        df = file_to_df(uploaded_data)
        st.dataframe(df)
        st.markdown("<h4>2.1 Data Visualization</h4>", unsafe_allow_html=True)

        #display the profiledata
        tab1, tab2 = st.tabs([f"{uploaded_data.name.split(".")[0]} Profiling", "3D Scatter plot"])
        with tab1:
            pr = ProfileReport(df)
            st_profile_report(pr)
            
        with tab2:

            columns = df.columns

            col_fea_1, col_fea_2, col_fea_3 = st.columns(3)

            with col_fea_1:
                fea_1 = st.selectbox("Please select the 1st numerical feature", columns)
            with col_fea_2:
                fea_2 = st.selectbox("Please select the 2nd numerical feature", columns)
            with col_fea_3:
                fea_3 = st.selectbox("Please select the 3rd numerical feature", columns)
            
            viz_button = st.button("Visualize")
            
            if viz_button:
                fig_3d = px.scatter_3d(df, x=fea_1, y=fea_2, z=fea_3) 
                st.plotly_chart(fig_3d, theme=None)

        st.markdown("""
                    <h4>2.2 Data Modeling </h4>
                    <p>Convert categorical object into numeric and Filling the missed values if exist</p>""",
                    unsafe_allow_html=True
                    )
        #One way to help turn all of our data into numbers is to convert the columns with the string datatype into a category datatype.
        # To do this we can use the pandas types API which allows us to interact and manipulate the types of data.
        df_tmp = CategoricalToNumerical(df)
        df_tmp = fillmissingvalue(df_tmp)

        st.dataframe(df_tmp)
        
        # Show the target and features 
        st.subheader("3. Split Data:")
            #select the target column
        columns = df.columns
        target_column = st.selectbox("Select the target column :", options=columns)
        
            # Split features and target
        X = df_tmp.drop(columns=[target_column])
        y = df_tmp[target_column]

        st.write("Features (X) ")
        st.dataframe(X.head())
        st.write("Target (y) ")
        st.dataframe(y.head())

        # Define model options for classification and regression
        classification_models = {
            'Logistic Regression': LogisticRegression(),
            'Decision Tree Classifier': DecisionTreeClassifier(),
            'Support Vector Classifier': SVC(),
            'Random Forest Classifier': RandomForestClassifier()
        }

    
        models =[]
        # Model comparison options
        st.subheader("4. Model Choice:")
        st.markdown("""
                    #### Now we've got our data prepared, we can start to fit models. We'll be using the following and comparing their results:

                    - **Logistic Regression** - `LogisticRegression()`
                    - **K-Nearest Neighbors** - `KNeighboursClassifier()`
                    - **RandomForest** - `RandomForestClassifier()`
                    - **Support Vector Classifier** - `SVC()`                   
                    """)

        
        st.markdown("""
                    #### Select the models you want to include:
                    """)
        for model_name, model in classification_models.items():
            if st.checkbox(f"{model_name}"):
                models.append((model_name, model))
        
          # show One-vs-Rest and One-vs-One options if some models are selected
        st.markdown("#### Select One-vs-Rest or/and One-vs-One if you want to (ensure that you select a model before):")

        if st.checkbox("Include One-vs-Rest Classifier"):
            # Apply One-vs-Rest to selected models
            models = [(f'OneVsRestClassifier ({name})', OneVsRestClassifier(model)) for name, model in models]

        if st.checkbox("Include One-vs-One Classifier"):
            # Apply One-vs-One to selected models
            models = [(f'OneVsOneClassifier ({name})', OneVsOneClassifier(model)) for name, model in models]
        
        # Polynomial features option
        st.markdown("""
                    #### Feature Engineer and Data Augmentation :
                    """, unsafe_allow_html=True)
        apply_poly = st.checkbox("Apply Polynomial Features")
        if apply_poly:
            degree = st.slider("Select the degree of polynomial features", 2, 5, 2)
            poly = PolynomialFeatures(degree)
            X = poly.fit_transform(X)

        # SMOTE option for data augmentation
        apply_smote = st.checkbox("Apply SMOTE for data augmentation")
        if apply_smote:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Ignore any warnings during SMOTE
                smote = SMOTE()
                X, y = smote.fit_resample(X, y)

        # Split data into training and test sets
        np.random.seed(42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Normalize the features
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        st.subheader("5. Model Evaluation")

        if models:
            st.markdown("<h5> 5.1 Classificatin Report</h5>", unsafe_allow_html=True)

            results = []

            for name, model in models:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                
                report = classification_report(y_test, predictions, output_dict=True)
                st.text(f"### {name} - Classification Report")
                st.text(classification_report(y_test, predictions))

                acc = report['accuracy']
                results.append((name, acc))


        
            
            results_df = pd.DataFrame(results, columns=['Model', 'Accuracy'])
            st.markdown("#### Classification Accuracy Comparison")
           
            # Show a table of results
            #st.dataframe(results_df)
            
            # Plot the comparison
            plt.figure(figsize=(3, 3))
            
            sns.barplot(x='Model', y='Accuracy', data=results_df)
            plt.ylabel('Accuracy')
            
            plt.xticks(rotation=45, ha='right')
            plt.title('Model Comparison')
            st.pyplot(plt)

            # i choosed the Random forest Classifier model to display the roc curve and confusion matrix for predicted model result
            for name, model in models:
                if name == 'Random Forest Classifier':
                    st.markdown("<h5> 5.2 ROC Curve for Random Forest Classifier</h5>", unsafe_allow_html=True)
                    with st.expander("Area Under Receiver Operating Characteristic (ROC) Curve"):
                        st.write("""

                            It's usually referred to as AUC for Area Under Curve and the curve they're talking about is the Receiver Operating Characteristic or ROC for short.

                            ROC curves are a comparison of true positive rate (tpr) versus false positive rate (fpr).

                            For clarity:

                            - True positive = model predicts 1 when truth is 1
                            - False positive = model predicts 1 when truth is 0
                            - True negative = model predicts 0 when truth is 0
                            - False negative = model predicts 0 when truth is 1

                        """)
                    y_probs = model.predict_proba(X_test)
                    # Keep the probabilites of the positive class only
                    y_probs = y_probs[:, 1]
                    # Calculate fpr, tpr and thresholds
                    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
                    plot_roc_curve(fpr, tpr)

                    st.markdown("<h5> Confusion Matrix </h5>", unsafe_allow_html=True)

                    # Create the confusion matrix plot
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test, ax=ax)
                    # Display the plot in Streamlit
                    st.pyplot(fig)


    else:
        st.info("please select your dataset to get started !")
    
else:
    st.title('👨🏻‍⚕️ Medical Cost Prediction ')
    st.markdown("<h4>Learn how to build a machine learning-powered regression model to predict Medical Cost.</h4>", unsafe_allow_html=True)

    st.subheader("2. Data")

    uploaded_data = st.file_uploader('Select the machine learning task then Upload your dataset :', type=["csv","tsv","xlsx","json","xml"])

    if uploaded_data is not None:
        
        df = file_to_df(uploaded_data)
        st.dataframe(df)
        st.markdown("<h4>2.1 Data Visualization</h4>", unsafe_allow_html=True)

        #display the profiledata
        tab1, tab2 = st.tabs([f"{uploaded_data.name.split(".")[0]} Profiling", "3D Scatter plot"])
        with tab1:
            pr = ProfileReport(df)
            st_profile_report(pr)
            
        with tab2:

            columns = df.columns

            col_fea_1, col_fea_2, col_fea_3 = st.columns(3)

            with col_fea_1:
                fea_1 = st.selectbox("Please select the 1st numerical feature", columns)
            with col_fea_2:
                fea_2 = st.selectbox("Please select the 2nd numerical feature", columns)
            with col_fea_3:
                fea_3 = st.selectbox("Please select the 3rd numerical feature", columns)
            
            viz_button = st.button("Visualize")
            
            if viz_button:
                fig_3d = px.scatter_3d(df, x=fea_1, y=fea_2, z=fea_3) 
                st.plotly_chart(fig_3d, theme=None)

        st.markdown("""
                    <h4>2.2 Data Modeling </h4>
                    <p>Convert categorical object into numeric and Filling the missed values if exist</p>""",
                    unsafe_allow_html=True
                    )
        #One way to help turn all of our data into numbers is to convert the columns with the string datatype into a category datatype.
        # To do this we can use the pandas types API which allows us to interact and manipulate the types of data.
        df_tmp = CategoricalToNumerical(df)
        df_tmp = fillmissingvalue(df_tmp)

        st.dataframe(df_tmp)
        
        # Show the target and features 
        st.subheader("3. Split Data:")
            #select the target column
        columns = df.columns
        target_column = st.selectbox("Select the target column :", options=columns)
        
            # Split features and target
        X = df_tmp.drop(columns=[target_column])
        y = df_tmp[target_column]

        st.write("Features (X) ")
        st.dataframe(X.head())
        st.write("Target (y) ")
        st.dataframe(y.head())

        
        regression_models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regressor': DecisionTreeRegressor(),
            'Support Vector Regressor': SVR(),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100)
        }
        models =[]
        # Model comparison options
        st.subheader("4. Model Choice:")
        st.markdown("""
                    #### Now we've got our data prepared, we can start to fit models. We'll be using the following and comparing their results:

                    - **Linear Regression** - `LinearRegression()`
                    - **K-Nearest Neighbors** - `KNeighboursClassifier()`
                    - **RandomForestRegressor** - `RandomForesRegressor()`
                    - **Support Vector CRegressor** - `SVR()`                   
                    """)

    
        for model_name, model in regression_models.items():
            if st.checkbox(f"{model_name}"):
                models.append((model_name, model))
        
        #we dont apply SMOTE data augmentation in regression task
        # Polynomial features option
        st.markdown("""
                    #### Feature Engineer :
                    """, unsafe_allow_html=True)
        apply_poly = st.checkbox("Apply Polynomial Features")
        if apply_poly:
            degree = st.slider("Select the degree of polynomial features", 2, 5, 2)
            poly = PolynomialFeatures(degree)
            X = poly.fit_transform(X)

        
        # Split data into training and test sets
        np.random.seed(42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Normalize the features
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        st.subheader("5. Model Evaluation")

        if models:
            st.markdown("<h5> 5.1 Regression Result</h5>", unsafe_allow_html=True)

            results = []

            for name, model in models:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                # For regression, calculate mean squared error and R2 score
                mse = mean_squared_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                st.text(f"### {name} - Regression Results")
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"R2 Score: {r2:.4f}")
                
                # Append results for comparison
                results.append((name, mse, r2))


        
    
            results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R2'])
            st.markdown("#### Regression Model Comparison")
        
            # Show a table of results
            #st.dataframe(results_df)
            
            # Plot the comparison
            plt.figure(figsize=(3, 3))
            
            sns.barplot(x='Model', y='MSE', data=results_df)
            plt.ylabel('Mean Squared Error')
            plt.xticks(rotation=45, ha='right')
            plt.title('Model Comparison')
            st.pyplot(plt)


    else:
        st.info("please select your dataset to get started !")

st.markdown("<center><h6>app made by yasmine baroud </h6></center>", unsafe_allow_html=True)
