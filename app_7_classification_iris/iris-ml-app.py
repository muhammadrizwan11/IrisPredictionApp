






import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import time

st.write("""
<style>
.title {
    animation: color_change 5s infinite;
    text-align: center;
    font-weight: bold;
    font-size: 45px;    
    text-shadow: 4px 4px 8px #000000;
    color: white;
    background-color: white;
    background-image: linear-gradient(57deg, #ff0000, #ff00ff);
    background-size: 100%;
         
   
         
}

.subheader {
    text-align: center;
}
    
@keyframes color_change {
    0% { color: green; }
    
    25% { color: blue; }
    50% { color: red; }
    75% { color: gray; }
    100% { color: white; }
}
</style>

<h1 class="title">🌸 Iris Flower Prediction App 🌼</h1>

This app predicts the **Iris flower** type!
""", unsafe_allow_html=True)

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

# Add a prediction button
if st.button('Predict'):
    with st.spinner(text="In progress..."):
        time.sleep(3)
        prediction = clf.predict(df)
        prediction_proba = clf.predict_proba(df)
        st.success("Done")

    st.subheader('Prediction 🌸')
    st.write(iris.target_names[prediction])

    # Display images based on prediction
    if prediction == 0:  # Setosa
        st.image("setosa.jpeg", width=200)
    elif prediction == 1:  # Versicolor
        st.image("versicolor.jpeg", width=200)
    elif prediction == 2:  # Virginica
        st.image("virginica.jpeg", width=200)
    st.balloons()
    st.snow()
    # Display prediction table with emojis


    

# # Other Streamlit functionalities
# bar = st.progress(50)
# time.sleep(3)
# bar.progress(100)

# with st.status("Authenticating...") as s:
#     time.sleep(2)
#     st.write("Some long response.")
#     s.update(label="Response")


# st.toast("Warming up...")
# st.error("Error message")
# st.warning("Warning message")
# st.info("Info message")
# st.success("Success message")
