import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import os


# ===============================
# Load trained model
# ===============================
model = load_model('FV.h5')


# ===============================
# Labels
# ===============================
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower',
    8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno',
    16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas',
    25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
    32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}


fruits = ['Apple', 'Banana', 'Grapes', 'Kiwi', 'Lemon', 'Mango', 'Orange', 'Paprika',
          'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']


vegetables = ['Beetroot', 'Bell Pepper', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower',
              'Chilli Pepper', 'Corn', 'Cucumber', 'Eggplant', 'Garlic', 'Ginger',
              'Jalepeno', 'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans',
              'Spinach', 'Sweetcorn', 'Sweetpotato', 'Tomato', 'Turnip']


# ===============================
# Calorie Lookup Dictionary
# ===============================
calorie_dict = {
    "apple": 37,
    "banana": 51,
    "beetroot": 42,
    "bell pepper": 15,
    "cabbage": 27,
    "capsicum": 21,
    "carrot": 10,
    "cauliflower": 30,
    "chilli pepper": 21,
    "corn": 54,
    "cucumber": 15,
    "eggplant": 8,
    "garlic": 149,
    "ginger": 80,
    "grapes": 72,
    "jalepeno": 27,
    "kiwi": 42,
    "lemon": 15,
    "lettuce": 10,
    "mango": 39,
    "onion": 43,
    "orange": 47,
    "paprika": 21,
    "pear": 57,
    "peas": 70,
    "pineapple": 58,
    "pomegranate": 79,
    "potato": 97,
    "raddish": 33,
    "soy beans": 140,
    "spinach": 24,
    "sweetcorn": 54,
    "sweetpotato": 86,
    "tomato": 18,
    "turnip": 28,
    "watermelon": 38
}

# ===============================
# Advantage Dictionary
# ===============================
advantage_dict = {
    "apple": "Rich in fiber and antioxidants.",
    "banana": "Good source of potassium and energy.",
    "beetroot": "Supports healthy blood pressure.",
    "bell pepper": "High in vitamin C, boosts immunity.",
    "cabbage": "Contains disease-fighting phytochemicals.",
    "capsicum": "Improves eye health with vitamin A.",
    "carrot": "Great for vision due to beta-carotene.",
    "cauliflower": "High in fiber, supports digestion.",
    "chilli pepper": "May boost metabolism.",
    "corn": "Good source of carbohydrates and fiber.",
    "cucumber": "Hydrating and low calorie.",
    "eggplant": "Contains heart-healthy antioxidants.",
    "garlic": "Supports immune system and heart health.",
    "ginger": "Has anti-inflammatory properties.",
    "grapes": "Rich in antioxidants, supports heart health.",
    "jalepeno": "May aid digestion and metabolism.",
    "kiwi": "High vitamin C content.",
    "lemon": "Excellent source of vitamin C.",
    "lettuce": "Low calorie, hydrating leafy green.",
    "mango": "Rich in vitamins A and C.",
    "onion": "Contains compounds with anti-inflammatory effects.",
    "orange": "Vitamin C rich, supports immunity.",
    "paprika": "Contains antioxidants and vitamins.",
    "pear": "Good source of dietary fiber.",
    "peas": "Rich in plant-based protein.",
    "pineapple": "Aids digestion with bromelain enzyme.",
    "pomegranate": "High in antioxidants.",
    "potato": "Good source of energy and potassium.",
    "raddish": "Helps with digestion and detoxification.",
    "soy beans": "Excellent plant protein source.",
    "spinach": "Rich in iron and vitamins.",
    "sweetcorn": "Provides fiber and B vitamins.",
    "sweetpotato": "High beta-carotene and fiber.",
    "tomato": "Contains lycopene, good for heart health.",
    "turnip": "Supports immune system with vitamin C.",
    "watermelon": "Hydrating and rich in vitamins A and C."
}

# ===============================
# Disadvantage Dictionary
# ===============================
disadvantage_dict = {
    "apple": "May cause allergies in sensitive individuals.",
    "banana": "High sugar content for diabetics.",
    "beetroot": "May cause kidney stones in susceptible people.",
    "bell pepper": "Can cause digestive upset in some individuals.",
    "cabbage": "Excess intake may lead to bloating or gas.",
    "capsicum": "Can cause heartburn if consumed in excess.",
    "carrot": "Excess can cause carotenemia, changing skin color.",
    "cauliflower": "May cause gas or digestive discomfort.",
    "chilli pepper": "Can irritate digestive tract and cause heartburn.",
    "corn": "High starch content may affect blood sugar.",
    "cucumber": "May cause allergy or indigestion in some people.",
    "eggplant": "Contains solanine which can be toxic in large amounts.",
    "garlic": "May cause bad breath or digestive issues.",
    "ginger": "Excess can cause heartburn or stomach upset.",
    "grapes": "High sugar content, can raise blood sugar levels.",
    "jalepeno": "May irritate stomach lining or cause heartburn.",
    "kiwi": "May cause allergic reactions in some individuals.",
    "lemon": "Highly acidic, can erode tooth enamel.",
    "lettuce": "Low nutritional density, can cause digestive sensitivity.",
    "mango": "High sugar content, not ideal for diabetics.",
    "onion": "May cause bad breath and digestive issues.",
    "orange": "Acidic; can aggravate acid reflux.",
    "paprika": "Can cause irritation if consumed excessively.",
    "pear": "May cause digestive gas in some people.",
    "peas": "Can lead to bloating or gas.",
    "pineapple": "Highly acidic; may irritate mouth and stomach.",
    "pomegranate": "May interact with certain medications.",
    "potato": "High glycemic index; can spike blood sugar.",
    "raddish": "May cause gas or exacerbate thyroid problems.",
    "soy beans": "Contains phytoestrogens which may affect hormone levels.",
    "spinach": "High in oxalates; can contribute to kidney stones.",
    "sweetcorn": "High starch content may affect blood sugar.",
    "sweetpotato": "Excess intake can cause digestive discomfort.",
    "tomato": "Can aggravate acid reflux or cause allergies.",
    "turnip": "May cause bloating or gas in sensitive individuals.",
    "watermelon": "High sugar content; may cause digestive upset."
}


# ===============================
# Calorie Retrieval Function
# ===============================
def fetch_calories(food_name):
    food_key = food_name.strip().lower()
    calories = calorie_dict.get(food_key, None)
    if calories is not None:
        return f"{calories} kcal (per 100 grams)"
    else:
        return "N/A (calorie info not found)"


# ===============================
# Process uploaded image
# ===============================
def processed_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(" ".join(str(x) for x in y_class))
    res = labels[y]
    return res.capitalize()


# ===============================
# Streamlit app
# ===============================
def run():
    st.title(" Fruit and Vegetable Classifier")
    st.text("Upload the image of a fruit or vegetable")

    img_file = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])

    if img_file is not None:
        try:
            img = Image.open(img_file)
            img = img.resize((250, 250))
            st.image(img)

            if not os.path.exists("./upload_image"):
                os.makedirs("./upload_image")

            save_image_path = "./upload_image/" + img_file.name
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())

            result = processed_image(save_image_path)
            if result in vegetables:
                st.info("Category : Vegetable")
            else:
                st.info("Category : Fruit")

            st.success("Predicted Result : " + result)

            cal = fetch_calories(result)
            st.warning(f"Calories: {cal}")

            adv = advantage_dict.get(result.lower(), "Advantage info not found.")
            disadv = disadvantage_dict.get(result.lower(), "Disadvantage info not found.")

            st.info("Advantage: " + adv)
            st.info("Disadvantage: " + disadv)

            if "N/A" in cal:
                st.info("If calorie info is missing, update the calorie_dict in this script.")

        except Exception as e:
            st.error(f"Error processing image: {e}")


if __name__ == "__main__":
    run()
