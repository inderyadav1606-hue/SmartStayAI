import streamlit as st
import pandas as pd
import random
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

# --- Setup NLTK ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    return SentimentIntensityAnalyzer()

sia = setup_nltk()

# --- 1. Data Mocking ---
@st.cache_data
def load_mock_data():
    """Generates synthetic hotel reviews to mimic a real dataset."""
    cities = ['Mumbai', 'Delhi', 'Goa', 'Bengaluru']
    hotels = {
        'Mumbai': ['The Taj Mahal Palace', 'Trident Nariman Point', 'JW Marriott Juhu', 'Residency Hotel'],
        'Delhi': ['ITC Maurya', 'The Leela Palace', 'Taj Palace', 'Radisson Blu'],
        'Goa': ['W Goa', 'Taj Holiday Village', 'Novotel Resort', 'Baga Beach Hostel'],
        'Bengaluru': ['The Oberoi', 'ITC Gardenia', 'Taj West End', 'St. Mark\'s Hotel']
    }
    
    food_phrases = ["The breakfast was amazing.", "Food was terrible.", "Loved the dining options.", "Coffee was cold and eggs were bad.", "The restaurant was Michelin star quality."]
    room_phrases = ["Bed was so uncomfortable.", "Room was spacious and clean.", "Bathroom was dirty.", "Slept like a baby on the huge bed.", "AC was broken and it was noisy."]
    service_phrases = ["Staff were incredibly friendly.", "Receptionist was rude.", "Great customer service.", "They ignored us at check-in.", "Concierge was very helpful."]
    location_phrases = ["Perfect location near the metro.", "Too far from the center.", "Great views of the city.", "Located in a very crowded neighborhood.", "Walking distance to all tourist spots."]
    
    data = []
    for _ in range(500):
        city = random.choice(cities)
        hotel = random.choice(hotels[city])
        
        review_parts = random.sample([
            random.choice(food_phrases),
            random.choice(room_phrases),
            random.choice(service_phrases),
            random.choice(location_phrases)
        ], k=random.randint(2, 4))
        
        review_text = " ".join(review_parts)
        
        data.append({
            'City': city,
            'Hotel_Name': hotel,
            'Review_Text': review_text,
            'Overall_Rating': random.randint(3, 10)
        })
        
    return pd.DataFrame(data)

# --- 2. Text Mining & Aspect Extraction ---
@st.cache_data
def analyze_aspects(df):
    aspect_keywords = {
        'Food': ['breakfast', 'food', 'restaurant', 'coffee', 'dining', 'egg'],
        'Room': ['bed', 'room', 'bathroom', 'shower', 'sleep', 'ac'],
        'Service': ['staff', 'reception', 'service', 'check-in', 'concierge', 'friendly', 'rude'],
        'Location': ['location', 'subway', 'metro', 'center', 'neighborhood', 'walking', 'tourist', 'view']
    }
    
    analyzed_data = []
    
    for index, row in df.iterrows():
        text = row['Review_Text']
        sentences = sent_tokenize(text)
        
        scores = {'Food': 0, 'Room': 0, 'Service': 0, 'Location': 0}
        counts = {'Food': 0, 'Room': 0, 'Service': 0, 'Location': 0}
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            sentiment_dict = sia.polarity_scores(sentence)
            compound_score = sentiment_dict['compound']
            
            for aspect, keywords in aspect_keywords.items():
                if any(kw in sentence_lower for kw in keywords):
                    scores[aspect] += compound_score
                    counts[aspect] += 1
        
        row_dict = row.to_dict()
        for aspect in scores.keys():
            if counts[aspect] > 0:
                normalized_score = ((scores[aspect] / counts[aspect]) + 1) * 5 
                row_dict[f'{aspect}_Score'] = round(normalized_score, 1)
            else:
                row_dict[f'{aspect}_Score'] = None 
                
        analyzed_data.append(row_dict)
        
    return pd.DataFrame(analyzed_data)

# --- 3. Build the Recommendation Matrix ---
def build_hotel_profiles(analyzed_df):
    profile_df = analyzed_df.groupby(['City', 'Hotel_Name']).mean(numeric_only=True).reset_index()
    profile_df = profile_df.fillna(5.0)
    return profile_df

# --- 4. The Recommender Engine ---
def recommend_hotels(profile_df, city, weight_food, weight_room, weight_service, weight_loc):
    city_hotels = profile_df[profile_df['City'] == city].copy()
    
    total_weight = weight_food + weight_room + weight_service + weight_loc
    if total_weight == 0: total_weight = 1 
    
    city_hotels['Match_Score'] = (
        (city_hotels['Food_Score'] * weight_food) +
        (city_hotels['Room_Score'] * weight_room) +
        (city_hotels['Service_Score'] * weight_service) +
        (city_hotels['Location_Score'] * weight_loc)
    ) / total_weight
    
    return city_hotels.sort_values(by='Match_Score', ascending=False)

# --- 5. Streamlit User Interface ---
def main():
    st.set_page_config(page_title="SmartStay AI", layout="wide")
    
    st.title("🏨 SmartStay AI")
    st.markdown("""
    **ZS Data Science Project Showcase**  
    Standard 5-star ratings are flawed. A hotel might have great food but terrible beds. 
    **SmartStay AI** uses Natural Language Processing (NLP) to mine unstructured text reviews, 
    extracting sentiment for specific business aspects (Food, Room, Service, Location) to provide hyper-personalized recommendations.
    """)
    
    st.write("---")
    
    with st.spinner("Mining text data and extracting aspects..."):
        raw_df = load_mock_data()
        analyzed_df = analyze_aspects(raw_df)
        hotel_profiles = build_hotel_profiles(analyzed_df)
    
    st.sidebar.header("🎯 Your Preferences")
    selected_city = st.sidebar.selectbox("Select a City", ['Mumbai', 'Delhi', 'Goa', 'Bengaluru'])
    
    st.sidebar.markdown("### What is most important to you?")
    st.sidebar.caption("Scale 0 (Don't care) to 10 (Must be perfect)")
    
    w_food = st.sidebar.slider("🍔 Food & Breakfast", 0, 10, 5)
    w_room = st.sidebar.slider("🛏️ Room Comfort & Cleanliness", 0, 10, 8)
    w_service = st.sidebar.slider("🛎️ Service & Staff", 0, 10, 5)
    w_loc = st.sidebar.slider("📍 Location", 0, 10, 7)
    
    recommendations = recommend_hotels(hotel_profiles, selected_city, w_food, w_room, w_service, w_loc)
    
    st.subheader(f"Top Recommended Hotels in {selected_city}")
    top_3 = recommendations.head(3)
    
    cols = st.columns(3)
    for i, (index, row) in enumerate(top_3.iterrows()):
        with cols[i]:
            st.metric(label=f"#{i+1}: {row['Hotel_Name']}", value=f"{row['Match_Score']:.1f}/10 Match")
            st.markdown(f"""
            * **Food:** {row['Food_Score']:.1f}/10
            * **Room:** {row['Room_Score']:.1f}/10
            * **Service:** {row['Service_Score']:.1f}/10
            * **Location:** {row['Location_Score']:.1f}/10
            """)
            
            st.markdown("**Sample Review (Text Mining):**")
            sample_reviews = analyzed_df[analyzed_df['Hotel_Name'] == row['Hotel_Name']]['Review_Text'].head(1).values
            if len(sample_reviews) > 0:
                st.caption(f'"{sample_reviews[0]}"')
                
    st.write("---")
    with st.expander("📊 View Full Analytical Matrix (Structured Data from Unstructured Text)"):
        st.dataframe(hotel_profiles[hotel_profiles['City'] == selected_city].style.highlight_max(axis=0, subset=['Food_Score', 'Room_Score', 'Service_Score', 'Location_Score']))
        
if __name__ == "__main__":
    main()