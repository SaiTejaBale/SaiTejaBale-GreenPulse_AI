import streamlit as st
import pandas as pd
import plotly.express as px
import emoji
import torch
import os
import google.generativeai as genai
from transformers import pipeline, RobertaForSequenceClassification, RobertaTokenizer, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import time
import nltk
import requests
import toml
from nltk.corpus import stopwords

# Page configuration
st.set_page_config(
    page_title="Energy Policy Sentiment Analysis",
    page_icon="⚡",
    layout="wide"
)

# CX = "04ad881c61b5f4ae0"
# API_KEY = "AIzaSyA4czfZ7wYpJRazh8Q5jZgkZfAp0Sj9AyQ"
# GEMINI_API_KEY = "AIzaSyDUifpjjNWmeZZxwA1Oq9Wu-2DZQ8nj43w"

# Accessing Google and Gemini API credentials from Streamlit's secrets manager
google_api_key = st.secrets["GOOGLE"]["API_KEY"]
google_cx = st.secrets["GOOGLE"]["CX"]
gemini_api_key = st.secrets["GEMINI"]["GEMINI_API_KEY"]

# Debug: Print the keys to ensure they are being loaded
# st.write("Google API Key:", google_api_key)
# st.write("Google CX:", google_cx)
# st.write("Gemini API Key:", gemini_api_key)

# Add caching for performance improvement
@st.cache_resource
def load_models():
    """Load and cache local RoBERTa model and Gemini for insights"""
    genai.configure(api_key=gemini_api_key)  # Set the Gemini API key

    try:
        model_path = "C://Users//sanja//OneDrive//Documents//projR//roberta_local"
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)

        gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")

        return tokenizer, model, gemini_model

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Call the function and capture the return values
tokenizer, model, gemini_model = load_models()
        
# Load data with caching
@st.cache_data(ttl=3600)
def load_data():
    """Load and preprocess the data with caching"""
    try:
        df = pd.read_excel('combined_cleaned_data.xlsx')
        
        # Clean column names and strings
        df.columns = df.columns.str.strip().str.lower()
        df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
        if 'platform' in df.columns:
            df['platform'] = df['platform'].str.strip().str.lower()
        
        df['sentiment_category'] = df['sentiment'].map({2: 'Positive', 1: 'Neutral', 0: 'Negative'}).fillna('Neutral')
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df['year'] = df['date'].dt.year
        invalid_dates = df[pd.to_datetime(df['date'], errors='coerce').isna()]['date'].unique()
        # st.write("⚠️ Unparsable date formats found:", invalid_dates)

        
        # Ensure keywords are in lowercase for consistent filtering
        if 'keyword' in df.columns:
            df['keyword'] = df['keyword'].str.lower()
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Initialize models and data
tokenizer, model, generator = load_models()
df = load_data()

# Text preprocessing function
def preprocess_text(text):
    """Clean and normalize text input"""
    if not text:
        return ""
    # Convert to string if not already
    text = str(text)
    # Basic cleaning
    text = text.strip().lower()
    return text

# Improved sentiment classification with error handling
def classify_sentiment_with_roberta(text):
    """Classify sentiment using RoBERTa model"""
    if not text or len(text.strip()) < 3:
        return 'Neutral'  # Default for empty or very short text
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        sentiment = torch.argmax(logits, dim=-1).item()
        return {0: 'Negative', 1: 'Neutral', 2: 'Positive'}.get(sentiment, 'Neutral')
    except Exception as e:
        st.warning(f"Error during sentiment classification: {e}")
        return 'Neutral'  # Fallback

def get_local_model_insights(text, tokenizer, model):
    """Extract detailed insights from the local RoBERTa model"""
    if not text or len(text.strip()) < 3:
        return None
    
    insights = {}
    
    try:
        # Get token-level insights
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Get probabilities for all classes
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            sentiment_probs = {
                'Negative': float(probabilities[0]),
                'Neutral': float(probabilities[1]),
                'Positive': float(probabilities[2])
            }
            insights['confidence_scores'] = sentiment_probs
            
            # Get the predicted class
            predicted_class = torch.argmax(probabilities).item()
            insights['sentiment'] = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}.get(predicted_class)
            
            # Calculate sentiment intensity (distance from neutral)
            if predicted_class == 0:  # Negative
                intensity = sentiment_probs['Negative'] - sentiment_probs['Neutral']
            elif predicted_class == 2:  # Positive
                intensity = sentiment_probs['Positive'] - sentiment_probs['Neutral']
            else:  # Neutral
                intensity = 0.0
            insights['intensity'] = round(float(intensity), 3)
            
            # Get key phrases (simplified approximation)
            tokens = tokenizer.tokenize(text)
            if len(tokens) > 5:  # Only if we have enough tokens
                # This is a simplified approach - for a real implementation you'd need
                # to use attention weights or feature attribution methods like LIME/SHAP
                # For now, let's just take the first 3-5 non-stopwords as key phrases
                common_stopwords = set(['the', 'a', 'an', 'in', 'of', 'to', 'and', 'is', 'are'])
                key_tokens = [t for t in tokens if t.lower() not in common_stopwords][:5]
                insights['key_phrases'] = key_tokens
                
        return insights
        
    except Exception as e:
        print(f"Error extracting local model insights: {e}")
        return None

def generate_insights_with_gemini(user_comment, sentiment, keyword, google_cx, google_api_key):
    """Generate insightful policy analysis using Gemini AI and real policy references."""
    
    # Check for comment validity
    if not user_comment or len(user_comment.strip()) < 10:
        return "Please provide a longer comment for meaningful insights."
    
    # Clean inputs
    user_comment = preprocess_text(user_comment)
    keyword = preprocess_text(keyword) if keyword else "energy policy"

    # Perform real-time policy reference search
    search_query = f"{keyword} subsidy site:.gov OR site:.org"
    references = perform_google_cse_search(search_query, google_cx, google_api_key)
    formatted_refs = "\n".join(
        [f"- {ref['snippet']} ({ref['link']})" for ref in references[:3]]
    )

    # Construct the Gemini prompt
    prompt = f"""
### 🎯 Goal
Generate a professional, policy-oriented analysis of the public comment below. Use ONLY what's in the comment for the insight. Include real-world policy relevance and a specific recommendation, incorporating citations when helpful.

---

**Keyword:** {keyword}  
**User Comment:** "{user_comment}"  
**Sentiment:** {sentiment}

---

### 📄 Format:

**Policy Insight:**  
(Summarize the user's concern based only on their comment. Make it clear and concise.)

**Policy Relevance:**  
(Explain how the concern fits into existing or emerging energy policy issues. Make it understandable for someone not directly involved in energy policy with a focus on the economic and technical challenges, including scalability and feasibility in different regions..)

**Policy-Specific Recommendation:**  
(Suggest realistic policies that address the user's concern directly. Discuss the economic impact, stakeholder benefits, and how the policies can be implemented at scale. Offer comparisons with other successful policies.
**Policy-Specific Recommendation Points:**

1. Projects in regions with documented inconsistent wind speeds receive higher subsidies to offset potential revenue losses.

2. Projects incorporating advanced energy storage solutions (e.g., battery storage or pumped hydro) receive priority funding to mitigate intermittency.

3. A significant portion of funding is directed toward infrastructure development, specifically expanding and upgrading transmission lines to connect remote wind farms to the national grid.

4. Public-private partnerships will facilitate knowledge transfer, risk sharing, and accelerated project deployment.)

**📚 Policy Reference:**  
(Include real-world programs or laws that relate to the recommendation. Provide tangible examples, case studies, or successful pilot projects that support your recommendation along with mentions about programs in different countries or states for comparison.)

{formatted_refs}
"""
    try:
        # Generate policy insight using Gemini AI
        response = gemini_model.generate_content(prompt)
        policy_insight = response.text.strip()

        # Post-process the result for user-friendly presentation
        if sentiment.lower() == "negative":
            policy_insight += "\n🔴 A critical perspective on this issue suggests that more focus is needed on overcoming these challenges."
        elif sentiment.lower() == "positive":
            policy_insight += "\n✅ This perspective highlights significant opportunities for growth and innovation in the sector."
        
        return policy_insight

    except Exception as e:
        return f"❌ Unable to generate insight. Error: {str(e)}"


def perform_google_cse_search(query, google_cx, google_api_key):
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "q": query,
        "cx": google_cx,
        "key": google_api_key,
        "num": 5
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "items" in data:
        return [{"snippet": item["snippet"], "link": item["link"]} for item in data["items"]]
    else:
        return []


# Initialize session state for tracking first load
if 'first_load' not in st.session_state:
    st.session_state.first_load = True

# Load models with spinner only on first load
if st.session_state.first_load:
    with st.spinner("Loading models and data (this will only happen once)..."):
        tokenizer, model, generator = load_models()
        df = load_data()
    st.session_state.first_load = False
else:
    # Use cached functions without spinner on subsequent loads
    tokenizer, model, generator = load_models()
    df = load_data()

# --- UI LAYOUT ---
st.title("⚡ Energy Policy Sentiment Analysis Dashboard")
st.markdown("""
This dashboard analyzes sentiment in energy policy discussions across different platforms.
Enter keywords, view sentiment distributions, and get AI-powered insights on your comments.
""")

# Create tabs for better organization - ADDED A FOURTH TAB FOR YEARLY ANALYSIS
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 Keyword Analysis", "💬 Comment Analysis", "📅 Yearly Trends"])

with tab1:
    st.header("Overview of Sentiment Distribution")
    
    # Add data summary
   # st.metric("Total Records", f"{len(df):,}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Visualization: Sentiment Overall
        sentiment_counts = df['sentiment_category'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig_overall = px.pie(
            sentiment_counts, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'blue'},
            title="Overall Sentiment Distribution"
        )
        st.plotly_chart(fig_overall)
    
    with col2:
        # Visualization: Sentiment by Platform
        platform_sentiment = df.groupby(['platform', 'sentiment_category']).size().unstack(fill_value=0).reset_index()
        platform_sentiment_melted = pd.melt(
            platform_sentiment, 
            id_vars=['platform'],
            value_vars=['Positive', 'Neutral', 'Negative'],
            var_name='Sentiment',
            value_name='Count'
        )
        fig_platform = px.bar(
            platform_sentiment_melted,
            x='platform',
            y='Count',
            color='Sentiment',
            barmode='stack',
            color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'blue'},
            title="Sentiment Distribution by Platform"
        )
        st.plotly_chart(fig_platform)

with tab2:
    st.header("🔍 Keyword-Specific Analysis")
    
    # Initialize session state for persistent values if they don't exist yet
    if 'selected_buttons' not in st.session_state:
        st.session_state.selected_buttons = []
    
    if 'keyword_input_value' not in st.session_state:
        st.session_state.keyword_input_value = ""

    # Show top 10 keywords as clickable buttons
    top_keywords = df['keyword'].value_counts().head(10).index.tolist()
    st.write("Quick select popular keywords:")
    cols = st.columns(5)
    
    # Track buttons clicked in this session
    current_session_buttons = []

    for i, keyword in enumerate(top_keywords):
        col_index = i % 5
        button_key = f"btn_{keyword}"
        if cols[col_index].button(keyword, key=button_key):
            current_session_buttons.append(keyword)
    
    # Update session state with any new buttons clicked
    if current_session_buttons:
        st.session_state.selected_buttons = current_session_buttons
    
    # Define a callback function for the text input
    def update_keyword_input():
        st.session_state.keyword_input_value = st.session_state.keyword_input
    
    # Text input for custom keywords, using the callback
    keywordi = st.text_input("Or enter a keyword to analyze (e.g., 'solar power'):", 
                           value=st.session_state.keyword_input_value,
                           key="keyword_input",
                           on_change=update_keyword_input)

    # Combine selected keywords using session state
    selected_keywords = st.session_state.selected_buttons + [keywordi]
    selected_keywords = [kw.strip().lower() for kw in selected_keywords if kw.strip()]

    if selected_keywords:
        st.subheader(f"📌 Analysis for: {', '.join(selected_keywords)}")

        # Filter the dataset based on selected keywords
        filtered_df = df[df['keyword'].str.lower().isin(selected_keywords)]

        if not filtered_df.empty:
            # Prepare sentiment counts
            sentiment_counts = filtered_df['sentiment_category'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']

            # Prepare sentiment by platform
            sentiment_by_platform = filtered_df.groupby(['platform', 'sentiment_category']).size().unstack(fill_value=0)

            # Create columns for side-by-side charts
            col1, col2 = st.columns(2)

            with col1:
                fig_sentiment = px.pie(
                    sentiment_counts,
                    values='Count',
                    names='Sentiment',
                    color='Sentiment',
                    color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'blue'},
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)

            with col2:
                if not sentiment_by_platform.empty:
                    fig = px.bar(
                        sentiment_by_platform,
                        x=sentiment_by_platform.index,
                        y=sentiment_by_platform.columns,
                        barmode='stack',
                        color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'blue'},
                        labels={'x': 'Platform', 'y': 'Count'},
                        title="Sentiment Distribution by Platform"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Reliability insight
            if not sentiment_by_platform.empty:
                platform_reliability = sentiment_by_platform[['Positive', 'Neutral', 'Negative']].sum(axis=1)
                most_reliable_platform = platform_reliability.idxmax()
                st.markdown(f"✅ **Most reliable platform for policymaking:** `{most_reliable_platform.capitalize()}`")
                st.markdown("This platform has the highest total mentions and positive sentiment. It may provide the most valuable insights for policy decisions.")
        else:
            st.warning("No data found for the selected keywords.")
            
with tab3:
    st.header("Analyze and Get AI Insights from Comments")
    
    # Add user input for comment and keyword
    user_comment = st.text_area("Enter a comment for analysis:", "", height=200)
    keyword_selected = st.text_input("Enter the Keyword", "")
    
    # Add option to include local model insights
    include_local_insights = st.checkbox("Include detailed insights from local model", value=True)
    
    if st.button("Generate Insight", key="generate_insight_button"):
        if user_comment:
            with st.spinner("Analyzing sentiment and generating insights..."):
                # Use the RoBERTa model to determine sentiment automatically
                sentiment_category = classify_sentiment_with_roberta(user_comment)
                
                # Display the detected sentiment
                sentiment_emoji = {"Positive": "😀", "Neutral": "😐", "Negative": "🙁"}
                st.write(f"Detected sentiment: {sentiment_emoji.get(sentiment_category, '')} **{sentiment_category}**")
                
                # Get local model insights if requested
                if include_local_insights:
                    local_insights = get_local_model_insights(user_comment, tokenizer, model)
                    if local_insights:
                        # Create confidence visualization
                        st.write("### Local Model Insights")
                        conf_data = pd.DataFrame({
                            'Sentiment': list(local_insights['confidence_scores'].keys()),
                            'Confidence': list(local_insights['confidence_scores'].values())
                        })
                        
                        # Create a horizontal bar chart for confidence scores
                        fig = px.bar(conf_data, 
                               y='Sentiment', 
                               x='Confidence', 
                               color='Sentiment',
                               color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'blue'},
                               orientation='h',
                               title="Sentiment Confidence Scores")
                        fig.update_layout(xaxis_range=[0, 1])
                        st.plotly_chart(fig)
                        
                        # Display intensity
                        st.metric("Sentiment Intensity", 
                                 f"{abs(local_insights['intensity']):.2f}", 
                                 delta="stronger" if abs(local_insights['intensity']) > 0.5 else "moderate")
                        
                        # Ensure the stopwords dataset is downloaded
                        nltk.download('stopwords')


                        
                        def clean_key_phrases(key_phrases):
    # Remove any unwanted characters (like 'Ġ' and others) from the key phrases
                            cleaned_phrases = [phrase.replace('Ġ', ' ').strip() for phrase in key_phrases]
                            return cleaned_phrases

                          # Define a function to filter out stopwords
                        def remove_stopwords(key_phrases):
                           stop_words = set(stopwords.words('english'))
                           filtered_phrases = [phrase for phrase in key_phrases if phrase.lower() not in stop_words]
                           return filtered_phrases

# Example usage in your app
                        if 'key_phrases' in local_insights:
                          st.write("**Key phrases detected:**")
                          cleaned_phrases = clean_key_phrases(local_insights['key_phrases'])
                          # Remove stopwords after cleaning
                          final_phrases = remove_stopwords(cleaned_phrases)
                          st.write(", ".join(final_phrases))
                          

                
                # Generate insights based on the detected sentiment
                insight = generate_insights_with_gemini(user_comment, sentiment_category, keyword_selected, google_cx, google_api_key)
                
                st.markdown("### Generated Policy Analysis")
                st.markdown(insight)
        else:
            st.warning("Please enter a comment for analysis.")
            
with tab4:
    st.header("📅 Yearly Sentiment Trends and Keyword Analysis")

    # === Part 1: Yearly Sentiment Line Plot ===
    yearly_sentiment = df.groupby(['year', 'sentiment_category']).size().unstack(fill_value=0).reset_index()
    yearly_sentiment_melted = pd.melt(
        yearly_sentiment, 
        id_vars=['year'],
        value_vars=['Positive', 'Neutral', 'Negative'],
        var_name='Sentiment',
        value_name='Count'
    )

    fig_yearly_trends = px.line(
        yearly_sentiment_melted,
        x='year',
        y='Count',
        color='Sentiment',
        markers=True,
        title="Yearly Sentiment Trends",
        color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'blue'}
    )
    st.plotly_chart(fig_yearly_trends, use_container_width=True)

    # === Part 2: Yearly Top Keyword Sentiment Breakdown ===
    st.header("📌 Analyze Top Keywords and Sentiments by Year")

    available_years = sorted(df['year'].dropna().unique())
    selected_years = st.multiselect(
        "Select up to 2 years to analyze:", 
        options=available_years, 
        default=available_years[-1:]
    )

    if len(selected_years) > 2:
        st.warning("⚠️ Please select no more than 2 years for comparison.")
    elif selected_years:
        top_n = st.slider("Select how many top keywords to display per year:", min_value=3, max_value=15, value=5)

        df_filtered_years = df[df['year'].isin(selected_years)]

        keyword_counts = df_filtered_years.groupby(['year', 'keyword']).size().reset_index(name='count')
        top_keywords_per_year = keyword_counts.groupby('year').apply(
            lambda x: x.nlargest(top_n, 'count')
        ).reset_index(drop=True)

        sentiment_filtered = df_filtered_years.merge(
            top_keywords_per_year[['year', 'keyword']], 
            on=['year', 'keyword']
        )

        sentiment_summary = sentiment_filtered.groupby(
            ['year', 'keyword', 'sentiment_category']
        ).size().reset_index(name='count')

        fig_keywords = px.bar(
            sentiment_summary,
            x='keyword', y='count', color='sentiment_category',
            facet_col='year', facet_col_wrap=2,
            category_orders={"sentiment_category": ['Negative', 'Neutral', 'Positive']},
            title="Top Keywords by Year with Sentiment Breakdown",
            labels={'count': 'Mentions', 'keyword': 'Keyword', 'sentiment_category': 'Sentiment'},
            color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'blue'}
        )

        fig_keywords.update_xaxes(tickangle=90)
        fig_keywords.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig_keywords, use_container_width=True)

        # === PART: Sequential Layout ===
        st.header("📊 Sentiment Analysis for Energy Keywords")

        # ==== SENTIMENT OVER TIME FOR A SPECIFIC KEYWORD ====
        st.subheader("📈 Sentiment Over Time for a Specific Keyword")

        unique_keywords = sorted(df['keyword'].dropna().unique())
        keyword_single = st.selectbox("Choose a keyword to explore:", unique_keywords, key="explore_keyword")

        if keyword_single:
            keyword_yearly = df[df['keyword'].str.strip().str.lower() == keyword_single.strip().lower()]
            # st.write("🔎 Exploring keyword:", keyword_single)
            # st.write("🧮 Rows matched:", keyword_yearly.shape[0])
            # st.dataframe(keyword_yearly.head())

            if not keyword_yearly.empty:
                keyword_yearly = keyword_yearly.groupby(['year', 'sentiment_category']).size().unstack(fill_value=0)

                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    if sentiment not in keyword_yearly.columns:
                        keyword_yearly[sentiment] = 0

                keyword_yearly = keyword_yearly.sort_index()

                keyword_yearly_melted = keyword_yearly.reset_index().melt(
                    id_vars='year',
                    value_vars=['Positive', 'Neutral', 'Negative'],
                    var_name='Sentiment',
                    value_name='Count'
                )

                fig_keyword_yearly = px.bar(
                    keyword_yearly_melted,
                    x='year',
                    y='Count',
                    color='Sentiment',
                    barmode='group',
                    title=f"Sentiment Trends for '{keyword_single}' Over Time",
                    color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'blue'}
                )

                st.plotly_chart(fig_keyword_yearly, use_container_width=True)

        # ==== COMPARE WITH OPPOSITE KEYWORD ====
        st.subheader("🔄 Compare with Opposite Keyword")

        @st.cache_data
        def load_opposites():
            df_opposites = pd.read_excel("rencon.xlsx")
            df_opposites.columns = df_opposites.columns.str.strip().str.title()
            return df_opposites[['Keyword', 'Opposite']].dropna()

        opposites_df = load_opposites()
        df['keyword_norm'] = df['keyword'].str.strip().str.lower()
        opposites_df['Keyword_norm'] = opposites_df['Keyword'].str.strip().str.lower()
        opposites_df['Opposite_norm'] = opposites_df['Opposite'].str.strip().str.lower()

        valid_pairs = opposites_df[opposites_df['Keyword_norm'].isin(df['keyword_norm'])]

        if not valid_pairs.empty:
            default_index = valid_pairs['Keyword'].tolist().index(keyword_single) if keyword_single in valid_pairs['Keyword'].tolist() else 0

            selected_keyword = st.selectbox(
                "Select a keyword to compare:",
                sorted(valid_pairs['Keyword'].unique()),
                index=default_index,
                key="compare_keyword"
            )

            opposite_row = valid_pairs[valid_pairs['Keyword'] == selected_keyword]

            if not opposite_row.empty:
                opposite_keyword = opposite_row['Opposite'].values[0]
                st.markdown(f"✅ Comparing **{selected_keyword}** and its opposite **{opposite_keyword}**")

                normalized_keywords = [
                    selected_keyword.strip().lower(),
                    opposite_keyword.strip().lower()
                ]

                paired_df = df[df['keyword_norm'].isin(normalized_keywords)]

                if paired_df.empty:
                    st.warning(f"No data found for '{selected_keyword}' or its opposite '{opposite_keyword}'. Showing only available keyword.")
                    available = [kw for kw in normalized_keywords if kw in df['keyword_norm'].values]
                    if available:
                        paired_df = df[df['keyword_norm'].isin(available)]
                    else:
                        st.stop()

                if not paired_df.empty:
                    pair_trends = paired_df.groupby(['year', 'keyword', 'sentiment_category']).size().reset_index(name='count')

                    fig_pair_compare = px.bar(
                        pair_trends,
                        x='year',
                        y='count',
                        color='sentiment_category',
                        barmode='group',
                        facet_col='keyword',
                        category_orders={"sentiment_category": ['Negative', 'Neutral', 'Positive']},
                        title=f"Sentiment Trends: '{selected_keyword}' vs '{opposite_keyword}'",
                        color_discrete_map={'Positive': 'green', 'Neutral': 'orange', 'Negative': 'blue'}
                    )

                    fig_pair_compare.update_layout(height=600, showlegend=True)
                    st.plotly_chart(fig_pair_compare, use_container_width=True)
                else:
                    st.warning("⚠️ No sentiment data found for the selected or opposite keyword.")
            else:
                st.warning("⚠️ Opposite keyword not found in mapping.")
        else:
            st.info("❗No matching keyword pairs found in your dataset.")
