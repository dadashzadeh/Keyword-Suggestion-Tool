import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time
from streamlit_echarts import st_echarts
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import hazm
from typing import List, Dict
from io import BytesIO

# Configuration
st.set_page_config(
    page_title="google Suggestions",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with RTL support
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 1300px;
    }
    .stButton>button {
        width: 100%;
    }
    .persian-text {
        direction: rtl;
        text-align: right;
        font-family: 'Vazir', 'Iran Sans', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

class PersianTextAnalyzer:
    """Persian text analysis using Hazm."""
    
    def __init__(self):
        self.normalizer = hazm.Normalizer()
        self.stemmer = hazm.Stemmer()
        self.lemmatizer = hazm.Lemmatizer()
        self.stop_words = hazm.stopwords_list()
    
    def normalize_text(self, text: str) -> str:
        return self.normalizer.normalize(str(text))
    
    def get_tokens(self, text: str) -> List[str]:
        return hazm.word_tokenize(self.normalize_text(text))
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [word for word in tokens if word not in self.stop_words]

def get_suggestions(keyword: str, source: str = 'google', language: str = 'english') -> list:
    """Get search suggestions from Google with language support."""
    if source.lower() == 'google':
        url = 'https://suggestqueries.google.com/complete/search'
        params = {
            'client': 'firefox',
            'q': keyword,
            'hl': 'fa' if language == 'persian' else 'en'
        }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        suggestions = response.json()[1]
        return suggestions[:5]  # Limit to top 5 suggestions
    except Exception as e:
        st.error(f"Error fetching suggestions: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def get_suggests_tree(keyword: str, source: str = 'google', max_depth: int = 3, language: str = 'english') -> pd.DataFrame:
    """Get a tree of search suggestions with language support."""
    edges = []
    timestamp = datetime.now()
    
    with st.spinner('Fetching suggestions...'):
        progress_bar = st.progress(0)
        
        # Level 1
        level1_suggests = get_suggestions(keyword, source, language)
        progress_bar.progress(33)
        
        for rank1, edge1 in enumerate(level1_suggests):
            edges.append({
                'root': keyword,
                'edge': edge1,
                'rank': rank1,
                'depth': 1,
                'search_engine': source,
                'datetime': timestamp,
                'level1': edge1,
                'level2': None,
                'level3': None
            })
            
            if max_depth >= 2:
                time.sleep(0.2)
                level2_suggests = get_suggestions(edge1, source, language)
                progress_bar.progress(66)
                
                for rank2, edge2 in enumerate(level2_suggests):
                    edges.append({
                        'root': keyword,
                        'edge': edge2,
                        'rank': rank2,
                        'depth': 2,
                        'search_engine': source,
                        'datetime': timestamp,
                        'level1': edge1,
                        'level2': edge2,
                        'level3': None
                    })
                    
                    if max_depth >= 3:
                        time.sleep(0.2)
                        level3_suggests = get_suggestions(edge2, source, language)
                        
                        for rank3, edge3 in enumerate(level3_suggests):
                            edges.append({
                                'root': keyword,
                                'edge': edge3,
                                'rank': rank3,
                                'depth': 3,
                                'search_engine': source,
                                'datetime': timestamp,
                                'level1': edge1,
                                'level2': edge2,
                                'level3': edge3
                            })
        
        progress_bar.progress(100)
        time.sleep(0.1)
        progress_bar.empty()
    
    return pd.DataFrame(edges)

def analyze_suggestions(df: pd.DataFrame, language: str = 'english') -> pd.DataFrame:
    """Analyze suggestions with language support."""
    enhanced_df = df.copy()
    
    if language == 'persian':
        analyzer = PersianTextAnalyzer()
        tokenize_func = analyzer.get_tokens
        normalize_func = analyzer.normalize_text
        stop_words = analyzer.stop_words
    else:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        tokenize_func = word_tokenize
        normalize_func = str.lower
        stop_words = set(stopwords.words('english'))
    
    # Text analysis
    suggestions_text = enhanced_df['edge'].fillna('').apply(normalize_func).tolist()
    
    # Word count
    enhanced_df['word_count'] = enhanced_df['edge'].fillna('').apply(
        lambda x: len(tokenize_func(normalize_func(x)))
    )
    
    # Character count
    enhanced_df['char_count'] = enhanced_df['edge'].fillna('').apply(len)
    
    # Word repetitions
    def count_word_repetitions(text):
        tokens = tokenize_func(normalize_func(text))
        return sum(count - 1 for count in Counter(tokens).values())
    
    enhanced_df['word_repetitions'] = enhanced_df['edge'].apply(count_word_repetitions)
    
    # TF-IDF
    tfidf = TfidfVectorizer(tokenizer=tokenize_func, preprocessor=normalize_func)
    tfidf_matrix = tfidf.fit_transform(suggestions_text)
    enhanced_df['avg_tfidf'] = np.array(tfidf_matrix.mean(axis=1)).flatten()
    
    # Keyword density
    def calculate_keyword_density(text, keyword):
        text_norm = normalize_func(text)
        keyword_norm = normalize_func(keyword)
        tokens = tokenize_func(text_norm)
        keyword_tokens = tokenize_func(keyword_norm)
        keyword_count = sum(1 for token in tokens if token in keyword_tokens)
        return (keyword_count / len(tokens) * 100) if tokens else 0
    
    enhanced_df['keyword_density'] = enhanced_df['edge'].apply(
        lambda x: calculate_keyword_density(x, enhanced_df['root'].iloc[0])
    )
    
    # Stop words ratio
    def calculate_stop_words_ratio(text):
        tokens = tokenize_func(normalize_func(text))
        stop_count = sum(1 for token in tokens if token in stop_words)
        return stop_count / len(tokens) if tokens else 0
    
    enhanced_df['stop_words_ratio'] = enhanced_df['edge'].apply(calculate_stop_words_ratio)
    
    # Unique words ratio
    def unique_words_ratio(text):
        tokens = tokenize_func(normalize_func(text))
        return len(set(tokens)) / len(tokens) if tokens else 0
    
    enhanced_df['unique_words_ratio'] = enhanced_df['edge'].apply(unique_words_ratio)
    
    # Complexity score
    enhanced_df['complexity_score'] = (
        enhanced_df['word_count'] * 0.25 +
        enhanced_df['unique_words_ratio'] * 0.25 +
        enhanced_df['avg_tfidf'] * 0.25 +
        (1 - enhanced_df['stop_words_ratio']) * 0.25
    ).round(2)
    
    # Round floating point columns
    float_columns = ['avg_tfidf', 'keyword_density', 'stop_words_ratio', 'unique_words_ratio']
    enhanced_df[float_columns] = enhanced_df[float_columns].round(3)
    
    return enhanced_df

def create_download_links(df: pd.DataFrame, language: str = 'english'):
    """Create download links with language support."""
    # Excel download
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    b64_excel = base64.b64encode(excel_data).decode()
    
    # CSV download
    csv_data = df.to_csv(index=False)
    b64_csv = base64.b64encode(csv_data.encode()).decode()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="suggestions.xlsx">{"دانلود Excel" if language == "persian" else "Download Excel"} 📥</a>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<a href="data:text/csv;base64,{b64_csv}" download="suggestions.csv">{"دانلود CSV" if language == "persian" else "Download CSV"} 📥</a>',
            unsafe_allow_html=True
        )

def main():
    # Header
    col1, _, col3 = st.columns(3)
    with col1:
        st.title("google Suggestions 🔍")
    with col3:
        st.write("")
        st.write("")
        st.markdown(
            '###### Made with :heart: by [@dadashzadeh](https://t.me/Dadashzadeh)'
        )
    
    # Language selection
    language = st.selectbox(
        'Select Language / انتخاب زبان',
        ('english', 'persian')
    )
    
    # Info sections
    with st.expander("ℹ️ About this app", expanded=True):
        if language == 'persian':
            st.write("""
            - به شما کمک می‌کند پیشنهادات کلیدواژه را کشف و تجسم کنید!
            - از موتور جستجوی گوگل پشتیبانی می‌کند
            - تا ۳ سطح عمق را پشتیبانی می‌کند
            - مناسب برای تحقیق محتوا و بهینه‌سازی SEO
            """)
        else:
            st.write("""
            - helps you discover and visualize keyword suggestions!
            - Supports Google search engine
            - Supports up to 3 levels of suggestions
            - Perfect for content research and SEO optimization
            """)
    
    # User Input
    keyword = st.text_input(
        'کلیدواژه را وارد کنید' if language == 'persian' else 'Enter your keyword'
    )
    
    col1, col2 = st.columns(2)
    with col1:
        search_engine = 'google'
        st.info("Google Search" if language == 'english' else "جستجوی گوگل")
    
    with col2:
        max_depth = st.number_input(
            'عمق درخت (۱-۳ سطح)' if language == 'persian' else 'Tree Depth (1-3 levels)',
            min_value=1,
            max_value=3,
            value=2
        )
    
    st.info(
        "سطوح بالاتر زمان بیشتری نیاز دارند" if language == 'persian' 
        else "Higher depth levels will take longer to process"
    )
    
    fetch_button = st.button(
        '🔥 دریافت پیشنهادات' if language == 'persian' else 'Fetch Suggestions! 🔥'
    )
    
    if fetch_button and keyword:
        try:
            # Fetch suggestions
            suggestions_df = get_suggests_tree(keyword, search_engine, max_depth, language)
            
            if not suggestions_df.empty:
                # Analyze suggestions
                enhanced_df = analyze_suggestions(suggestions_df, language)
                
                # Build tree data
                tree_data = {
                    "name": keyword,
                    "children": [
                        {
                            "name": group['level1'],
                            "children": [
                                {
                                    "name": row['level2'],
                                    "children": [
                                        {"name": row2['level3']}
                                        for _, row2 in enhanced_df[
                                            (enhanced_df['level1'] == group['level1']) & 
                                            (enhanced_df['level2'] == row['level2']) & 
                                            (enhanced_df['depth'] == 3)
                                        ].iterrows()
                                    ]
                                }
                                for _, row in enhanced_df[
                                    (enhanced_df['level1'] == group['level1']) & 
                                    (enhanced_df['depth'] == 2)
                                ].iterrows()
                            ]
                        }
                        for _, group in enhanced_df[enhanced_df['depth'] == 1].iterrows()
                    ]
                }
                
                # Tree visualization
                tree_config = {
                    "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
                    "series": [{
                        "type": "tree",
                        "data": [tree_data],
                        "top": "1%",
                        "left": "7%",
                        "bottom": "1%",
                        "right": "20%",
                        "symbolSize": 7,
                        "initialTreeDepth": max_depth,
                        "label": {
                            "position": "left",
                            "verticalAlign": "middle",
                            "align": "right",
                            "fontSize": 12,
                        },
                        "leaves": {
                            "label": {
                                "position": "right",
                                "verticalAlign": "middle",
                                "align": "left",
                            }
                        },
                        "expandAndCollapse": True,
                        "animationDuration": 550,
                        "animationDurationUpdate": 750,
                        "layout": "orthogonal",
                        "orient": "LR"
                    }]
                }
                
                st.success(
                    '✅ پیشنهادات با موفقیت دریافت شدند!' if language == 'persian'
                    else '✅ Suggestions retrieved successfully!'
                )
                # Display results
                st.markdown(
                    '### 🌳 نمای درختی تعاملی' if language == 'persian'
                    else '### 🌳 Interactive Tree View'
                )
                st.markdown(
                    '*برای ذخیره تصویر کلیک راست کنید* 📸' if language == 'persian'
                    else '*Right-click to save as image* 📸'
                )
                st_echarts(options=tree_config, height="800px")
                
                # Display enhanced DataFrame
                st.markdown(
                    '### 📊 نمای جدول تحلیلی' if language == 'persian'
                    else '### 📊 Analysis Table'
                )
                
                # Define column configuration based on language
                if language == 'persian':
                    column_config = {
                        'word_count': st.column_config.NumberColumn(
                            'تعداد کلمات',
                            help='تعداد کلمات در هر پیشنهاد'
                        ),
                        'char_count': st.column_config.NumberColumn(
                            'تعداد حروف',
                            help='تعداد حروف در هر پیشنهاد'
                        ),
                        'word_repetitions': st.column_config.NumberColumn(
                            'تکرار کلمات',
                            help='تعداد کلمات تکراری'
                        ),
                        'avg_tfidf': st.column_config.NumberColumn(
                            'نمره TF-IDF',
                            help='معیار اهمیت کلمات',
                            format="%.3f"
                        ),
                        'keyword_density': st.column_config.NumberColumn(
                            'تراکم کلیدواژه',
                            help='درصد تکرار کلیدواژه اصلی',
                            format="%.2f%%"
                        ),
                        'stop_words_ratio': st.column_config.NumberColumn(
                            'نسبت کلمات عمومی',
                            help='نسبت کلمات پرکاربرد به کل',
                            format="%.2f"
                        ),
                        'unique_words_ratio': st.column_config.NumberColumn(
                            'نسبت کلمات یکتا',
                            help='نسبت کلمات غیرتکراری به کل',
                            format="%.2f"
                        ),
                        'complexity_score': st.column_config.NumberColumn(
                            'نمره پیچیدگی',
                            help='معیار پیچیدگی متن (0-1)',
                            format="%.2f"
                        )
                    }
                else:
                    column_config = {
                        'word_count': st.column_config.NumberColumn(
                            'Word Count',
                            help='Number of words in suggestion'
                        ),
                        'char_count': st.column_config.NumberColumn(
                            'Char Count',
                            help='Number of characters'
                        ),
                        'word_repetitions': st.column_config.NumberColumn(
                            'Word Repetitions',
                            help='Number of repeated words'
                        ),
                        'avg_tfidf': st.column_config.NumberColumn(
                            'TF-IDF Score',
                            help='Term Frequency-Inverse Document Frequency score',
                            format="%.3f"
                        ),
                        'keyword_density': st.column_config.NumberColumn(
                            'Keyword Density',
                            help='Percentage of keyword occurrence',
                            format="%.2f%%"
                        ),
                        'stop_words_ratio': st.column_config.NumberColumn(
                            'Stop Words Ratio',
                            help='Ratio of common words',
                            format="%.2f"
                        ),
                        'unique_words_ratio': st.column_config.NumberColumn(
                            'Unique Words Ratio',
                            help='Ratio of unique words to total words',
                            format="%.2f"
                        ),
                        'complexity_score': st.column_config.NumberColumn(
                            'Complexity Score',
                            help='Custom complexity metric (0-1)',
                            format="%.2f"
                        )
                    }
                
                # Create tabs for different views
                tab1, tab2 = st.tabs(
                    ["نمای ساده", "تحلیل پیشرفته"] if language == 'persian'
                    else ["Basic View", "Advanced Analysis"]
                )
                
                with tab1:
                    basic_columns = ['root', 'edge', 'depth', 'word_count', 
                                   'keyword_density', 'complexity_score']
                    st.dataframe(
                        enhanced_df[basic_columns],
                        column_config=column_config,
                        hide_index=True
                    )
                
                with tab2:
                    st.dataframe(
                        enhanced_df,
                        column_config=column_config,
                        hide_index=True
                    )
                
                # Add download section
                st.markdown(
                    '### 📥 دانلود نتایج' if language == 'persian'
                    else '### 📥 Download Results'
                )
                create_download_links(enhanced_df, language)
                
                # Add metrics summary
                st.markdown(
                    '### 📊 خلاصه تحلیل' if language == 'persian'
                    else '### 📊 Analysis Summary'
                )
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "میانگین کلمات" if language == 'persian' else "Average Words",
                        f"{enhanced_df['word_count'].mean():.1f}",
                        f"{enhanced_df['word_count'].std():.1f} σ"
                    )
                
                with col2:
                    st.metric(
                        "میانگین تراکم کلیدواژه" if language == 'persian' else "Avg Keyword Density",
                        f"{enhanced_df['keyword_density'].mean():.1f}%",
                        f"{enhanced_df['keyword_density'].std():.1f}% σ"
                    )
                
                with col3:
                    st.metric(
                        "میانگین پیچیدگی" if language == 'persian' else "Avg Complexity",
                        f"{enhanced_df['complexity_score'].mean():.2f}",
                        f"{enhanced_df['complexity_score'].std():.2f} σ"
                    )
                
                with col4:
                    st.metric(
                        "نسبت کلمات یکتا" if language == 'persian' else "Unique Words Ratio",
                        f"{enhanced_df['unique_words_ratio'].mean():.2f}",
                        f"{enhanced_df['unique_words_ratio'].std():.2f} σ"
                    )
                
            else:
                st.warning(
                    'پیشنهادی یافت نشد. کلیدواژه دیگری را امتحان کنید.' if language == 'persian'
                    else 'No suggestions found. Try a different keyword.'
                )
                
        except Exception as e:
            st.error(
                f'خطایی رخ داد: {str(e)}' if language == 'persian'
                else f'An error occurred: {str(e)}'
            )
            
    elif fetch_button:
        st.warning(
            'لطفاً یک کلیدواژه وارد کنید.' if language == 'persian'
            else 'Please enter a keyword first.'
        )

if __name__ == "__main__":
    main()
