import os
import time
import googleapiclient.discovery
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import google.generativeai as genai
from dotenv import load_dotenv
import re
import unicodedata
import json
import random

# Create output directory
output_dir = "analysis_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not GEMINI_API_KEY or not YOUTUBE_API_KEY:
    raise ValueError("❌ API Keys are missing!")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-001")

# Cache for API responses
cache_dir = os.path.join(output_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)

def get_cache_path(video_id, cache_type):
    """Get path for cached data"""
    return os.path.join(cache_dir, f"{video_id}_{cache_type}.json")

def load_from_cache(video_id, cache_type):
    """Load data from cache if available"""
    cache_path = get_cache_path(video_id, cache_type)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None

def save_to_cache(video_id, cache_type, data):
    """Save data to cache"""
    cache_path = get_cache_path(video_id, cache_type)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False

def sanitize_filename(filename):
    """Convert a string into a valid filename by removing or replacing invalid characters."""
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
    
    # Replace invalid characters with underscores
    invalid_chars = r'[<>:"/\\|?*]'
    filename = re.sub(invalid_chars, '_', filename)
    
    # Limit length to avoid path too long errors
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename

def extract_video_id(url):
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/shorts\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_video_title(video_id):
    """Fetch video title using YouTube API."""
    # Check cache first
    cached_data = load_from_cache(video_id, "title")
    if cached_data:
        return cached_data
    
    try:
        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()
        if response['items']:
            title = response['items'][0]['snippet']['title']
            # Save to cache
            save_to_cache(video_id, "title", title)
            return title
        return f"Video {video_id}"
    except Exception as e:
        print(f"⚠️ Error fetching video title: {e}")
        return f"Video {video_id}"

def fetch_youtube_comments(video_id, max_results=100):
    """Fetch comments from YouTube with caching"""
    # Check cache first
    cached_comments = load_from_cache(video_id, "comments")
    if cached_comments:
        print(f"📦 Using cached comments for: {video_id}")
        return cached_comments
    
    print(f"🌐 Fetching comments from YouTube API for: {video_id}")
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments, nextPageToken = [], None

    while len(comments) < max_results:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_results - len(comments)),
            pageToken=nextPageToken
        )
        response = request.execute()
        comments.extend(response.get("items", []))
        nextPageToken = response.get("nextPageToken", None)
        if not nextPageToken:
            break
        # Add a small delay to avoid rate limiting
        time.sleep(0.5)
    
    # Save to cache
    save_to_cache(video_id, "comments", comments)
    return comments

def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

def classify_comments(df):
    df['sentiment_score'] = df['text'].apply(analyze_sentiment)
    df['label'] = df['sentiment_score'].apply(lambda x: 1 if x > 0 else 0 if x < 0 else 2)

    return pd.DataFrame({
        'Good Comments': df[df['label'] == 1]['text'].reset_index(drop=True),
        'Bad Comments': df[df['label'] == 0]['text'].reset_index(drop=True),
        'Neutral Comments': df[df['label'] == 2]['text'].reset_index(drop=True)
    })

def train_classifiers(df):
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    lr_model = LogisticRegression()
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)

    return y_test, y_pred_lr, y_pred_nb, lr_model, nb_model, vectorizer

def evaluate_model(y_test, y_pred, name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"\n📊 {name} Model")
    print(f"🔹 Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
    return f"{name} - Acc: {acc:.2f}, Prec: {prec:.2f}, Rec: {rec:.2f}, F1: {f1:.2f}"

def generate_suggestions(df, video_title):
    """Generate suggestions with rate limiting and caching"""
    suggestions, bad_comments = [], df['Bad Comments'].dropna().head(5)  # Reduced from 10 to 5
    
    # Check if we have cached suggestions
    cache_key = f"suggestions_{sanitize_filename(video_title)}"
    cached_suggestions = load_from_cache(cache_key, "suggestions")
    
    if cached_suggestions:
        print(f"📦 Using cached suggestions for: {video_title}")
        current_suggestions = cached_suggestions
    else:
        print(f"🤖 Generating suggestions with Gemini AI for: {video_title}")
        current_suggestions = []
        
        # Add a delay between API calls
        for i, comment in enumerate(bad_comments):
            prompt = f"Rewrite this comment in a polite way without changing its meaning: {comment}"
            
            # Add random delay between 1-3 seconds
            if i > 0:
                delay = random.uniform(1, 3)
                print(f"⏱️ Waiting {delay:.1f} seconds before next API call...")
                time.sleep(delay)
            
            for attempt in range(3):
                try:
                    response = model.generate_content(prompt)
                    current_suggestions.append({
                        'Video Title': video_title,
                        'Bad Comments': comment,
                        'Suggested Comments': response.text
                    })
                    break
                except Exception as e:
                    print(f"⚠️ Gemini Error (Attempt {attempt+1}): {e}")
                    time.sleep(2)
            else:
                current_suggestions.append({
                    'Video Title': video_title,
                    'Bad Comments': comment,
                    'Suggested Comments': "⚠️ Gemini failed"
                })
        
        # Save to cache
        save_to_cache(cache_key, "suggestions", current_suggestions)
    
    # Convert current suggestions to DataFrame
    current_df = pd.DataFrame(current_suggestions)
    
    # Save to Excel with timestamp to avoid permission issues
    timestamp = int(time.time())
    excel_path = os.path.join(output_dir, f"suggested_comments_{timestamp}.xlsx")
    current_df.to_excel(excel_path, sheet_name='Suggestions', index=False)
    print(f"✅ Suggestions saved to: {excel_path}")

def visualize_sentiment_distribution(df, video_title):
    labels = ['Negative', 'Positive', 'Neutral']
    counts = df['label'].value_counts().sort_index()
    plt.figure(figsize=(5,5))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f"Sentiment Distribution - {video_title}")
    
    # Sanitize the filename
    safe_title = sanitize_filename(video_title)
    file = os.path.join(output_dir, f"{safe_title}_pie.png")
    
    plt.savefig(file)
    plt.close()  # Close the figure to free memory
    return file

def generate_pdf_report(results):
    pdf_path = os.path.join(output_dir, "final_output.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    content = []

    for result in results:
        content.append(Paragraph(f"🎬 {result['title']}", styles['Title']))
        content.append(Paragraph("Model Performance:", styles['Heading2']))
        content.append(Paragraph(result['metrics'], styles['Normal']))
        content.append(Spacer(1, 10))

        # Add sentiment distribution pie chart
        content.append(Image(result['pie_chart'], width=400, height=250))
        content.append(Spacer(1, 12))

    doc.build(content)
    print(f"📄 Report saved at: {pdf_path}")

if __name__ == "__main__":
    print("🎥 YouTube Comment Analysis Tool")
    print("Enter YouTube video URLs (one per line). Press Enter twice when done.")
    print("Supported formats:")
    print("- Regular videos: https://www.youtube.com/watch?v=VIDEO_ID")
    print("- Short videos: https://youtube.com/shorts/VIDEO_ID")
    print("- Shortened URLs: https://youtu.be/VIDEO_ID")
    print("\nEnter URLs:")

    videos = []
    while True:
        url = input().strip()
        if not url:
            break
            
        video_id = extract_video_id(url)
        if video_id:
            title = get_video_title(video_id)
            videos.append({"video_id": video_id, "title": title})
            print(f"✅ Added: {title}")
        else:
            print("❌ Invalid YouTube URL. Please try again.")

    if not videos:
        print("❌ No valid videos provided. Exiting...")
        exit()

    print(f"\n📊 Analyzing {len(videos)} videos...")
    final_report = []

    for video in videos:
        print(f"\n📥 Analyzing: {video['title']}")
        comments = fetch_youtube_comments(video['video_id'])
        raw_df = pd.DataFrame([{'text': item['snippet']['topLevelComment']['snippet']['textDisplay']} for item in comments])

        classified_df = classify_comments(raw_df)
        y_test, y_pred_lr, y_pred_nb, *_ = train_classifiers(raw_df)

        metrics_lr = evaluate_model(y_test, y_pred_lr, "Logistic Regression")
        metrics_nb = evaluate_model(y_test, y_pred_nb, "Naive Bayes")

        generate_suggestions(classified_df, video['title'])

        pie_path = visualize_sentiment_distribution(raw_df, video['title'])
        
        final_report.append({
            'title': video['title'],
            'metrics': f"{metrics_lr}\n{metrics_nb}",
            'pie_chart': pie_path
        })

    generate_pdf_report(final_report)
