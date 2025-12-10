RedditSense: Emotion and Sentiment Analysis Dashboard for Reddit Communities
============================================================================

RedditSense is an end-to-end system for acquiring, classifying, visualizing, 
and interpreting emotional dynamics in Reddit communities. It transforms 
unstructured Reddit posts into structured emotion insights through 
transformer-based NLP models, statistical event detection, and interactive dashboards.

This repository contains the full system, including Reddit data acquisition, 
preprocessing, emotion and sentiment classification, post-processing heuristics, 
event detection algorithms, and the Streamlit-based user interface.


1. Features
-----------

1. Data Acquisition
- Fetches posts from multiple subreddits using PRAW.
- Handles missing fields, inconsistent timestamp formats, and rate limits.
- Normalizes timestamps using a universal timestamp parser.
- Outputs structured data in Pandas DataFrame format.

2. Text Preprocessing
- URL removal and Markdown stripping
- Whitespace normalization
- Title-body concatenation
- Length trimming for model stability
- Timestamp conversion

3. Transformer-Based Classification
Two parallel pipelines are used:
- Emotion Classification: CardiffNLP RoBERTa emotion model (8 categories)
- Sentiment Classification: DistilBERT SST-2 model (positive/negative polarity)

4. Post-Processing and Confidence Scoring
- Corrects emotion–sentiment contradictions
- Keyword-based heuristics for negative/business contexts
- Confidence thresholds to flag uncertain predictions
- Automatic "needs review" tagging

5. Event Detection Engine
Includes three anomaly detectors:
- Sentiment Spike Detection (z-score)
- Activity Spike Detection (post-count anomalies)
- Emotion Shift Detection (KL divergence)

6. Interactive Visualization (Streamlit)
Dashboard includes:
- Emotion distribution charts
- Subreddit comparison tools
- Sentiment timelines with rolling averages
- Event markers on plots
- Emotion entropy visualization
- Word cloud generation
- Review-needed post identification
- Event Dashboard summarizing all anomalies


2. System Architecture
----------------------

Reddit API (PRAW)
      ↓
Data Acquisition and Cleaning
      ↓
Text Preprocessing
      ↓
Emotion Model (RoBERTa)      Sentiment Model (DistilBERT)
      ↓                               ↓
        Merged Predictions + Rule-Based Post-Processing
                           ↓
                Structured DataFrame Output
                           ↓
                 Event Detection Engine
                           ↓
              Streamlit Visualization Dashboard


3. Installation
---------------

1. Clone the repository:
   git clone https://github.com/khairulakmal/gradproject.git
   cd gradproject

2. Create a virtual environment:
   python -m venv .venv
   source .venv/bin/activate        (Linux/Mac)
   .venv\Scripts\activate           (Windows)

3. Install dependencies:
   pip install -r requirements.txt

4. Add Reddit API credentials:
   Create a .env file:
   REDDIT_CLIENT_ID=your_id
   REDDIT_CLIENT_SECRET=your_secret
   REDDIT_USER_AGENT=your_agent_name


4. Running the Application
--------------------------

Run the Streamlit dashboard:

   streamlit run classifier_app.py

The dashboard will allow:
- Selecting multiple subreddits
- Running emotion and sentiment analysis
- Viewing timelines, heatmaps, entropy trends
- Inspecting detected events
- Exporting review-needed posts


5. Project Structure
--------------------

.
├── reddit_fetcher.py          # Reddit data acquisition module
├── classifier_app.py          # Main Streamlit application
├── models/                    # Optional model storage
├── requirements.txt           # Python dependencies
├── README.txt                 # Project documentation


6. Use Cases
------------

- Tracking emotional shifts in Reddit communities
- Detecting abnormal discussion patterns
- Comparing sentiment across topics or subreddits
- Research on online public opinion or digital sociology
- Visual analytics for social-media interpretation


7. Limitations
--------------

- Dependent on Reddit API rate limits
- Transformer models are resource-intensive for large datasets
- Event detection performs statistical anomaly detection, not causal inference
- Results reflect linguistic sentiment, not actual intent or truthfulness


8. Planned Extensions
---------------------

- Topic modeling to explain event causes
- Keyword spike detection
- Correlation with external news or financial data
- Real-time monitoring and alerts
- Multi-language emotion classification
your.email@example.com

