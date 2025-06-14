# Hey everyone :> Hope you're doing great!!



# 🔍 Sentiment Analysis & Loan Approval Based on Credit Score Using Machine Learning

***

## 📌 Project Overview
This end-to-end solution combines **credit risk assessment** with **customer sentiment analysis** to automate loan approval decisions. The system evaluates applicants in two sequential stages:
1. **Creditworthiness Check**: Filters applicants based on credit score threshold
2. **Sentiment Evaluation**: Analyzes customer reviews using NLP
3. **Final Decision**: Approves loans only when both conditions are satisfied

The interactive dashboard provides visual analytics for both credit profiles and sentiment patterns.

***

## ✨ Key Features
### 📈 Dual-Stage Approval System
- **Stage 1**: Credit score screening (threshold-based)
- **Stage 2**: Sentiment analysis of product reviews
- **Approval Criteria**: 
  - Credit Score > 650 
  - Positive Sentiment > 50%

### 📊 Interactive Dashboard
- Credit score distributions and statistics
- Sentiment analysis visualizations:
  - Compound sentiment heatmap
  - Positive/negative sentiment bar charts
  - Sentiment trend timelines
- Real-time approval/rejection indicators

### 🤖 ML Components
- Rule-based credit scoring
- VADER (Valence Aware Dictionary and sEntiment Reasoner) NLP model
- Automated decision pipeline

***

### How to Run
-Open cmd
-change directory to your project location(cd 'path')
-type streamlit run senti_app.py
-feed the data set in the dashboard


***

## ⚙️ Technical Requirements
```text
Python 3.8+
Libraries:
- pandas | numpy | scikit-learn
- nltk (with vader_lexicon)
- matplotlib | seaborn | plotly
- streamlit (dashboard)

###This project is free to use and edit :)
