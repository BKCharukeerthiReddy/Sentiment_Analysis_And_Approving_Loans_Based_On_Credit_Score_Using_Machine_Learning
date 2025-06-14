# Disable file watcher before any Streamlit imports
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

plt.style.use('ggplot')

# Download NLTK resources
nltk.download(['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 
              'words', 'vader_lexicon'])

def process_reviews_chunk(chunk):
    """Process reviews in chunks for memory efficiency"""
    sia = SentimentIntensityAnalyzer()
    chunk_results = []
    for _, row in chunk.iterrows():
        try:
            scores = sia.polarity_scores(row['Text'])
            chunk_results.append({
                'compound': scores['compound'],
                'pos': scores['pos'],
                'neu': scores['neu'],
                'neg': scores['neg'],
                'Text': row['Text'],
                'Score': row['Score']
            })
        except Exception as e:
            st.warning(f"Error processing review: {str(e)}")
    return pd.DataFrame(chunk_results)

def main():
    st.set_page_config(
        page_title="Loan Approval System",
        page_icon="🏦",
        layout="wide"
    )
    
    # Initialize session state
    if 'phase' not in st.session_state:
        st.session_state.phase = 1
        st.session_state.selected_candidate = None
        st.session_state.credit_results = None

    st.title("🏦 SME Loan Approval System")

    # Phase 1: Credit Score Analysis (UNCHANGED)
    if st.session_state.phase == 1:
        st.header("1️⃣ Credit Worthiness Analysis")
        try:
            # Load credit scoring models
            try:
                scaler = joblib.load("C:\\Users\\Charu\\Desktop\\Credit Score\\f2_Normalisation_CreditScoring.pkl")
                model = joblib.load("C:\\Users\\Charu\\Desktop\\Credit Score\\f1_Classifier_CreditScoring.pkl")
            except Exception as e:
                st.error(f"❌ Model loading failed: {str(e)}")
                st.stop()

            # File upload section
            uploaded_file = st.file_uploader(
                "Upload Financial Data (Excel)",
                type=["xlsx"],
                help="Should contain 28 financial features including income, debt ratio, and credit history"
            )

            if uploaded_file:
                with st.spinner("Analyzing financial viability..."):
                    try:
                        # Process financial data
                        df = pd.read_excel(uploaded_file)
                        
                        # Preserve IDs
                        ids = df['ID'] if 'ID' in df.columns else [f"APP-{i+1000}" for i in range(len(df))]
                        if 'ID' in df.columns: df = df.drop('ID', axis=1)
                        if 'TARGET' in df.columns: df = df.drop('TARGET', axis=1)
                        
                        df = df.fillna(df.mean())
                        
                        if df.shape[1] != 28:
                            st.error(f"Invalid data format: Expected 28 features, got {df.shape[1]}")
                            st.stop()
                        
                        # Make predictions
                        scaled_data = scaler.transform(df.values)
                        probabilities = model.predict_proba(scaled_data)
                        
                        # Store results
                        st.session_state.credit_results = pd.DataFrame({
                            'ID': ids,
                            'Repayment Probability': probabilities[:, 0],
                            'Default Risk': probabilities[:, 1],
                            'Recommendation': ['Approve' if x == 0 else 'Review' for x in model.predict(scaled_data)]
                        }).sort_values('Repayment Probability', ascending=False)
                        
                    except Exception as e:
                        st.error(f"🚨 Financial analysis error: {str(e)}")
                        st.stop()

                # Display financial results
                st.success(f"Analyzed {len(st.session_state.credit_results)} applicants")
                
                # Filter controls
                col1, col2 = st.columns(2)
                with col1:
                    min_prob = st.slider("Minimum repayment probability (%)", 50, 100, 75) / 100
                with col2:
                    top_n = st.slider("Number to display", 5, 100, 20)

                # Filter and display
                filtered = st.session_state.credit_results[
                    (st.session_state.credit_results['Repayment Probability'] >= min_prob) &
                    (st.session_state.credit_results['Recommendation'] == 'Approve')
                ].head(top_n)

                # Create selection box
                st.session_state.selected_candidate = st.selectbox(
                    "Select candidate for business analysis",
                    filtered['ID'],
                    help="Select top candidate to proceed with business sentiment analysis"
                )

                # Display financial metrics
                st.subheader("Financial Health Overview")
                fin_cols = st.columns(3)
                candidate_data = filtered[filtered['ID'] == st.session_state.selected_candidate].iloc[0]
                
                fin_cols[0].metric("Repayment Probability", 
                                 f"{candidate_data['Repayment Probability']:.2%}")
                fin_cols[1].metric("Default Risk", 
                                 f"{candidate_data['Default Risk']:.2%}")
                fin_cols[2].metric("Credit Recommendation", 
                                 candidate_data['Recommendation'],
                                 delta_color="off")

                # Show detailed financial data
                with st.expander("View Full Financial Analysis"):
                    st.dataframe(
                        filtered.style.format({
                            'Repayment Probability': "{:.2%}",
                            'Default Risk': "{:.2%}"
                        }),
                        height=400,
                        use_container_width=True
                    )

                # Proceed to business analysis
                if st.button("🔍 Proceed to Business Analysis", 
                           help="Analyze business viability through customer sentiment"):
                    st.session_state.phase = 2
                    st.rerun()

        except Exception as e:
            st.error(f"🔥 System error: {str(e)}")
            st.stop()

    # Phase 2: Business Sentiment Analysis (MODIFIED)
    elif st.session_state.phase == 2:
        st.header("2️⃣ Business Viability Analysis")
        st.subheader(f"Analyzing business sentiment for {st.session_state.selected_candidate}")
        
        with st.spinner("Analyzing customer sentiment..."):
            try:
                # Load sample sentiment data (replace with your data source)
                data_url = "C:\\Users\\Charu\Downloads\\archive (2)\\Reviews.csv"
                df = pd.read_csv(data_url)
                df = df.head(500)  # Limit to 500 reviews for demo

                # Sentiment analysis
                sia = SentimentIntensityAnalyzer()
                res = {}
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    text = row['Text']
                    myid = row['Id']
                    res[myid] = sia.polarity_scores(text)
                
                vaders = pd.DataFrame(res).T
                vaders = vaders.reset_index().rename(columns={'index': 'Id'})
                vaders = vaders.merge(df, how='left')

                # Sentiment visualization
                st.subheader("Customer Sentiment Analysis")
                
                # Create 2x2 grid for visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Compound Score Analysis
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    sns.barplot(x='Score', y='compound', data=vaders, ax=ax1)
                    ax1.set_title('Compound Sentiment Score by Rating')
                    st.pyplot(fig1)

                    # Positive Sentiment Analysis
                    fig3, ax3 = plt.subplots(figsize=(10, 4))
                    sns.barplot(x='Score', y='pos', data=vaders, ax=ax3)
                    ax3.set_title('Positive Sentiment Distribution')
                    st.pyplot(fig3)

                with col2:
                    # Rating Distribution
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    sns.countplot(x='Score', data=vaders, ax=ax2)
                    ax2.set_title('Customer Rating Distribution')
                    st.pyplot(fig2)

                    # Negative/Neutral Analysis
                    fig4, ax4 = plt.subplots(figsize=(10, 4))
                    sns.barplot(x='Score', y='neg', data=vaders, ax=ax4)
                    ax4.set_title('Negative Sentiment Distribution')
                    st.pyplot(fig4)

                # Approval logic
                st.subheader("Business Viability Decision")
                approval_cols = st.columns(3)
                
                # Calculate approval metrics
                positive_threshold = 0.5  # 50% positive threshold
                approval_metrics = {
                    'Avg Compound Score': vaders['compound'].mean(),
                    'Positive Reviews (%)': (vaders['compound'] > positive_threshold).mean(),
                    'Critical Reviews (%)': (vaders['compound'] < -0.5).mean()
                }
                
                # Display metrics
                with approval_cols[0]:
                    st.metric("Average Sentiment", 
                            f"{approval_metrics['Avg Compound Score']:.2f}",
                            help="-1 (Negative) to +1 (Positive)")
                    
                with approval_cols[1]:
                    st.metric("Positive Reviews", 
                            f"{approval_metrics['Positive Reviews (%)']:.2%}",
                            help=f"Score > {positive_threshold}")
                    
                with approval_cols[2]:
                    st.metric("Critical Reviews", 
                            f"{approval_metrics['Critical Reviews (%)']:.2%}",
                            help="Score < -0.5")

                # Final decision
                st.markdown("---")
                decision_container = st.container()
                credit_approval = st.session_state.credit_results[
                    st.session_state.credit_results['ID'] == st.session_state.selected_candidate
                ].iloc[0]['Recommendation']
                
                business_approval = approval_metrics['Positive Reviews (%)'] > 0.5
                
                with decision_container:
                    if credit_approval == 'Approve' and business_approval:
                        st.success("✅ Final Decision: Loan Approved")
                        st.balloons()
                        st.markdown("**Approval Criteria Met:**")
                        st.markdown("- Credit repayment probability ≥ 75%  \n"
                                  "- Positive business reviews ≥ 50%")
                    else:
                        st.error("❌ Final Decision: Loan Rejected")
                        st.markdown("**Rejection Reasons:**")
                        if credit_approval != 'Approve':
                            st.markdown("- Does not meet credit requirements")
                        if not business_approval:
                            st.markdown("- Insufficient positive business reviews")

            except Exception as e:
                st.error(f"Sentiment analysis error: {str(e)}")
                st.stop()

        # Navigation
        st.markdown("---")
        if st.button("← Return to Financial Analysis"):
            st.session_state.phase = 1
            st.rerun()

if __name__ == "__main__":
    main()