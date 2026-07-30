# 🧴 Customer Feedback Analysis & AI-Powered Customer Support
### Transforming Customer Reviews into Actionable Insights using Python & Google Gemini AI

## 📖 Project Overview

Customer feedback provides valuable insights into product quality, customer satisfaction, and areas for improvement. Analyzing large volumes of customer reviews manually is time-consuming and inefficient.

This project automates the analysis of skincare product reviews using **Python** and **Google Gemini AI**. It cleans and preprocesses customer review data, identifies critical reviews, extracts common complaint keywords, visualizes customer feedback trends, and generates personalized AI-powered customer support emails. The project demonstrates an end-to-end data analytics workflow combined with generative AI to help businesses respond to customer concerns more efficiently.
---

## Objective

The main objectives of this project are:

- Clean and preprocess customer review data.
- Identify critical customer reviews using rule-based filtering.
- Extract the most common complaint keywords.
- Visualize review insights using charts.
- Generate personalized customer support emails using Google Gemini AI.

---

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- google-genai

---

## Workflow

### Step 1
Import the required libraries.

### Step 2
Load all customer review datasets.

### Step 3
Merge the datasets into a single DataFrame.

### Step 4
Explore the dataset using:

- head()
- shape
- info()
- describe()
- sample()

### Step 5
Handle missing values and remove duplicate records.

### Step 6
Clean the review text by:

- converting to lowercase
- removing punctuation
- removing extra spaces

### Step 7
Identify critical reviews using ratings.

- Critical Reviews → Rating ≤ 2
- Non-Critical Reviews → Rating > 2

### Step 8
Perform complaint keyword analysis by calculating the frequency of words in critical reviews.

### Step 9
Create visualizations:

- Critical vs Non-Critical Reviews (Bar Chart)
- Critical vs Non-Critical Reviews (Pie Chart)
- Top 10 Complaint Keywords (Bar Chart)

### Step 10
Select the Top 3 Critical Reviews.

### Step 11
Generate professional customer support emails using Google Gemini AI.

---

## Project Outputs

- Cleaned review dataset
- Critical review dataset
- Complaint keyword frequency table
- Bar Chart
- Pie Chart
- Top 10 Complaint Keywords Chart
- Top 3 Critical Reviews
- AI-generated Customer Support Emails

---

## Project Structure

```
Customer_Feedback_Analysis/
│
├── Customer_feedback_analysis.ipynb
├── README.md
├── product_info.csv
├── product_info_skincare.csv
├── reviews_0_250_masker.csv
├── reviews_250_500_masker.csv
├── reviews_500_750_masker.csv
├── reviews_750_1250_masker.csv
├── reviews_1250_end_masker.csv
└── generated_customer_emails.csv
```

---

## How to Run

1. Install the required libraries.
2. Generate a Google Gemini API key.
3. Add the API key in the notebook.
4. Run all cells in sequence.
5. View the generated charts and AI-generated customer support emails.

---

## Google API key
API key: "Please Enter your API key"
## Future Improvements

- Perform sentiment analysis using machine learning.
- Build an interactive dashboard.
- Support multilingual reviews.
- Automate email delivery.

## Challenges faced while working on the project

During the development of this project, several practical challenges were encountered, which helped improve my understanding of Python, data analysis, and project debugging.
- Stopwords drowning out real complaint keywords: the first pass at keyword extraction just split and counted every word in the review text, which meant common English words ("the", "and", "it", "was"...) dominated the top results instead of actual complaint terms. I had to go back to steps 6 and 7, build a custom stopword list, and add a filtering loop to strip these words out before counting — only after that did meaningful complaint keywords (related to things like skin reaction, dryness, smell, etc.) actually surface at the top.

- Getting the Gemini API key: working out the right place to generate a Google Gemini API key (Google AI Studio vs. Google Cloud Console) and getting it enabled/authorized correctly took a bit of trial and error before requests started going through.

- Gemini model/SDK compatibility: not every Gemini model worked out of the box — I tried a couple of variants including a "lite" model, and several returned errors or weren't accessible with the SDK version I was using. gemini-2.0-flash (the latest flash model) ended up being the one that worked reliably.

- Prompt design for the apology emails: getting the prompt right so that Gemini generated a response that was genuinely personalized to each review's specific complaint (rather than a generic apology), and correctly passing that prompt + review text into the SDK call, took a few iterations to get the output format and tone right.

## Resources
**Google Gen AI (Gemini) Python SDK Documentation** – https://ai.google.dev/gemini-api/docs
- **Google AI Studio** – https://aistudio.google.com/
- **Kaggle** (Dataset and Data Science Resources) – https://www.kaggle.com/
- **Stack Overflow** – https://stackoverflow.com/ (used for troubleshooting Python and API-related issues)

---

## Bhavana Nagula
