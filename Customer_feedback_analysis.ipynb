{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7607acaf",
   "metadata": {},
   "source": [
    "# Using the Skincare Products▴EDA & Sentiment Analysis dataset for the assessment"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e85aba46",
   "metadata": {},
   "source": [
    "# Project steps\n",
    "\n",
    "Step-1: Import Libraries\n",
    "\n",
    "Step-2: Load all the reviews data set\n",
    "\n",
    "step-3: merge the datasets into a dataframe\n",
    "\n",
    "step-4: Explore the dataset\n",
    "\n",
    "step-5: Handle the missing values\n",
    "\n",
    "step-6: clean the reviews datasets\n",
    "\n",
    "step-7: Rule based filtering\n",
    "\n",
    "step-8: complaint keyword analysis\n",
    "\n",
    "step-9: visualising the data\n",
    "\n",
    "step-10: select top 3 critical values\n",
    "\n",
    "step-11: Generate AI emails using google gemini API\n",
    "\n",
    "step-12: creating the readme file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62b21f0a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step-1: Import the libraries\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy import stats\n",
    "from google import genai # will use this to generate ai response emails\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8205a610",
   "metadata": {},
   "outputs": [],
   "source": [
    "#step-2: load all the customer reviews datasets\n",
    "review1=pd.read_csv(\"reviews_0-250_masked.csv\")\n",
    "review2=pd.read_csv(\"reviews_250-500_masked.csv\")\n",
    "review3=pd.read_csv(\"reviews_500-750_masked.csv\")\n",
    "review4=pd.read_csv(\"reviews_750-1250_masked.csv\")\n",
    "review5=pd.read_csv(\"reviews_1250-end_masked.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "38b231e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "#step-3: Merge all the reviews\n",
    "reviews=pd.concat([review1,review2,review3,review4,review5],\n",
    "                   ignore_index=True) #used to create new row numbers to avoid the confusion"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cbd75786",
   "metadata": {},
   "outputs": [],
   "source": [
    "#step-4: Explore the dataset\n",
    "reviews.head() #displays the first 5 rows of the dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ad1c662",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews.shape #displays no of rows and columns exist in the given data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c5baca04",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews.columns #displays the list of all the column names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17e422ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews.info() #displays summary of datatypes, structure layout and memory usage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6e5809fd",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews.describe() #gives summary statistic of numeric columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a7566424",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews.sample(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "74f22551",
   "metadata": {},
   "outputs": [],
   "source": [
    "# step-5: Handle the missing values (Data cleaning)\n",
    "reviews.isnull() #used to find the null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "628d5c69",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews.isnull().sum() # used to count the no of null values in the series of the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "741c4572",
   "metadata": {},
   "outputs": [],
   "source": [
    "missing_values=reviews.isnull().sum()\n",
    "missing_values=missing_values[missing_values>0]\n",
    "print(missing_values)\n",
    "# here we are filtering the sum of data with null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2fcdb3b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews=reviews.dropna(subset=[\"review_text\"]) #creating a subset only remove rows where the review_text column is missing.\n",
    "reviews[\"review_text\"].isnull().sum()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d2566a9",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviews[\"review_text\"].isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "172fbe4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#check for duplicate rows\n",
    "reviews.duplicated().sum()\n",
    "#remove duplicatesif existed\n",
    "reviews = reviews.drop_duplicates()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8a444860",
   "metadata": {},
   "outputs": [],
   "source": [
    "#check final dataset size\n",
    "# Verify duplicates are removed\n",
    "print(\"Duplicate rows after removal:\", reviews.duplicated().sum())\n",
    "\n",
    "# Check final dataset size\n",
    "print(reviews.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce2fd6fa",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "418c33df",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Step-6: Step 6 – Clean the Reviews dataset\n",
    "\n",
    "# Convert to lowercase\n",
    "reviews[\"review_text\"] = reviews[\"review_text\"].str.lower()\n",
    "\n",
    "# Remove punctuation and numbers\n",
    "reviews[\"review_text\"] = reviews[\"review_text\"].str.replace(\n",
    "    r\"[^a-zA-Z\\s]\",\n",
    "    \"\",\n",
    "    regex=True\n",
    ")\n",
    "\n",
    "# Remove extra spaces\n",
    "reviews[\"review_text\"] = reviews[\"review_text\"].str.replace(\n",
    "    r\"\\s+\",\n",
    "    \" \",\n",
    "    regex=True\n",
    ")\n",
    "\n",
    "# Remove leading/trailing spaces\n",
    "reviews[\"review_text\"] = reviews[\"review_text\"].str.strip()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11bc47a5",
   "metadata": {},
   "source": [
    "^ - NOT\n",
    "a-z - Lower case\n",
    "A_Z - Upper Case\n",
    "\\s - spaces\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e00c428",
   "metadata": {},
   "outputs": [],
   "source": [
    "# step-7: Rule based filtering\n",
    "\n",
    "#creating a dataframe for critical reviews\n",
    "critical_reviews=reviews[reviews[\"rating\"]<=2]\n",
    "\n",
    "#reset the index\n",
    "critical_reviews=critical_reviews.reset_index(drop=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f2b6d23",
   "metadata": {},
   "outputs": [],
   "source": [
    "# creating dataframe for non critical reviews\n",
    "non_critical_reviews = reviews[reviews[\"rating\"] > 2]\n",
    "\n",
    "# Reset the index\n",
    "non_critical_reviews = non_critical_reviews.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "427bfbb2",
   "metadata": {},
   "outputs": [],
   "source": [
    "critical_reviews.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "89b95513",
   "metadata": {},
   "outputs": [],
   "source": [
    "non_critical_reviews.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f119cad",
   "metadata": {},
   "outputs": [],
   "source": [
    "# creating the count of the reviews\n",
    "print(\"total_reviews:\",len(reviews))\n",
    "print(\"Critical_reviews:\",len(critical_reviews))\n",
    "print(\"Non_critical_reviews:\",len(non_critical_reviews))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83c301b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# comparing the critical and non critical reviews\n",
    "summary=pd.DataFrame({\"Category\":[\"Critical\",\"Non-Critical\"],\n",
    "                                  \"Count\":[len(critical_reviews),\n",
    "                                           len(non_critical_reviews)\n",
    "                                           ]\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "00d39245",
   "metadata": {},
   "outputs": [],
   "source": [
    "# calculating the percentage\n",
    "summary[\"percentage\"]=(summary[\"Count\"]/len(reviews))*100\n",
    "summary[\"percentage\"] = summary[\"percentage\"].round(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34d45209",
   "metadata": {},
   "outputs": [],
   "source": [
    "#comparing the average of the ratings\n",
    "summary[\"Average Rating\"] = [critical_reviews[\"rating\"].mean(),non_critical_reviews[\"rating\"].mean()]\n",
    "summary[\"Average Rating\"] = summary[\"Average Rating\"].round(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d8573d5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(summary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9945deb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#step-8:complaint keyword analysis\n",
    "\n",
    "#finding the top complaint keywords from critical_reviews dataframe\n",
    "\n",
    "review_text = critical_reviews[\"review_text\"]\n",
    "all_reviews = \" \".join(review_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dc036432",
   "metadata": {},
   "outputs": [],
   "source": [
    "#splitting all the reviews into individual words\n",
    "words = all_reviews.split()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "571a6c32",
   "metadata": {},
   "outputs": [],
   "source": [
    "# filtering the common words and symbols\n",
    "stop_words = [\n",
    "    \"the\",\"and\",\"is\",\"a\",\"an\",\"to\",\"of\",\"for\",\"in\",\"on\",\"at\",\"with\",\n",
    "    \"this\",\"that\",\"it\",\"its\",\"i\",\"my\",\"me\",\"we\",\"our\",\"you\",\"your\",\n",
    "    \"after\",\"before\",\"when\",\"where\",\"then\",\"than\",\n",
    "    \"only\",\"really\",\"also\",\"still\",\"just\",\"even\",\n",
    "    \"got\",\"get\",\"use\",\"used\",\"using\",\n",
    "    \"reviews\",\"review\",\"product\",\"products\",\n",
    "    \"one\",\"two\",\"first\",\"last\",\"time\",\n",
    "    \"nice\",\"good\",\"better\",\"give\",\"stars\",\n",
    "    \"went\",\"made\",\"make\",\"like\",\"love\",\"want\",\n",
    "    \"think\",\"well\",\"much\",\"more\",\"now\",\"around\",\n",
    "    \"all\",\"out\",\"dont\",\"didnt\"\n",
    "]\n",
    "\n",
    "filtered_words = []\n",
    "\n",
    "for word in words:\n",
    "    if word not in stop_words and word not in [\"|\",\"\",\"-\",\".\",\",\",\"(\",\")\"]:\n",
    "        filtered_words.append(word)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "593af2af",
   "metadata": {},
   "outputs": [],
   "source": [
    "# counting each word by creating an empty dictionary\n",
    "word_frequency = {}\n",
    "\n",
    "for word in filtered_words:\n",
    "    if word in word_frequency:\n",
    "        word_frequency[word] += 1\n",
    "    else:\n",
    "        word_frequency[word] = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb4e9b50",
   "metadata": {},
   "outputs": [],
   "source": [
    "# converting the dictionary into a data frame\n",
    "df_freq = pd.DataFrame(\n",
    "    word_frequency.items(),\n",
    "    columns=[\"Keyword\", \"Frequency\"]\n",
    ")\n",
    "\n",
    "df_freq = df_freq.sort_values(\n",
    "    by=\"Frequency\",\n",
    "    ascending=False\n",
    ")\n",
    "\n",
    "df_freq = df_freq.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba6c374c",
   "metadata": {},
   "source": [
    "Above we can see that these are not complaint keywords they are comon english keywords called stop words"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "200036eb",
   "metadata": {},
   "source": [
    "# Top 10 complaint keywords\n",
    "\n",
    "skin\n",
    "\n",
    "dry\n",
    "\n",
    "face\n",
    "\n",
    "out\n",
    "\n",
    "its\n",
    "\n",
    "all\n",
    "\n",
    "did'nt\n",
    "\n",
    "cleanser\n",
    "\n",
    "mask\n",
    "\n",
    "don't"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc36f9ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "# step-9: Visualizing the Data\n",
    "\n",
    "#comparing the critical vs non critical reviews using the bar chart\n",
    "plt.figure(figsize=(6,4))\n",
    "plt.bar(summary[\"Category\"],summary[\"Count\"])\n",
    "plt.title(\"Critical vs Non-Critical Reviews\")\n",
    "plt.xlabel(\"Review Category\")\n",
    "plt.ylabel(\"Number of reviews\")\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "id": "8947ada9",
   "metadata": {},
   "source": [
    "# Bar chart representing the critical vs Non-Critical reviews\n",
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d08f8033",
   "metadata": {},
   "outputs": [],
   "source": [
    "# percentage of critical and non critical reviews\n",
    "plt.figure(figsize=(6,6))\n",
    "\n",
    "plt.pie(\n",
    "    summary[\"percentage\"],\n",
    "    labels=summary[\"Category\"],\n",
    "    autopct=\"%1.1f%%\",\n",
    "    startangle=90\n",
    ")\n",
    "\n",
    "plt.title(\"Percentage of Critical and Non-Critical Reviews\")\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "iVBORw0KGgoAAAANSUhEUgAAAeEAAAH3CAIAAAADp3DfAAAQAElEQVR4AeydB5wTxfvG31w/7ui99yIoIMWCKAIqKIogoiIoIjZEFP2JDXvF9lc60quIDRULIiBFepWOdDh6P7je/k9YDDGX5HKXTbKz+/AZ9nZnZ9555/vuPjs7u0nCcviPBEiABEjAqATChP9IgARIgASMSoAabdTI0C8SIAFPBKyUT422UrTZVxIgAdUIUKNVixj9JQESsBIBarSVos2+mpkA+2ZOAtRoc8aVvSIBEjAHAWq0OeLIXpAACZiTADXanHFlrzQCXJKA6gSo0apH0NX//fv3//bbb99+++2pU6dc9xV0e9u2bTCYnZ3txYAvZbxUd+zSy47DoPeVffv2oWvnzp3zXkyVvb7Q86WML/3Vy44vbbmUOXnyJKJ29OhRl3xTbhpCo7VgAzrSDz/8sHz58rS0NAPi3rx5MzzMyckxoG+aS4MGDapbt+6QIUO++uorHMdaZu5lVlbW+vXrf/nll/nz5+/atcu7+KI6et21a9f09HSsI7nl4FIGxQqW9LLjY+t//vknunbw4EG35QEH/syaNctlLw5RHKgumYHYVDdSGjrQQwKrJUuWJCcn64Vo69atiNq6dev0MmhkO4bQaEQRxL/44gsoy+jRozt27FixYsUZM2YYDdz06dPhJ04bozmm+XP69OnXX3/9zTff1MbRtWvX1vKdl3B+0KBBZcuWvf3224cNG/bBBx80b968Zs2akHXnYi7rl112WZcuXcLDw7V8txxcymglVV+CJCKOA3L27NnOffnkk0969OjhnKP7uuqR0tANHToUJ/XYsWPvu+++cuXKjRo1ShdQpUqVwgEJg7pYM7gRQ2i0xujjjz+GWP/666+4AteqVQvnwPbt27VdXPpCANwyMjIwjvZUGHcAUJy33377s88+O3DgAM6iP/7449ChQ4899tjIkSM91UI+zgeEJjIyEuueki9lPNU1eH5MTMyLL76Y592Gjr0wTaTeeustHDk///wzDs7rr7++T58+q1at8h9UvXr1YLZx48b+mzK+BQNptANW4cKFn3766czMTOi1lolDduPGjbjlXLhwYUpKipaJJUaOCJV2o7phwwbcUjlu8FNTU5cuXYrb+R07dqCkc4KFxYsX//TTT6jinI8mvv/+e+RgpgWTAHPmzElM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"
    }
   },
   "cell_type": "markdown",
   "id": "a13cc0ea",
   "metadata": {},
   "source": [
    "# pie chart representing the Critical and Non Critical reviews percentage\n",
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "54a1bab7",
   "metadata": {},
   "outputs": [],
   "source": [
    "top_keywords = df_freq.head(10)\n",
    "\n",
    "plt.figure(figsize=(10,5))\n",
    "\n",
    "plt.bar(\n",
    "    top_keywords[\"Keyword\"],\n",
    "    top_keywords[\"Frequency\"]\n",
    ")\n",
    "\n",
    "plt.title(\"Top 10 Complaint Keywords\")\n",
    "plt.xlabel(\"Keywords\")\n",
    "plt.ylabel(\"Frequency\")\n",
    "\n",
    "plt.xticks(rotation=45)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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5BHPc86fP584ceJkHj/Tp0+3Tvj999+feuopHUybNq1S6cWLF2N5yLqcHQQQQAABBBBAAIGYCniVF1esWFG6dOlevXrlzZvXeoIKFSosW7bMenj7Tnh4eOfOnatVq9asWbPbj6pm9erV1iRj06ZNVaOiGoXFhAkTHjx4cP369d99913btm1Vr+LbIV3ookJXEEAAAQQQQACBuBbwKi8eO3YsV65calpISIi2pmiJ+cqVK2Y/ym3SpEnXrFmjIKiZwihPiLJy6dKlO3fuHD16dLZs2R544IG+ffvOmDHj+PHjOtm3Q7qQggACCCCAgM0EaA4CThLwKi8WKVJk1apVERERISH/zYtnz5794osvYvnfA6xRo0aSJEnuv//+oUOHKn0atrVr1+bLly937tzm4ZNPPqnn3bBhgx76dkgXUhBAAAEEEEAAAQR8FvAqL1aqVClTpkxVq1bVfOGePXsGDBhQrFixDBkyhIaG+vbEISEh7dq127x58+nTp99//32tdPfp08fcSlOJmTNnNvva6nl1siq1r60Ph3ShVbRErqRrFaueHQSiFqAWAQQQQAABBO65x6u8qMS2ZMmSEiVKrF+/XjtDhgypUqXK8uXLEyVK5JthypQpteJcoEAB7dStW7dLly6aYrzLX5FWA+70RDE6pKSb9taPWWG/022pRwABBBBAAAH3CNCT2Al4lRf1FKlSpVJMPHTokFaH//zzz48//ljzi6r3SylatOi5c+dOnDihu2XPnt3saF/l5MmTypHZsmXTvm+HdKFVunXrdubWT1hYmFXPDgIIIIAAAggggMCdBLzNi3e63i/127dvT548efr06XW3Rx999ODNH+2rrFixImHChOaf4/HtkG5ilaRJk6bx+LHq2UEAATsI0AYEEEAAAXsKeJUXp0yZoinA24tmHO/eq+vXr1+7dk0ThDpNOyraUdFi9LBhw/bv33/q1KkZM2YMHjy4Q4cOSZIk0aHq1asXK1asTZs2v//++9atW999991mzZqZP7bo2yHdk4IAAggggAACCCDgs4BXeVEBruV/f/75v4YNG6ZOnTo8PLxMmTJ3f+KSJUsmS5ZM6XDbtm3aUblw4YIuadKkiRaaa9Soce+99w4dOlS5c+DAgapX0WzikiVLUqRIoSdVQAwNDR01apTqVXw7pAspCCCAAAIIIIAAAj4LeJUXH374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mjVr1KhRMETGP/74Y/To0SLZvHlzsWLFLl68qH1HF8X9tm3bOroLXjb+Tqf1799//fr1dzpq8/pLly6NHTvW5o30pnnnzp0bMGBARESEfsX0m3XgwAFvrrLzOeRFO49OjNvmji8xMe62Qy4ICQl56qmnWrVqVapUqeeff/6NN95wSMOjaeb58+d79+793HPPHT9+PJpTbX/4yJEj169fVzN79OjRs2fPYIiMv/zyi16KClihoaEffvhhihQp1H1Hl9OnT0+cOHHVqlWO7oXPjV+4cOH06dMLFizo8x3i+EK9b+i7tPWk48ePX716tfXQuTsKiIMHD9Z7SKVKldq3b58vXz7n9sW0nLxoHBy/nTFjRsaMGdOmTduxY0eX/dGrTz/9tGjRolmyZGnSpMmJEyccNVSRG9ugQYMcOXKEhYXlzp078jFnPv7rr79KliypGZ1Ro0ZlzZrVmZ34b6v1oVWzZs3GjRsHVWSsVatWr169xo0bV61aNe3/18LJ/6cP5meeeeajjz5ycid8bPvnn38+adIk5X59HPh4i7i9TL9rilOvvPKKfvvMM589ezZlypRm39FbzSlqLObMmZM5c+Z27do5ui+m8eRF4+Ds7bfffqvpEOUqfa388ssvNYmlKX1nd+lW68eMGaNpHi2vqHdaL6tevbr1tnLrFCf9f6pUqfSNU9FKs4xamHZS0+/Q1i5dumhQhg4dqhysU369+aMdJxZNAH/88cdff/11sEVGfUJrHGfPnt2vXz8nDtztbX7zzTcXLVr022+/3X7IxTWHDx9u3bq1Op4sWTKndDNhwoRafdZ8hxUZL168aNdJ7hij/vnnn3op7tmzp0WLFlqYjvH1NruAvGizAYl5c9avX6/fN5UaNWpolXPNmjX79u3T0pILIqM+xrp3767P76efflowp06dUnzUh7r2HVc0D/fqq6/qLUNTjPquOWLECM/I+Pvvv2vUnNUpva2rwevWrStTpox2FBT1ReWBBx64//77R44cqRonltKlSy9btkwvuaCKjANv/igv9u7d2zMyOug9RDPcgwYNOnbsmHnVlS9fXkM5bNgw8zBItv/5z3/06tUqk8bx2rVrTun1E0888dVXX1mRUW8srsmLiu9DhgzRoCxYsMAzMjroN8vzVURe9NRwwL7eBbp27apvLaate/fufeyxx/TLpiUYU5M3b96VK1cqfLggMm7fvj1XrlwFChTQBKo+v+fPn1+uXDl1c8KECdo6qyj7qv0KvpcvX1bLrciod5MNGzZUrFjxxx9/VL3Niz6VO3fufOXKlf379xctWlTfm/UV5bXXXtM6plJj2bJlz5w5o4/teFsK9Aefcobe34MhMipdaeFy4sSJFy5ckNxzzz1nIqPShh5+8803Dz30kMZa+zYvShg//fSTliDy5MnTvHnzX375RQ1+4403pkyZoi+Z2g+eYl69EmjSpImWep3Scc/IeP78+a1btyrre5awsDCn9EXt1C/U6NGj1X7zB6jMoCgyvvTSS1evXtXoFC5c+OjRozrTWYW86KzxukexY8mSJc8++6xpt7LUzJkz9b7g+QltRUatMZnTHLrNmjWrcokW2T3DotZcNMvouB4p0CvH79692zMyTp48uVevXlWqVNEndLNmzezfKX1R0TRA7dq1n3zyyW7duhUqVEgt12K03u537tz53nvvacE9Xbp09957r/37cpcWmvf3KCPjtm3b7nKhgw5pvEqUKLF48eJ3331X/T1w8y9vKjJ+/vnnmm3U55kCh/JWkiRJbN4pfQGrWrXquXPnfv75Z31z1ktU/apWrZpeihkzZhw3bpzN2x/75mkBWqsWepP87rvvdDeNZqQvPKq0c9GqiwZR7yEaPr296BWoRSTNEagX+tKiX0N96mniwM5d8GzbyZMnS5YsOW3aNC0iFS9eXAuAOqpBUY/UHS2/qKeDBw/Onj276p1V4jovOkvHhq3NkCGD3hSGDx+ul6PeGdXCF154QZHRJA89NEWR8YcfftBnuXnouK2mqdTm++67r1SpUk2bNv3kk0/MzOKNGzc0M6dPMh11RPFcd8ifP3+kyKiuaY7n+PHj2rFzd5YuXaoPY7VQ30/0StP7uEKhviurRu/sarwmHc2fX9SLU6mxf//+OuToovd3dVOfVfoY1vcx9aVHjx56qCyifRcU/R598MEHekH++uuvGruKFSuayFinTh2NdceOHTW7rwlj+/dUb4a5c+dWs9XUypUrKzzt2bNHn8qKUPpuOWrUKEdMkarxvpW+ffu++uqryiXJkydXSjZ/FOT2V69vNw/0VeHh4W+//bbeVfRS1HMpSCkyKj4qHeqbjPb1zqNfOr2l6KhTit4otOqiVSO9DmvWrFm9evV169ap8fos08yi+quH+tRW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AIH4FiAvxvcI8PwIIIAAAsEgQB8RcLIAedHJo0fbEUAAgVsCv/766zfffHPrEf+PAAII+FOAvOhPTe7ldAHaj8DtAjt37ly4cKFn/Q8//PDZZ5+Fh4d7Vsb7/ldffdWlS5d4bwYNQAABVwqQF105rHQKAQT8JvDll1+2a9fOut2QIUOqVat27dq1pEmTWpXsIICA3QRoj38FyIv+9eRuCCDgZoFu3bq999578+fPb9SokenniRMnFi9e/N133/3999+m5urVq3PmzDl+/Lh5qK1mImfPnq0aXRgWFqYalU2bNn3xxRfaMeXzzz/XCWZ/3759OnPlypVXrlwxNdoeO3ZMk5o3btzYuHGjLjx9+rQqFVu///77ZcuWnTx5Ug89y9GjR5cuXbp69epLly551rOPAAII+CBAXvQBjUsQQMAvAk66SURERKtWrcaPH//tt9/WrFnTNF1zjffdd9/IkSP79u177733Llq0SPWJEyf+4IMPhg8frn1TFP5atmyZMmVKVU6cONFUvvrqqy+88ML+/fv1cM+ePfXq1dNTaP+NN94oXrz42LFjW7duXahQIR1Spcq2bdsaNmxYq1YtTXaavKiE+sgjjzRu3Hjo0KElSpRYsGCBTjNF7VTDVN+zZ8+HH35469atpp4tAggg4JsAedE3N65CAIEgEtA0nvLcV199tWrVqvLly5uea06xX79+mib85ptvNMk3ZsyYZs2anT17VkeVLKdNm3b9+nXtq0yePFmXp0qVqmLFipo1VM2FCxd++umnkiVLmofaKt5lz55dM4VKnz/88IPuuXPnzgceeKBt27Y63xQFSoW/zZs3a7Yyf/78/fv319zhrl27NI+4du1az1DYq1cvtUd3W7FixfLlyzXlae7AFgEEEPBNILq86NtduQoBBBBwkYBWe+fOndu+ffuiRYta3ZoyZcr999+/fft2LSVrpVhhThN+eqgTtFqtfWU+7YeFhSmxtWjRQvvKixs2bLh48aKWiQsWLKj5RQVN1Ssv6pB2Zs2aVb16deVI7SdKlKhLly46QUveemiKZiXNjrZ60jZt2qRJk0b7efLkqV+/vnZMSZ48ueKmYq4e/uc//ylTpox2KAgggIDPAuRFn+m4EAEEgkVAM38fffRRjx49rNVk9fzgwYN//fWXloYVJb/88kstByv/JUmSRIdSp06tfU0ral+xskCBAhUqVNB+2bJlEyRIoLlAExCVERUHVa8JRe1r5/fff9fEoXZM0Rq3dlSprUrChAmzZMmiHZUbN24cOnQob9682jclX758ZkfbTz75ZOHChZkzZ3766aenTp2qLKvKe/gfAggg4KsAedFXOa5DAIFgEnjjjTeGDRv2yiuvjB071vRbE3tFihTR0rBneeSRR8xRLUkvWrRIE5PKi82bNzeVSZMmVWRUWFSpVKlSqVKltH69ZMmSo0ePmryYKVOmU6dOmZO1Nfuq1H6kEhISkj59evO3Xswhz/0nnnhix44d27ZtCw0N7Xrzx5zDFgEEEPBNgLzom1ugruK+CCBgW4GOHTuOHj26Q4cOo0aNUiNr1Kjx9ddfa5ZR+6YcO3bMmskrX768phVbt26t2cGXXnrJnKCtcuHixYt/+uknRTrNFz766KO9evUyf3hRRzUNqVXsCxcuaF9FK925cuXKnTu39m8vOnn+/Pmm/vr165rgNPtqg1qifS1SqwFNmzZdv369HlIQQAABnwXIiz7TcSECCASdgOYXx48f/9prr2muUVFMaa9cuXL9+/fXOnX79u0ff/xx5TYLpWXLlspzNWvW1HK2Vam8uH379kKFCplZQz3cvHmztuaENm3a6GQ9HDduXKdOnQYNGjR06FDFSnM00rZ3797Lli1r0qTJhAkT9CxaHDcnqA2PPfaY2jNp0qTBgwd//PHHWhw3h9jGpQDPhYCbBMiLbhpN+oIAAv4X0KJznTp1rPsqBU6fPn3Dhg27d+/W/KLy3B9//KHM9+CDD2r9N3HixNaZzzzzjPbN33TRjilaj27YsKHCnHlYu3bt+vXrN2jQwDzUgvXatWt1wrp1665du7Zq1arnn3/eHFKO1Jlm32yLFSumZmhVWs+rScSpU6dqylOH1AZF0qJFi2pa8ciRI1ou18yo6ikIIICAzwLkRZ/puNAFAnQBgegFnn32WS1De57XqFGjWbNmKa5p5k9RT3OBmnTU1GOKFCk8T1u6dKlCXmhoqGelEuHMmTM1j2gqCxcurDxXqVIl81Db1KlTv/nmmwp/w4cPV7hUjSnFixf/9NNPzb61VUgdMWLEmDFjGjdu/NRTTw0cONAcUkvatm2rWU/Ng1atWtVUskUAAQR8FiAv+kzHhQgggEDUAr/++qtWgfv06dO5c2fN9kV9ErUIIOBPAe4VWAHyYmB9uTsCCAShwL59+1auXNm1a9fXXnstCLtPlxFAwH0C5EX3jSk9QsCuAkHTrpo1a86YMeP1118PCQkJmk7TUQQQcLMAedHNo0vfEEAAAQQQQACB2AtEzouxvyN3QAABBBBAAAEEEHCTAHnRTaNJXxBAAIH/CbCHAAII+EuAvOgvSe6DAAIIIIAAAgi4U4C8GL/jyrMjgAACCCCAAAJ2FyAv2n2EaB8CCCCAgBMEaCMCbhYgL7p5dOkbAggggAACCCAQewHyYuwNuYNzBGgpAggggAACCMRcgLwYczOuQAABBBBAAIH4FeDZ41bg/wEAAP//TfhTswAAAAZJREFUAwDtHSQwUbgTogAAAABJRU5ErkJggg=="
    }
   },
   "cell_type": "markdown",
   "id": "f104b06c",
   "metadata": {},
   "source": [
    "# Bar Graph showing Top 10 complaint keywords\n",
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c990bd1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(filtered_words[:50])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "af581f16",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step-10: select top 3 critical values\n",
    "\n",
    "top_3_reviews=critical_reviews.head(3)\n",
    "print(top_3_reviews[[\"product_name\",\"rating\",\"review_title\",\"review_text\"]])\n",
    "top_3_reviews=top_3_reviews.reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76ebe0b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from google import genai\n",
    "client = genai.Client(\n",
    "    api_key=\"YOUR API KEY\"\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d86c3b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "generated_emails = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad01aa5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "for index, row in top_3_reviews.iterrows():\n",
    "\n",
    "    prompt = f\"\"\"\n",
    "You are a professional customer support executive.\n",
    "\n",
    "Write a polite, empathetic, and professional email responding to the customer's review.\n",
    "\n",
    "Product Name:\n",
    "{row['product_name']}\n",
    "\n",
    "Rating:\n",
    "{row['rating']}\n",
    "\n",
    "Review Title:\n",
    "{row['review_title']}\n",
    "\n",
    "Review:\n",
    "{row['review_text']}\n",
    "\n",
    "The email should:\n",
    "1. Thank the customer for sharing their feedback.\n",
    "2. Apologize for their negative experience.\n",
    "3. Acknowledge the specific issue mentioned.\n",
    "4. Assure them that their feedback will be shared with the product team.\n",
    "5. Offer further assistance if needed.\n",
    "6. End the email professionally.\n",
    "\n",
    "Generate only the email.\n",
    "\"\"\"\n",
    "\n",
    "    response = client.models.generate_content(\n",
    "        model=\"gemini-2.0-flash\",\n",
    "        contents=prompt\n",
    "    )\n",
    "\n",
    "    generated_emails.append(response.text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4bc9222",
   "metadata": {},
   "outputs": [],
   "source": [
    "top_3_reviews[\"generated_email\"] = generated_emails "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "732581cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\n",
    "    top_3_reviews[\n",
    "        [\n",
    "            \"product_name\",\n",
    "            \"review_title\",\n",
    "            \"generated_email\"\n",
    "        ]\n",
    "    ]\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f7d01907",
   "metadata": {},
   "source": [
    "# Generated email responses\n",
    "Unnamed: 0.1,Unnamed: 0,rating,is_recommended,helpfulness,total_feedback_count,total_neg_feedback_count,total_pos_feedback_count,submission_time,review_text,review_title,skin_tone,eye_color,skin_type,hair_color,product_id,product_name,brand_name,price_usd,generated_email\n",
    "1,1,1,0.0,,0,0,0,2023-03-21,i bought this lip mask after reading the reviews and the hype unfortunately it did not meet my expectations as vaseline petroleum jelly works way better for me,Disappointed,,,,,P420652,Lip Sleeping Mask Intense Hydration with Vitamin C,LANEIGE,24.0,\"Subject: We're sorry to hear about your experience with our Lip Sleeping Mask\n",
    "\n",
    "Dear [Customer Name],\n",
    "\n",
    "Thank you for taking the time to share your feedback regarding your recent purchase of our **Lip Sleeping Mask Intense Hydration with Vitamin C**. We truly appreciate you giving our product a try.\n",
    "\n",
    "I am so sorry to hear that the lip mask did not meet your expectations, especially after all the hype and positive reviews you had read. We completely understand how disappointing it is when a product doesn't deliver the results you were hoping for, particularly when a simple staple like petroleum jelly works better for your needs. \n",
    "\n",
    "Since skin and lip chemistry can vary greatly from person to person, we know that not every product will be a perfect fit for everyone. That said, your satisfaction is extremely important to us, and we want to make things right. \n",
    "\n",
    "I would be happy to process a full refund for your purchase, or if you prefer, help you find an alternative product from our line that might suit you better. Please reply to this email with your order number or purchase details, and I will take care of this for you right away.\n",
    "\n",
    "Thank you again for your honest feedback—it helps us continuously improve. \n",
    "\n",
    "Warm regards,\n",
    "\n",
    "[Your Name]  \n",
    "Customer Support Representative  \n",
    "[Company Name]\"\n",
    "6,6,2,0.0,0.25,8,6,2,2023-03-19,ill give this stars for nice packaging and lovely scent upon initial application it feels very nice but as i continued to use it i noticed i was getting more and more blackheadsclogged pores and pimples around my lips i thought it was my aquaphor but its this it also makes my lips even more dry than they initially were disappointing around my,Dried my lips out and clogged my pores,light,blue,combination,brown,P420652,Lip Sleeping Mask Intense Hydration with Vitamin C,LANEIGE,24.0,\"Subject: We're sorry to hear about your experience with our Lip Sleeping Mask\n",
    "\n",
    "Dear [Customer Name],\n",
    "\n",
    "Thank you for taking the time to leave us your feedback regarding your recent purchase of our Lip Sleeping Mask Intense Hydration with Vitamin C. We truly appreciate you mentioning that you enjoyed the packaging and the lovely scent.\n",
    "\n",
    "However, I am sincerely sorry to hear about your experience with the product. We always aim to provide deeply hydrating skincare solutions, so it is very disappointing to learn that the mask caused increased dryness as well as clogged pores and breakouts around your lips. We understand how frustrating this must be, especially when you were expecting intense hydration.\n",
    "\n",
    "Because everyone's skin is unique, certain ingredients can sometimes cause unexpected reactions. Your satisfaction and well-being are our top priorities, and we would love the opportunity to make this right for you. \n",
    "\n",
    "Please reply directly to this email with your order number or purchase details, and I will be more than happy to issue a full refund or assist you with an exchange for a product better suited to your skin.\n",
    "\n",
    "Thank you again for bringing this to our attention, as your feedback helps us continuously improve. We look forward to resolving this for you soon.\n",
    "\n",
    "Warm regards,\n",
    "\n",
    "[Your Name]  \n",
    "Customer Care Team  \n",
    "[Company/Brand Name]\"\n",
    "13,13,1,0.0,0.4444440007209778,9,5,4,2023-03-17,honestly i was so excited when i got this in the mail but unfortunately it is not worth the hype i have boughten two of these full size and i went through them so quickly the ingredients are not very clean and this product made my lips so dry i wanted to like this product so bad as its so cute and smells nice but unfortunately it is a hard pass for me i will be sticking to my vaseline and aquaphor for now,I WISHED I LOVED THIS (not clean ingredients),,brown,dry,,P420652,Lip Sleeping Mask Intense Hydration with Vitamin C,LANEIGE,24.0,\"Subject: We’re so sorry to hear about your experience – Lip Sleeping Mask\n",
    "\n",
    "Dear [Customer Name],\n",
    "\n",
    "Thank you for taking the time to share your honest feedback regarding our Lip Sleeping Mask Intense Hydration with Vitamin C. \n",
    "\n",
    "I am truly sorry to hear that your experience fell short of your expectations. We know how exciting it is to try a new product—especially one you were looking forward to—so it is disappointing to learn that it left your lips feeling dry rather than hydrated, and that you went through the jars so quickly. \n",
    "\n",
    "We also deeply appreciate your feedback regarding our ingredients. We strive to create products that our customers can use with confidence, and hearing your thoughts on clean formulations is extremely valuable to our team as we continuously work to improve. \n",
    "\n",
    "Since you purchased two full-sized jars and were not satisfied with the results, we would love the opportunity to make things right. Please reply to this email with your order number or purchase details, and our team will be happy to assist you with a refund or help process a return. If you'd like, we can also recommend alternative products from our line that better align with your preference for clean ingredients and intense moisture.\n",
    "\n",
    "Thank you again for reaching out and helping us do better. We wish you the very best and hope to have the chance to restore your confidence in us in the future.\n",
    "\n",
    "Warm regards,\n",
    "\n",
    "[Your Name]  \n",
    "Customer Care Team  \n",
    "[Company Name]\""
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
