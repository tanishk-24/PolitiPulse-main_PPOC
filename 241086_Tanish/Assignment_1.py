"""
Assignment 1: Sentiment Analysis with VADER
"""

import string
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Public comments
comments = [
    "The New Roads Policy is AMAZING!!!",
    "i dont think the new tax rule helps anyone.",
    "The committee met on Tuesday to review the budget.",
    "Honestly, this healthcare reform is a disaster.",
    "Farmers finally got the support they deserve, great move!!"
]

print("="*50)
print("Question 1 — Preprocessing Function + VADER Scoring")
print("="*50)

# (a)
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return tokens

for comment in comments:
    print(f"Original: {comment}")
    print(f"Tokens: {preprocess(comment)}\n")

# (b)
analyzer = SentimentIntensityAnalyzer()
for comment in comments:
    score = analyzer.polarity_scores(comment)
    print(f"Original: {comment}")
    print(f"Compound Score: {score['compound']}\n")


print("="*50)
print("Question 2 — Build a Mini Approval-Rating Pipeline")
print("="*50)

# (a)
def get_compound_scores(comments_list):
    return [analyzer.polarity_scores(c)['compound'] for c in comments_list]

# (b)
def average_score(scores):
    if not scores: return 0
    return sum(scores) / len(scores)

# (c)
def approval_rating(avg):
    if avg >= 0.5:
        return "Strongly Approve"
    elif 0.05 <= avg < 0.5:
        return "Approve"
    elif -0.05 < avg < 0.05:
        return "Neutral"
    elif -0.5 < avg <= -0.05:
        return "Disapprove"
    else:
        return "Strongly Disapprove"

# (d)
scores = get_compound_scores(comments)
avg = average_score(scores)
rating = approval_rating(avg)
print(f"Average Compound Score: {avg:.4f}")
print(f"Final Approval Rating Label: {rating}\n")


print("="*50)
print("Question 3 — Per-Comment Sentiment Classifier")
print("="*50)

def classify_sentiment(text):
    compound = analyzer.polarity_scores(text)['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

for comment in comments:
    label = classify_sentiment(comment)
    print(f"Comment: {comment}")
    print(f"Label: {label}\n")
