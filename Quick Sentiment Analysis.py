# Quick Sentiment Analysis

from textblob import TextBlob

feedback1 = "The Food at Radison was awesome"
feedback2 = "I'm fine"
feedback3 = "The Service at this flight was very bad"

blob1 = TextBlob(feedback1)
blob2 = TextBlob(feedback2)
blob3 = TextBlob(feedback3)

print(blob1.sentiment)
print('')
print(blob2.sentiment)
print('')
print(blob3.sentiment)

