import joblib

# Load the trained model
model = joblib.load('spam_classifier.pkl')

# Example usage
sample_messages = [
    "Congratulations! You’ve won a free ticket to Bahamas.",
    "Hey, just checking if we're still on for lunch tomorrow.",
    "An Amazing Game; Won through Free Kicks"
    "Kenya won an entertaining match against Angola yesterday. The match was played at a Sold out MISC Kasarani. The players who did an amazing job had one player down since the 21st minute. They have gave a spirited fight. They are guaranteed to receive a large sum of money from president Ruto"
]

predictions = model.predict(sample_messages)

for msg, label in zip(sample_messages, predictions):
    print(f"Message: {msg}\nPrediction: {'Spam' if label == 1 else 'Ham'}\n")
