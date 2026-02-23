#List of Spam Words
spam_keywords = [
    "win", "winner", "congratulations", "urgent", "act now", "limited time", "exclusive deal",
    "click here", "free", "offer", "amazing", "guaranteed", "no cost", "risk-free", "4U",
    "credit card", "loan", "get rich", "easy money", "cheap", "discount", "subscribe now",
    "buy now", "apply now", "investment", "claim your prize", "unsecured", "clear debt"
]

message = input("Enter Your Message: ")

is_spam = False

for keyword in spam_keywords:
    if keyword.lower() in message.lower():
        is_spam = True
        break

if is_spam:
    print("Spam")
else:
    print("Not Spam")
    


