import joblib

# Load your model
loaded_model = joblib.load('FakeNewsClassifier1.pkl')

# Sample political news titles
political_news_titles = [
    "President Signs Landmark Climate Change Legislation",
    "Supreme Court Rules on Major Voting Rights Case",
    "Congress Passes New Infrastructure Bill",
    "Prime Minister Addresses Nation on Economic Recovery Plan",
    "Senate Confirms New Secretary of Defense",
    "Election Commission Announces Dates for General Elections",
    "UN Security Council Discusses Global Security Challenges",
    "Opposition Leader Criticizes Government's Healthcare Policy",
    "Government Implements New Tax Reforms",
    "Diplomatic Talks Begin Between Neighboring Countries Over Border Dispute",
    "Politician Caught Using Time Machine to Win Elections",
    "Government Plans to Ban All Public Protests Permanently",
    "World Leaders Secretly Meet to Form One-World Government",
    "Politician Claims to Have Superpowers During Election Rally",
    "New Law Requires All Citizens to Submit DNA Samples",
    "President Declares Martial Law Nationwide Without Warning",
    "Senator Admits to Being a Reptilian Alien in Disguise",
    "Secret Documents Reveal Plan to Control Minds of Voters",
    "Prime Minister Seen Performing Magic Rituals in Office",
    "Politician Claims to Have Found Fountain of Youth"
]

# Function to clean and predict
def predict_news(news_titles, model):
    # Clean the titles (if needed)
    
    # Predict
    predictions = model.predict(news_titles)
    
    return predictions

# Get predictions
predictions = predict_news(political_news_titles, loaded_model)

# Print results
for title, prediction in zip(political_news_titles, predictions):
    print(f"Title: {title}\nPrediction: {'Fake' if prediction == 1 else 'Real'}\n")
