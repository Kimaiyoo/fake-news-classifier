from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("FakeNewsClassifier1.pkl")  # Load your trained joblib model

def requestResults(name):
    # Corrected: Use the predict method on your model
    check = model.predict([name])
    out = "Possible Fake News" if check == 1 else "Possible Real News"
    return out

@app.route("/", methods=["GET", "POST"])
def predict():
    predictions = ''
    if request.method == "POST":
        # Get input data from the user (e.g., form submission)
        input_data = request.form.get("input_data")  # Adjust this based on your form field name

        # Process the input data (if needed)
        # Make predictions using your model
        predictions = requestResults(input_data)

    # Render the main.html template with predictions
    return render_template("main.html", predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)


