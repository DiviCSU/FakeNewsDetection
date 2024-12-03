from fake_news_detector import FakeNewsDetector

def main():
    # Instantiate the class
    detector = FakeNewsDetector('Cleaned_True_Data.csv', 'Cleaned_False_Data.csv')

    # Preprocess the data
    detector.preprocess_data()

    # Train the model
    detector.train_model()

    # Predict new text
    new_text = "Your sample news article text here."
    print(f"The news is: {detector.predict(new_text)}")

if __name__ == "__main__":
    main()
