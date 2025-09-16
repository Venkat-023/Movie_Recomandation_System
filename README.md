Movie Recommendation System
A Streamlit web app that recommends movies based on genres using KNN (NearestNeighbors) with pandas and scikit-learn. Deployed on Render for cloud access.

Features
Genre Selection: User selects genres using checkboxes for personalized recommendations.

Top-5 Recommendations: Shows top-5 movies similar to selected genres, powered by KNN algorithm.

Clean Interface: Built with Streamlit for quick interactivity and a simple, friendly UI.

Requirements
Python 3.8+

Libraries: streamlit, pandas, scikit-learn, etc.

Install dependencies:

bash
pip install -r requirements.txt
or auto-generate:

bash
pip install pipreqs
pipreqs --encoding=utf8
Usage
Clone the Repository:

bash
git clone https://github.com/your-username/movie-recommend-app.git
cd movie-recommend-app
Run Locally:

bash
streamlit run app.py
Deploy on Render:

Push your code to GitHub.

Login to Render and select “New Web Service.”

Connect your GitHub repo.

Set start command to: streamlit run app.py

Deploy and share your Render URL.

Project Structure
text
movie-recommend-app/
│
├── app.py                # Streamlit main app
├── requirements.txt      # Python dependencies
├── data/                 # Movie dataset (CSV)
└── README.md             # Project documentation
How it Works
Loads a movie dataset with boolean genre columns.

User selects genres via Streamlit interface.

Encodes genre selections; computes similarity using scikit-learn’s NearestNeighbors.

Displays top-5 recommended movies based on KNN scores.

Updating the App
Make changes locally, commit, and push to GitHub. Render redeploys automatically.

To update libraries, edit requirements.txt and redeploy.
