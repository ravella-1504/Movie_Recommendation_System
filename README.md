# Movie_Recommendation_System
This project is a movie recommendation system that suggests similar movies based on user input. It uses the concept of cosine similarity to calculate the similarity between movies and provides recommendations accordingly.

**Prerequisites**
Before running the code, make sure you have the following dependencies installed:
NumPy
pandas
scikit-learn

**You can install them using pip:**
pip install numpy pandas scikit-learn

Getting Started
**Clone the repository:**
git clone https://github.com/ravella-1504/Movie_Recommendation_System.git

**Change into the project directory:**
cd Movie-Recommendation-System

**Prepare the data:**

Download the movie dataset in CSV format.
Place the CSV file in the project directory.

**Run the code:**
python movie_recommendation.py
Enter your favorite movie name when prompted.

The system will provide a list of similar movies based on your input.

**Understanding the Code**
The code uses the pandas library to read the movie dataset from a CSV file and preprocess the data.
It combines selected features like genres, keywords, tagline, cast, and director into a single text representation for each movie.
The text data is converted into feature vectors using the TfidfVectorizer from scikit-learn.
Cosine similarity is calculated between all pairs of movies using the feature vectors.
The user enters their favorite movie, and the code finds the closest match in the dataset using the difflib library.
Recommendations are provided based on the similarity scores of the matched movie.
**Dataset**
The dataset used in this project should be a CSV file containing movie information. The file should include columns for 'title', 'genres', 'keywords', 'tagline', 'cast', and 'director'. Make sure the CSV file is formatted correctly and contains relevant data.

**Limitations and Future Improvements**
The current implementation uses cosine similarity and a simple text-based representation of movies. It may not capture all aspects of movie similarity.
The dataset used for recommendations plays a significant role. Expanding the dataset with more movies and additional features can improve the quality of recommendations.
The code can be further optimized for performance and efficiency.

**License**
This project is licensed under the MIT License - see the LICENSE file for details.

**Acknowledgments**
The project is inspired by the concept of content-based recommendation systems.
Thanks to the contributors of numpy, pandas, and scikit-learn for their open-source libraries.
