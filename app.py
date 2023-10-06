from flask import Flask, request
import pandas as pd
import numpy as np
import skfuzzy as skfuzz
from fuzzywuzzy import fuzz
from flask_cors import CORS

df = pd.read_csv('udemy_courses.csv')

min_price = df.price.min()
max_price = df.price.max()

min_duration = df.content_duration.min()
max_duration = df.content_duration.max()

price_range = np.arange(min_price, max_price + 1, 1)
low_price = skfuzz.trimf(price_range, [min_price, min_price, max_price * 0.3])
medium_price = skfuzz.trimf(price_range, [min_price, max_price * 0.3, max_price * 0.7])
high_price = skfuzz.trimf(price_range, [max_price * 0.3, max_price * 0.7, max_price])

duration_range = np.arange(min_duration, max_duration + 1, 1)
short_duration = skfuzz.trimf(duration_range, [min_duration, min_duration, max_duration * 0.3])
medium_duration = skfuzz.trimf(duration_range, [min_duration, max_duration * 0.3, max_duration * 0.7])
long_duration = skfuzz.trimf(duration_range, [max_duration * 0.3, max_duration * 0.7, max_duration])

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# recommend route takes in title, price, and duration as post request body
@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        data = request.get_json()

        input_title = data['title']
        input_price = data['price']
        input_duration = data['duration']

        price_membership = {
            "Low": skfuzz.interp_membership(price_range, low_price, input_price),
            "Medium": skfuzz.interp_membership(price_range, medium_price, input_price),
            "High": skfuzz.interp_membership(price_range, high_price, input_price)
        }

        duration_membership = {
            "Short": skfuzz.interp_membership(duration_range, short_duration, input_duration),
            "Long": skfuzz.interp_membership(duration_range, medium_duration, input_duration),
            "High": skfuzz.interp_membership(duration_range, long_duration, input_duration)
        }

        highest_duration_membership = max(duration_membership.values())
        highest_price_membership = max(price_membership.values())

        filtered_courses = df[df['content_duration'].apply(
            lambda x: max(duration_membership.values()) == highest_duration_membership
        )]

        filtered_courses = df[df['price'].apply(
            lambda x: max(price_membership.values()) == highest_price_membership
        )]

        # Calculate title similarity scores against all courses in the DataFrame
        filtered_courses['title_similarity'] = filtered_courses['course_title'].apply(
            lambda x: fuzz.ratio(input_title, x) / 100.0
        )

        top_similar_courses = filtered_courses.sort_values(by='title_similarity', ascending=False).head(5)

        return top_similar_courses[['course_title', 'url', 'price', 'content_duration', 'level', 'subject']].to_json()
    

if __name__ == '__main__':
    app.run(debug=True)
