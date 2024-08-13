from flask import Flask
from flask import render_template
from flask import request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
pipe = pickle.load(open('model/pipe.pkl', 'rb'))

# Define list of teams and cities
teams = sorted([
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
])

cities = sorted([
    'Hyderabad',
    'Bangalore',
    'Mumbai',
    'Indore',
    'Kolkata',
    'Delhi',
    'Chandigarh',
    'Jaipur',
    'Chennai',
    'Cape Town',
    'Port Elizabeth',
    'Durban',
    'Centurion',
    'East London',
    'Johannesburg',
    'Kimberley',
    'Bloemfontein',
    'Ahmedabad',
    'Cuttack',
    'Nagpur',
    'Dharamsala',
    'Visakhapatnam',
    'Pune',
    'Raipur',
    'Ranchi',
    'Abu Dhabi',
    'Sharjah',
    'Mohali',
    'Bengaluru'
])

# Define route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        batting_team = request.form.get('batting_team')
        bowling_team = request.form.get('bowling_team')
        selected_city = request.form.get('city')
        target = int(request.form.get('target', 0))
        score = int(request.form.get('score', 0))
        wickets = int(request.form.get('wickets', 0))
        overs = float(request.form.get('overs', 0))

        runs_left = target - score
        balls_left = 120 - overs * 6
        wickets_left = 10 - wickets
        crr = score / overs if overs != 0 else 0
        rrr = runs_left * 6 / balls_left if balls_left != 0 else 0

        # Create DataFrame for prediction
        df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict probabilities
        result = pipe.predict_proba(df)
        r_1 = round(result[0][0] * 100, 2)
        r_2 = round(result[0][1] * 100, 2)

        return render_template('index.html',
                               teams=teams,
                               cities=cities,
                               batting_team=batting_team,
                               bowling_team=bowling_team,
                               selected_city=selected_city,
                               target=target,
                               score=score,
                               wickets=wickets,
                               overs=overs,
                               r_1=r_1,
                               r_2=r_2)

    return render_template('index.html',
                           teams=teams,
                           cities=cities,
                           selected_city=None,
                           target=None,
                           score=None,
                           wickets=None,
                           overs=None,
                           r_1=None,
                           r_2=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)