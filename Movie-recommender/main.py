from flask import Flask, render_template, request
import pandas as pd
import subprocess

app = Flask(__name__, template_folder='template')


# Load the movies.csv file using pandas
movies_df = pd.read_csv('Data/movies.csv')


@app.route('/', methods=['GET', 'POST'])
def home():
    # Get the sixth row of the movies DataFrame
    row = movies_df.iloc[:, 6]
    # Convert the row to a dictionary
    data = row.to_dict()
    if request.method == 'POST':
        # Get the column name that was clicked
        print(request.form)
        column = request.form['movie']
        # Call the test.py script and pass the column name as an argument
        result = subprocess.check_output(['python3', 'recommender.py', column])
        # Convert the result to a string
        result_str = result.decode('utf-8')
        print(result_str)
        return render_template('result.html', data=data, result=result_str)
    else:
        return render_template('index.html', data=data)


if __name__ == '__main__':
    app.run(debug=True)
