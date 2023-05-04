from flask import Flask, render_template, request

from src.models.optimizer import Optimizer

application = Flask(__name__)

app = application


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = Optimizer(
            chromasity=float(request.form.get('chromasity')),
            feculence=float(request.form.get('feculence')),
            ph=float(request.form.get('ph')),
            mn=float(request.form.get('mn')),
            fe=float(request.form.get('fe')),
            alkalinity=float(request.form.get('alkalinity')),
            nh4=float(request.form.get('nh4')),
            lime=float(request.form.get('lime')),
            PAA_kk=float(request.form.get('PAA_kk')),
            PAA_f=float(request.form.get('PAA_f')),
            sa=float(request.form.get('sa')),
            permanganate=float(request.form.get('permanganate'))

        )
        pred_df = data.get_weights_and_features()
        print(pred_df)
        print("Before Prediction")

        results_ = data.predict(pred_df)
        print("Mid Prediction")
        print("after Prediction: ", results_)
        return render_template('home.html', results=results_)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
