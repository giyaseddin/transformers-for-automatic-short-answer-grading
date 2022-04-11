import os
import sys
import flask
from flask import request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import utils

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# preload the model, and set ready
model = utils.load_best_model()


@app.route('/api/answer_evaluation')
def my_form():
    return render_template('qa-input-form.html')


@app.route('/api/answer_evaluation', methods=['POST'])
def home():
    query_parameters = request.form
    context = query_parameters['context']
    question = query_parameters['question']
    ref_answer = query_parameters['ref_answer']
    student_answer = query_parameters['student_answer']

    body = utils.pre_process_body(context, question, ref_answer, student_answer)

    pred = model([body])

    return jsonify(utils.format_results(pred)[0])


app.run()
