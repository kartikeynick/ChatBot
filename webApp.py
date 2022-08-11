from flask import Flask, render_template, jsonify, request

from chat import reply

wApp = Flask(__name__)


# now we define 2 route to render

# also decorate it
@wApp.get("/")#route("/", methods=['GET'])
def index_get():
    return render_template("base.html")


# making a post request
@wApp.post("/predict")
def predict():
    txt = request.get_json().get("message")
    # to do some error checking
    resp = reply(txt)  # to get a response
    msg = {"answer": resp}
    return jsonify(msg)


# done with the flask app

if __name__ == "__main__":
    wApp.run(debug=True)  # for testing
