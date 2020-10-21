from flask import Flask, request,make_response,jsonify

import demo_image
import test
import predict
import threading




app = Flask(__name__)


@app.route('/uploadImg',methods=["POST"])
def hello_world():
    file = request.files.get("img")
    path = "./img/"
    img_name = file.filename
    file_path = path + img_name
    file.save(file_path)
    result=predict.predictImg(file_path)
    list = result.split("|")
    response = make_response(jsonify({"code": 200, "result": list[0], "confirm": list[1], "msg": "success"}))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with'
    return response
if __name__ == '__main__':
    def video():
        predict.star()


    def api():
        app.run(host="0.0.0.0", port=5000, debug=False)


    threads = []
    threads.append(threading.Thread(target=video))
    threads.append(threading.Thread(target=api))
    for t in threads:
        t.start()
