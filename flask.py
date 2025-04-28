from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from mainscript import reflection

ans=reflection()
app = Flask(__name__)
api = Api(app)


class Square(Resource):

    def post(self):
        data = request.get_json()
        #print(data)
        question = data.get('prompt')
        #print(question)
        if question is None:
            return jsonify({'error': 'Missing "prompt" in the request body'}), 400
        answer=ans.main(question)
        print(answer)
        return jsonify({'answer': answer})



api.add_resource(Square, '/square')


# driver function
if __name__ == '__main__':

    app.run(debug = True)