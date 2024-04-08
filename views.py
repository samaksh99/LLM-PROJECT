from flask import Flask, jsonify, request
from model import Solution
import time
import json
from validate_results import ValidateResultsInteractor

from rich.console import Console

console = Console()

app = Flask(__name__)
validate_text_interactor = ValidateResultsInteractor()
Solution_interactor= Solution()

# Route to handle POST requests with chunk_id input
@app.route("/req", methods=["POST"])
def req():
    print("API called again")
    data = request.json
    print(data)
    sentence_value = data["sentence"]
    print("Sentence:", sentence_value)
    similar_chunk = Solution()
    start = time.time()
    answer = similar_chunk.final_result(query=sentence_value)
    end = time.time()
    bot_response_dict = {
        "query": answer['query'],
        "result": answer['result'],
        "time_taken": round(end-start, 1)
    }
    bot_response_dict1 = {
        "query": answer['query'],
        "result": "Could please Redefine your Question",
        "time_taken": round(end-start, 1)
    }
    bot_response_dict2 = {
        "query": answer['query'],
        "result": "Please Ask A relevant Query",
        "time_taken": round(end-start, 1)
    }
    score = validate_text_interactor.validate_results(bot_response_dict['result'], sentence_value)
    similar_text= Solution_interactor.semantic_search(sentence_value)
    console.print(similar_text)
    print(score)
    print(bot_response_dict["time_taken"])
    print(bot_response_dict["result"])

    if score > 0.04:
        return bot_response_dict
    elif score>0.03 and score<0.04:
        return bot_response_dict1
    else:
        return bot_response_dict2

if __name__ == "__main__":
    app.run(debug=True,port=3300)
