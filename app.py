from flask import Flask, render_template, request
import index
import numpy as np
app = Flask(__name__)



@app.route('/')
def results():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def process_res():
    user_search_query = request.form['msg']
    corpus, ranking_function, cache_dict= index.corpus_index()
    date_list = list(cache_dict.keys())
    summary_list = list(cache_dict.values())
    tokenized_query = index.remove_stopwords(user_search_query.lower().split(" "))
    scores = ranking_function.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:10]
    result_list = []
    for i in top_n:
        if scores[i] !=0:
            result_list.append(corpus[i])

    return render_template('index.html', search_results_list = result_list,
                                          user_query=user_search_query,
                                          date_list=date_list,
                                          summary_list=summary_list)






if __name__ == "__main__":
    app.run(debug=True)
