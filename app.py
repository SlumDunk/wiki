import numpy as np
import scipy.sparse
import json
from sparsesvd import sparsesvd
from scipy.sparse.linalg import svds
from flask import Flask, render_template, request, redirect, Response
from sklearn.decomposition import TruncatedSVD
from flask_cors import *


app = Flask(__name__)
# allow cross web sites
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/output", methods=['POST'])
def output():
    # if the input json is not null
    if request.get_json() is not None:
        print(request.get_json());
        print(len(request.get_json()), len(request.get_json()[0]))
        if np.asarray(request.get_json()).T.shape[0] > 1000:
            svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
        else:
            svd = TruncatedSVD(n_components=np.asarray(request.get_json()).T.shape[0] - 1, n_iter=7, random_state=42)
        svd.fit(np.asarray(request.get_json()).T)
        return json.dumps(svd.components_.T.tolist())
    # if the input json is null, return empty json
    else:
        result = {}
        return json.dumps(result)


@app.route("/svds", methods=['POST'])
def function_svds():
    svd_input = np.empty((len(request.get_json()), len(request.get_json()[0])))
    for i in range(0, len(request.get_json())):
        svd_input[i] = request.get_json()[i]
    u, s, vh = svds(svd_input, k=min(svd_input.shape) - 1)
    return json.dumps(u.tolist())


@app.route("/sparsesvd", methods=['POST'])
def sparse_svd():
    svd_input = np.empty((len(request.get_json()), len(request.get_json()[0])))
    for i in range(0, len(request.get_json())):
        svd_input[i] = request.get_json()[i]
    smat = scipy.sparse.csc_matrix(svd_input)
    u, s, vh = sparsesvd(smat, min(smat.shape))
    return json.dumps(u.T.tolist())


if __name__ == "__main__":
    app.run()
    CORS(app, supports_credentials=True)
