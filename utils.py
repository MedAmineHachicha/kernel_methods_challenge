import numpy as np
import pandas as pd

def read_data(paths, delimiter, skip_header):
    data = []
    for path in paths:
        subdata = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)
        data.append(subdata)
    data = np.concatenate(data)
    return data

def export_predictions(X, model, filename):
    y_pred = model.predict(X)
    ids = np.array(range(3000))
    df = {'Id': ids,
          'Bound': y_pred.astype(int)}
    df = pd.DataFrame(df).set_index('Id')
    df.to_csv(filename)