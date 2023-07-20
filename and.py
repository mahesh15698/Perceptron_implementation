import pandas as pd
from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import os
import logging

gate = "AND gate"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "running_logs.log"),
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode='a'
    )

def main(data,modelName,plotName,eta,epochs):
    df_AND = pd.DataFrame(data)
    X, y = prepare_data(df_AND)
    model_and = Perceptron(eta=eta, epochs=epochs)
    model_and.fit(X, y)
    _ = model_and.total_loss()
    model_and.save(filename=modelName,model_dir="model")
    save_plot(df_AND, model_and, filename=plotName)
    
if __name__ == "__main__":
    AND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [0,0,0,1]
    }
    ETA = 0.1 # 0 and 1
    EPOCHS = 10
    try:
        logging.info(f">>>>> starting training for {gate}>>>>>")
        main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)
        logging.info(f"<<<<< done training for {gate} <<<<<\n\n")
    except Exception as e:
        logging.exception(e)
        raise e

