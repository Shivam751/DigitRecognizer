import gradio as gr
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
sc = pickle.load(open("sc.pkl", "rb"))

def recognize_digit(draw_digit):

    draw_digit = np.reshape(draw_digit,(1,784))
    draw_digit = sc.transform(draw_digit)
    
    return model.predict(draw_digit)[0]


gr.Interface(fn=recognize_digit, inputs="sketchpad", outputs="label").launch(share=True)

