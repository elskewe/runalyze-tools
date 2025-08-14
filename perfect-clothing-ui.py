import gradio as gr
import pandas as pd
from perfect_clothing import predict

MODEL, CANDIDATE_OUTFITS = predict.load_data()


def predict_outfit(temp, duration_min, is_race: bool):
    # Build your feature row from inputs
    features = pd.DataFrame([{"temperature": temp, "duration": duration_min*60, "is_race": is_race}])
    best_outfits = predict.recommend_best(features, MODEL, CANDIDATE_OUTFITS, top_k=100)
    return best_outfits


demo = gr.Interface(
    fn=predict_outfit,
    inputs=[
        gr.Slider(-10, 35, step=1, label="Temperature (°C)"),
        gr.Slider(0, 100, step=2.5, label="Duration (min)"),
        gr.Checkbox(label="Race")
    ],
    outputs="dataframe"
)

demo.launch()
