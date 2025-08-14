import gradio as gr
import pandas as pd
from perfect_clothing import predict

MODEL, CANDIDATE_OUTFITS = predict.load_data()


def predict_outfit(temp, duration_min, is_race: bool):
    # Build your feature row from inputs
    features = pd.DataFrame([{"temperature": temp, "duration": duration_min*60, "is_race": is_race}])
    best_outfits = predict.recommend_best(features, MODEL, CANDIDATE_OUTFITS, top_k=100)
    return best_outfits


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            temp = gr.Slider(-10, 35, step=1, value=15, label="Temperature (°C)")
            duration = gr.Slider(0, 100, step=2.5, value=30, label="Duration (min)")
            race = gr.Checkbox(label="Race")
        with gr.Column(scale=2):
            output_df = gr.DataFrame()

    inputs = [temp, duration, race]
    change_args = {"fn": predict_outfit, "inputs": inputs, "outputs": output_df}
    for i in inputs:
        i.change(**change_args)

demo.launch()
