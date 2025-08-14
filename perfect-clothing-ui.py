from datetime import datetime
from functools import lru_cache
import gradio as gr
import pandas as pd
from perfect_clothing import predict, weather

MODEL, CANDIDATE_OUTFITS = predict.load_data()


@lru_cache
def predict_outfit(temperature: float, wind_speed_: float, cloud_cover_perc: float, duration_min: float, pace_min_km,
                   date_: datetime, is_race: bool, latitude_: float, longitude_: float):
    # Build your feature row from inputs
    features = pd.DataFrame([{
        "temperature": temperature,
        "wind_speed": wind_speed_,  # is already in km/h in Runalyze data
        "duration": duration_min*60,
        "x_pace": 60/pace_min_km,
        "ghi": weather.get_radiation(latitude_, longitude_, date_, cloud_cover_perc)[0],
        "is_race": is_race}])
    best_outfits = predict.recommend_best(features, MODEL, CANDIDATE_OUTFITS, top_k=100)
    return best_outfits, f"{int(pace_min_km)}:{round(pace_min_km*60%60):02d} min/km"


with gr.Blocks(fill_width=True) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            temp = gr.Slider(-10, 35, step=1, value=15, label="Temperature (°C)")
            wind_speed = gr.Slider(0, 30, step=1, value=0, label="Wind speed (km/h)")
            cloud_cover = gr.Slider(0, 100, step=10, value=0, label="Cloud cover (%)")
            with gr.Row():
                pace = gr.Slider(2.5, 10, step=1/60, value=4.5, label="Pace (min/km)", scale=6)
                pace_display = gr.Textbox(label="", min_width=120)
            duration = gr.Slider(0, 100, step=2.5, value=30, label="Duration (min)")
            with gr.Row():
                date = gr.DateTime(label="Date", value=datetime.now(), type="datetime")
                race = gr.Checkbox(label="Race")
            with gr.Row():
                latitude = gr.Number(label="Latitude", value=47.7664456)
                longitude = gr.Number(label="Longitude", value=9.1605106)
        with gr.Column(scale=2):
            output_df = gr.DataFrame()

    inputs = [temp, wind_speed, cloud_cover, duration, pace, date, race, latitude, longitude]
    change_args = {"fn": predict_outfit, "inputs": inputs, "outputs": [output_df, pace_display]}
    for i in inputs:
        i.change(**change_args)

demo.launch()
