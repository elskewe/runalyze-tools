from datetime import datetime, timedelta
from functools import lru_cache

import gradio as gr
import numpy as np
import pandas as pd
from tzlocal import get_localzone

from perfect_clothing import assumptions, predict, weather

MODEL, CANDIDATE_OUTFITS = predict.load_data()
ALTERNATIVE_MODEL, _ = predict.load_data(assumptions.ALTERNATIVE_MODEL_FILENAME)


def filter_compression_socks(data: pd.DataFrame, include_compression_socks: bool) -> pd.DataFrame:
    """Removes outfits with compression socks from the data if `include_compression_socks` is False."""
    if not include_compression_socks:
        return data[~data["outfit"].str.contains(assumptions.CLOTHING_ABBREVIATIONS["Kompressionsstrümpfe"])]
    return data

def format_pace(pace_min_km: float):
    return f"{int(pace_min_km)}:{round(pace_min_km*60%60):02d} min/km"

@lru_cache
def predict_outfit(temperature: float, wind_speed_: float, cloud_cover_perc: float, duration_min: float, pace_min_km: float,
                   date_: datetime, is_race: bool, include_compression_socks: bool, latitude_: float, longitude_: float):
    ghi = weather.get_radiation(latitude_, longitude_,
                                [date_, date_ + timedelta(minutes=duration_min)/2,
                                 date_ + timedelta(minutes=duration_min)],
                                cloud_cover_perc)
    # Build your feature row from inputs
    features = pd.DataFrame([{
        "temperature": temperature,
        "wind_chill": weather.wind_chill(temperature, wind_speed_),
        "wind_speed": wind_speed_,  # is already in km/h in Runalyze data
        "duration": duration_min*60,
        "x_pace": 60/pace_min_km,
        "ghi_start": ghi[0],
        "ghi_middle": ghi[1],
        "ghi_end": ghi[2],
        "is_race": is_race}])
    print(features)
    best_outfits = predict.recommend_best(features, MODEL, CANDIDATE_OUTFITS, top_k=100)
    best_outfits_alternative = predict.recommend_best(features, ALTERNATIVE_MODEL, CANDIDATE_OUTFITS, top_k=100)
    return filter_compression_socks(best_outfits, include_compression_socks), \
           filter_compression_socks(best_outfits_alternative, include_compression_socks), \
           format_pace(pace_min_km), np.average(ghi)


with gr.Blocks(fill_width=True) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            temp = gr.Slider(-10, 35, step=1, value=15, label="Temperature (°C)")
            wind_speed = gr.Slider(0, 30, step=1, value=0, label="Wind speed (km/h)")
            cloud_cover = gr.Slider(0, 100, step=10, value=50, label="Cloud cover (%)")
            with gr.Row():
                pace = gr.Slider(2.5, 10, step=1/60, value=4.5, label="Pace (min/km)", scale=6)
                pace_display = gr.Textbox(label="", min_width=120)
            duration = gr.Slider(0, 100, step=2.5, value=30, label="Duration (min)")
            with gr.Row():
                date = gr.DateTime(label="Date", value=datetime.now(),
                                   # the timezone should be set automatically to the local timezone
                                   # when no value is passed, but this doesn't work unfortunately
                                   timezone=get_localzone().key, type="datetime", scale=2, min_width=200)
                with gr.Column(min_width=80):
                    race = gr.Checkbox(label="Race")
                    compression_socks = gr.Checkbox(label="Compression socks")
                ghi = gr.Number(label="GHI (W/m^2)", min_width=100)
            with gr.Row():
                latitude = gr.Number(label="Latitude", value=47.7664456)
                longitude = gr.Number(label="Longitude", value=9.1605106)
        with gr.Column(scale=7):
            output_df = gr.DataFrame(show_row_numbers=True)
            output_df_alternative = gr.DataFrame(show_row_numbers=True)

    inputs = [temp, wind_speed, cloud_cover, duration, pace, date, race, compression_socks, latitude, longitude]
    change_args = {"fn": predict_outfit, "inputs": inputs,
                   "outputs": [output_df, output_df_alternative, pace_display, ghi]}
    for i in inputs:
        i.change(**change_args)

demo.launch()
