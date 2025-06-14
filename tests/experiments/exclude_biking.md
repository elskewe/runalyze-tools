Kurze Dokumentation wie man das exkludieren kann:

## Option 1

~~Gemini fragen wie ich Radkilometer aus Google Fit bekomme~~ => konnte mir nicht weiterhelfen

## Option 2

1. aggregierte Daten mit der `dataSourceId` `derived:com.google.activity.segment:com.google.android.gms:merge_activity_segments`
2. Rückgabe ist `com.google.activity.summary` mit den Werten `activity` (https://developers.google.com/fit/rest/v1/reference/activity-types), `duration` (in ms) und `num_segments`
3. Damit kann zumindest festgestellt werden, dass es an einem Tag was auf dem Fahrrad gab und darauf basierend der Nutzer nach der richtigen Distanz gefragt werden

## Option 3

Erkennung von unplausiblen Daten aus Schrittlänge mit `derived:com.google.step_count.delta:com.google.android.gms:estimated_steps` oder `merge_step_deltas`?

