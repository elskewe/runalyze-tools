"""Contains functions to aid the creation of simple TCX files."""
import datetime


def create(date: datetime.datetime, duration: int, activity_type: str = "Other") -> str:
    """Creates a tcx from the activity duration.

    Args:
        date (datetime.datetime): The date of the activity.
        duration (int): The time in seconds.
        activity_type (str): The activity type.

    Returns:
        str: The content of the tcx file.
    """
    tcx_string = '<?xml version="1.0" encoding="UTF-8"?>\n'
    tcx_string += f'<TrainingCenterDatabase><Activities><Activity Sport="{activity_type}">\n'
    tcx_string += '<Id>' + date.strftime("%Y-%m-%dT%H:%M:%SZ") + '</Id>\n'
    tcx_string += '<Lap><TotalTimeSeconds>' + str(duration) + '</TotalTimeSeconds></Lap>\n'
    tcx_string += '</Activity></Activities></TrainingCenterDatabase>\n'
    return tcx_string


def create_from_distance_and_pace(date: datetime.datetime, distance: int, pace=10.0, activity_type="Other") -> str:
    """Creates a tcx from the distance and pace.

    Args:
        date (datetime.datetime): The date of the activity.
        distance (int): The distance in meters.
        pace (float): The pace in minutes per kilometer.
        activity_type (str): The activity type.

    Returns:
        str: The content of the tcx file.
    """
    return create(date, int(distance * (pace/1000) * 60), activity_type=activity_type)
