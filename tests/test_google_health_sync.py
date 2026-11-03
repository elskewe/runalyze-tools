import datetime
import importlib

google_sync = importlib.import_module("google-fit-to-runalyze")


class FakeRequest:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class FakeHealthService:
    def __init__(self, daily_pages=None, exercise_pages=None):
        self.daily_pages = iter(daily_pages or [])
        self.exercise_pages = iter(exercise_pages or [])
        self.daily_requests = []
        self.exercise_requests = []

    def users(self):
        return self

    def dataTypes(self):
        return self

    def dataPoints(self):
        return self

    def dailyRollUp(self, parent, body):
        self.daily_requests.append((parent, body.copy()))
        return FakeRequest(next(self.daily_pages))

    def list(self, parent, **params):
        self.exercise_requests.append((parent, params))
        return FakeRequest(next(self.exercise_pages))


def test_activity_datetime_uses_noon_utc_on_requested_date():
    date = datetime.date(2026, 11, 1)

    activity_datetime = google_sync._activity_datetime_for_date(date)
    tcx = google_sync.create_from_distance_and_pace(activity_datetime, 1000)

    assert activity_datetime == datetime.datetime(
        2026, 11, 1, 12, tzinfo=datetime.timezone.utc
    )
    assert "<Id>2026-11-01T12:00:00Z</Id>" in tcx


def test_extract_distance_data_converts_rollup_millimeters_to_meters():
    rollups = [
        {
            "civilStartTime": {"date": {"year": 2026, "month": 11, "day": 1}},
            "distance": {"millimetersSum": "1234567"},
        },
        {
            "civilStartTime": {"date": {"year": 2026, "month": 11, "day": 2}},
            "distance": {"millimetersSum": "0"},
        },
        {"civilStartTime": {"date": {"year": 2026, "month": 11, "day": 3}}},
    ]

    assert google_sync.extract_distance_data(rollups) == {
        datetime.date(2026, 11, 1): 1234,
        datetime.date(2026, 11, 2): 0,
    }


def test_extract_activity_data_filters_types_and_sums_durations():
    data_points = [
        {
            "exercise": {
                "exerciseType": "WALKING",
                "interval": {
                    "civilStartTime": {"date": {"year": 2026, "month": 11, "day": 1}},
                    "startTime": "2026-11-01T10:00:00Z",
                    "endTime": "2026-11-01T10:10:00Z",
                },
                "activeDuration": "600s",
            }
        },
        {
            "exercise": {
                "exerciseType": "RUNNING",
                "interval": {
                    "civilStartTime": {"date": {"year": 2026, "month": 11, "day": 1}},
                    "startTime": "2026-11-01T10:10:00Z",
                    "endTime": "2026-11-01T10:20:00Z",
                },
            }
        },
        {
            "exercise": {
                "exerciseType": "BIKING",
                "interval": {
                    "civilStartTime": {"date": {"year": 2026, "month": 11, "day": 1}},
                    "startTime": "2026-11-01T11:00:00Z",
                    "endTime": "2026-11-01T11:30:00Z",
                },
                "activeDuration": "1800s",
            }
        },
    ]

    assert google_sync.extract_activity_data(
        data_points, google_sync.WALKING_ACTIVITY_TYPES
    ) == {datetime.date(2026, 11, 1): {"duration": 1_200_000, "num_segments": 2}}
    assert google_sync.extract_activity_data(
        data_points, google_sync.BIKING_ACTIVITY_TYPES
    ) == {datetime.date(2026, 11, 1): {"duration": 1_800_000, "num_segments": 1}}


def test_daily_rollup_chunks_ranges_and_follows_page_tokens():
    page_with_token = {"rollupDataPoints": [], "nextPageToken": "next"}
    last_page = {"rollupDataPoints": []}
    health_service = FakeHealthService(
        daily_pages=[page_with_token, last_page, page_with_token, last_page]
    )
    google_sync._request_daily_rollup(
        health_service,
        datetime.date(2026, 1, 1),
        datetime.date(2026, 4, 2),
    )

    assert len(health_service.daily_requests) == 4
    assert [request[1]["range"] for request in health_service.daily_requests[::2]] == [
        {
            "start": {"date": {"year": 2026, "month": 1, "day": 1}},
            "end": {"date": {"year": 2026, "month": 4, "day": 1}},
        },
        {
            "start": {"date": {"year": 2026, "month": 4, "day": 1}},
            "end": {"date": {"year": 2026, "month": 4, "day": 2}},
        },
    ]
    assert all(
        parent == google_sync.GOOGLE_HEALTH_DISTANCE_PARENT
        for parent, _ in health_service.daily_requests
    )
    assert health_service.daily_requests[1][1]["pageToken"] == "next"
    assert health_service.daily_requests[3][1]["pageToken"] == "next"


def test_get_exercise_data_uses_generated_list_method_and_paginates():
    health_service = FakeHealthService(
        exercise_pages=[
            {"dataPoints": [{"exercise": {"exerciseType": "WALKING"}}], "nextPageToken": "next"},
            {"dataPoints": [{"exercise": {"exerciseType": "RUNNING"}}]},
        ]
    )

    result = google_sync.get_exercise_data(
        health_service,
        datetime.date(2026, 11, 1),
        datetime.date(2026, 11, 2),
    )

    assert len(result) == 2
    assert [parent for parent, _ in health_service.exercise_requests] == [
        google_sync.GOOGLE_HEALTH_EXERCISE_PARENT,
        google_sync.GOOGLE_HEALTH_EXERCISE_PARENT,
    ]
    assert health_service.exercise_requests[0][1] == {
        "filter": 'exercise.interval.civil_start_time >= "2026-11-01" AND '
        'exercise.interval.civil_start_time < "2026-11-02"',
        "pageSize": google_sync.EXERCISE_PAGE_SIZE,
        "pageToken": None,
    }
    assert health_service.exercise_requests[1][1]["pageToken"] == "next"
