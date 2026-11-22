"""Merge activity files into Runalyze using its browser-only merge workflow."""

import os
import re
from pathlib import Path

from playwright.sync_api import Locator, sync_playwright

MERGE_ALIGNMENT = "Date"
MERGE_FIELDS = {
    "Total duration": False,
    "Total distance": True,
    "GPS path": True,
    "Altitude": True,
    "Original pace": True,
    "Heart rate": False,
    "Cadence": False,
    "Power": False,
}
LOGIN_TIMEOUT = 5 * 60 * 1000
PAGE_TIMEOUT = 30 * 1000


def _browser_profile_directory() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data is None:
        local_app_data = str(Path.home() / "AppData" / "Local")
    return Path(local_app_data) / "runalyze-tools" / "playwright-profile"


def _merge_row(dialog: Locator, label: str) -> Locator:
    rows = dialog.get_by_role("row", name=re.compile(rf"^{re.escape(label)}\b"))
    if rows.count() != 1:
        raise RuntimeError(f"Runalyze merge dialog did not contain exactly one {label!r} row.")
    return rows


def _apply_merge_settings(dialog: Locator) -> None:
    date_row = _merge_row(dialog, MERGE_ALIGNMENT)
    date_radio = date_row.get_by_role("radio")
    if date_radio.count() != 1:
        raise RuntimeError("Runalyze merge dialog did not contain a Date alignment option.")
    date_radio.check()

    for label, merge_field in MERGE_FIELDS.items():
        checkbox = _merge_row(dialog, label).get_by_role("checkbox")
        if checkbox.count() != 1:
            raise RuntimeError(f"Runalyze merge dialog did not contain one checkbox for {label!r}.")
        checkbox.set_checked(merge_field)


def merge_activity_file(activity_id: int, fit_path: Path) -> None:
    if not fit_path.is_file():
        raise FileNotFoundError(f"Zwift FIT file does not exist: {fit_path}")

    profile_directory = _browser_profile_directory()
    profile_directory.mkdir(parents=True, exist_ok=True)
    merge_url = f"https://runalyze.com/activity/{activity_id}/merge"

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_directory),
            headless=False,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(merge_url, wait_until="domcontentloaded")
            login_heading = page.get_by_role(
                "heading",
                name="Welcome back to RUNALYZE!",
                exact=True,
            )
            if login_heading.count():
                print("Sign in to Runalyze in the opened browser; this browser profile will be reused.")
                login_heading.wait_for(state="hidden", timeout=LOGIN_TIMEOUT)
                page.goto(merge_url, wait_until="domcontentloaded")

            page.get_by_role(
                "heading",
                name="Merge trackdata from other file",
                exact=True,
            ).wait_for(timeout=PAGE_TIMEOUT)
            dropzones = page.get_by_text("Drop files here or click to upload", exact=True)
            if dropzones.count() != 2:
                raise RuntimeError("Runalyze merge dialog did not show the expected CSV and track-data upload areas.")
            with page.expect_file_chooser(timeout=PAGE_TIMEOUT) as chooser_info:
                dropzones.nth(1).click()
            chooser_info.value.set_files(str(fit_path))

            page.get_by_role(
                "heading",
                name="Merge data into activity",
                exact=True,
            ).wait_for(timeout=PAGE_TIMEOUT)
            dialog = page.get_by_role("dialog").last
            _apply_merge_settings(dialog)
            merge_button = dialog.get_by_role(
                "button",
                name="Merge data into activity",
                exact=True,
            )
            if merge_button.count() != 1:
                raise RuntimeError("Runalyze merge confirmation button was not available.")
            merge_button.click()
            page.wait_for_url(
                re.compile(rf"https://runalyze\.com/activity/{activity_id}/merge/.+"),
                timeout=PAGE_TIMEOUT,
            )
            # Runalyze redirects to the dashboard after merging, so the URL change is the success signal.
            page.wait_for_load_state("domcontentloaded", timeout=PAGE_TIMEOUT)
        finally:
            context.close()
