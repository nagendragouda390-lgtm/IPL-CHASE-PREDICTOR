"""
IPL Chase Predictor - Flask Application
-----------------------------------------------------------------------------
Loads a pre-trained RandomForestClassifier (model/model.pkl) and predicts
whether a chasing team can successfully chase down a target score.

Model expects features in this EXACT order (confirmed from the pickled
model's `feature_names_in_`):

    [batting_team, bowling_team, target, curr_run, curr_wick, ball_number,
     cr, req_run, balls_left, wick_left, rr, rrpw, rpw]

ASSUMPTIONS (the original training notebook was not provided, only the
.pkl file -- update these if your predictions look off):

  1. batting_team / bowling_team are encoded as the float values supplied
     for this project (see TEAM_TO_CODE below) -- e.g. Rajasthan Royals
     = 1.0, Kolkata Knight Riders = 0.1, etc.

  2. cr  = current run rate  (curr_run * 6 / ball_number)
     rr  = required run rate (req_run * 6 / balls_left)
     rpw = runs scored per wicket lost so far  (curr_run / curr_wick)
     rrpw= required runs needed per wicket remaining (req_run / wick_left)
-----------------------------------------------------------------------------
"""

from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__),"model.pkl")
model = joblib.load(MODEL_PATH)

# Exact column order the model was trained on
FEATURE_ORDER = [
    "batting_team", "bowling_team", "target", "curr_run", "curr_wick",
    "ball_number", "cr", "req_run", "balls_left", "wick_left", "rr",
    "rrpw", "rpw",
]

# Team -> encoded value mapping (as provided).
TEAM_TO_CODE = {
    "Rajasthan Royals": 1.0,
    "Royal Challengers Bengaluru": 0.9,
    "Gujarat Titans": 0.8,
    "Punjab Kings": 0.7,
    "Mumbai Indians": 0.6,
    "Sunrisers Hyderabad": 0.5,
    "Lucknow Super Giants": 0.4,
    "Chennai Super Kings": 0.3,
    "Delhi Capitals": 0.2,
    "Kolkata Knight Riders": 0.1,
}
# Dropdown order follows the same order
TEAMS = list(TEAM_TO_CODE.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def overs_to_balls(overs_str: str) -> int:
    """
    Convert cricket overs notation (e.g. 7.4 = 7 overs + 4 balls) into a
    total ball count. NOTE: the decimal part represents BALLS (0-5), not a
    true decimal fraction.
        7.4 overs -> 7*6 + 4 = 46 balls
        0.2 overs -> 0*6 + 2 = 2 balls
    """
    overs_str = str(overs_str).strip()
    if "." in overs_str:
        whole_part, ball_part = overs_str.split(".", 1)
        whole = int(whole_part) if whole_part else 0
        # only the first digit after the decimal point is meaningful (0-5)
        balls = int(ball_part[0]) if ball_part else 0
    else:
        whole = int(overs_str)
        balls = 0

    if balls > 5:
        balls = 5  # safety clamp - a valid over only has 6 balls (0-5)

    return whole * 6 + balls


def safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of raising on divide-by-zero."""
    if denominator is None or denominator == 0:
        return 0.0
    return numerator / denominator


def get_verdict(probability: float):
    """
    Map a chase-success probability (0-1) to a verdict message + color band,
    per the project's prediction-logic spec.
    """
    pct = probability * 100
    if probability > 0.85:
        return "Easy chase for batting team", "green", pct
    elif probability >= 0.50:
        return "Possible but needs good batting", "yellow", pct
    elif probability >= 0.20:
        return "Difficult chase", "orange", pct
    else:
        return "Almost impossible chase", "red", pct


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", teams=TEAMS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        batting_team = request.form.get("batting_team")
        bowling_team = request.form.get("bowling_team")
        target = int(request.form.get("target", 0))
        curr_run = int(request.form.get("curr_run", 0))
        curr_wicket = int(request.form.get("curr_wicket", 0))
        overs_input = request.form.get("overs", "0.0")

        # ---- Input validation ----------------------------------------------
        if batting_team == bowling_team:
            return render_template(
                "result.html",
                r=None,
                error="Batting team and bowling team cannot be the same. Please select two different teams.",
            )

        if curr_run > target:
            return render_template(
                "result.html",
                r=None,
                error=f"Current score ({curr_run}) cannot be greater than the target ({target}).",
            )

        # ---- Core calculations (per project spec) -------------------------
        ball_number = overs_to_balls(overs_input)
        ball_number = max(0, min(ball_number, 120))  # clamp to a legal innings
        curr_wicket = max(0, min(curr_wicket, 10))

        curr_rr = safe_div(curr_run * 6, ball_number)          # current run rate
        req_run = max(target - curr_run, 0)                     # required runs
        balls_left = max(120 - ball_number, 0)                  # balls left
        wicket_left = max(10 - curr_wicket, 0)                  # wickets left
        req_rr = safe_div(req_run * 6, balls_left)               # required run rate

        rpw = safe_div(curr_run, curr_wicket) if curr_wicket > 0 else float(curr_run)
        rrpw = safe_div(req_run, wicket_left) if wicket_left > 0 else float(req_run)

        # ---- Hard edge cases that override the model ----------------------
        already_won = req_run <= 0
        all_out = wicket_left <= 0 and req_run > 0
        out_of_balls = balls_left <= 0 and req_run > 0

        if already_won:
            probability = 1.0
        elif all_out or out_of_balls:
            probability = 0.0
        else:
            batting_code = TEAM_TO_CODE.get(batting_team, 0)
            bowling_code = TEAM_TO_CODE.get(bowling_team, 0)

            features = pd.DataFrame(
                [[
                    batting_code, bowling_code, target, curr_run, curr_wicket,
                    ball_number, curr_rr, req_run, balls_left, wicket_left,
                    req_rr, rrpw, rpw,
                ]],
                columns=FEATURE_ORDER,
            )
            probability = float(model.predict_proba(features)[0][1])

        message, color, pct = get_verdict(probability)

        if already_won:
            message, color = "Target already achieved!", "green"
        elif all_out:
            message, color = "All out - cannot chase", "red"
        elif out_of_balls:
            message, color = "Out of balls - cannot chase", "red"

        result = {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "target": target,
            "curr_run": curr_run,
            "curr_wicket": curr_wicket,
            "overs_input": overs_input,
            "probability": round(pct, 1),
            "message": message,
            "color": color,
            "curr_rr": round(curr_rr, 2),
            "req_rr": round(req_rr, 2),
            "req_run": req_run,
            "balls_left": balls_left,
            "wicket_left": wicket_left,
        }
        return render_template("result.html", r=result)

    except Exception as exc:
        return render_template(
            "result.html",
            r=None,
            error=f"Something went wrong while predicting: {exc}",
        )


if __name__ == "__main__":
    app.run(debug=True)
