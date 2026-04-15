#!/usr/bin/env python3
"""
13-Run Pool Tracker
Fetches 2026 MLB game data and generates a self-contained HTML tracker.
Run anytime to refresh: python mlb_pool_tracker.py
"""

import requests
import random
from datetime import date, timedelta
from collections import defaultdict

SEASON_START = date(2026, 3, 25)
TODAY = date.today()
SLOTS = list(range(14))  # 0 through 13
N_SIMS = 10000
MAX_SIM_GAMES = 600   # cap at ~4 seasons — race ends whenever a team finishes
OUTPUT_FILE = "index.html"  # GitHub Pages serves index.html at the root URL

# Historical MLB run distribution (based on 2015-2024 league averages).
# Used as a Bayesian prior — gives realistic probabilities for rare high-run games
# even when early-season sample sizes are small.
HISTORICAL_PRIOR = {
    0: 0.085, 1: 0.102, 2: 0.128, 3: 0.142, 4: 0.138,
    5: 0.120, 6: 0.088, 7: 0.065, 8: 0.045, 9: 0.030,
    10: 0.018, 11: 0.012, 12: 0.007, 13: 0.005,
    14: 0.015,   # 14+ runs — doesn't fill any slot
}


# ------------------------------------------------------------------
# 1. DATA FETCHING
# ------------------------------------------------------------------

def fetch_season_data():
    print("Fetching game data...", flush=True)
    game_log = []
    team_run_dist = defaultdict(list)

    d = SEASON_START
    while d <= TODAY:
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": d.isoformat()}
        ).json()
        for date_entry in r.get("dates", []):
            for game in date_entry.get("games", []):
                if game["status"]["detailedState"] not in ("Final", "Game Over", "Completed Early"):
                    continue
                gk = game["gamePk"]
                away = game["teams"]["away"]["team"]["name"]
                home = game["teams"]["home"]["team"]["name"]
                ls = requests.get(
                    f"https://statsapi.mlb.com/api/v1/game/{gk}/linescore"
                ).json()
                t = ls.get("teams", {})
                ar = t.get("away", {}).get("runs")
                hr = t.get("home", {}).get("runs")
                if ar is None or hr is None:
                    continue
                game_log.append((d, away, ar, home, hr))
                team_run_dist[away].append(ar)
                team_run_dist[home].append(hr)
        d += timedelta(days=1)
        print(f"  {d - timedelta(days=1)}: done", end="\r", flush=True)

    print(f"\nLoaded {len(game_log)} completed games across {len(team_run_dist)} teams.")
    return game_log, dict(team_run_dist)


# ------------------------------------------------------------------
# 2. TRACKER STATE
# ------------------------------------------------------------------

def build_tracker(game_log):
    team_runs = defaultdict(set)
    first_to_score = {}

    for gdate, away, ar, home, hr in game_log:
        for team, runs, opp in [(away, ar, home), (home, hr, away)]:
            if 0 <= runs <= 13:
                team_runs[team].add(runs)
            if 0 <= runs <= 13 and runs not in first_to_score:
                first_to_score[runs] = (team, gdate, opp)

    return dict(team_runs), first_to_score


# ------------------------------------------------------------------
# 3. RUN DISTRIBUTION
# Blend: historical prior (heavy weight early season) + team's 2026 data
# The prior encodes that 13 runs is ~0.5% likely, 4 runs is ~14% likely, etc.
# ------------------------------------------------------------------

def build_run_distributions(team_run_dist):
    # 2026 league average from observed data
    all_runs = [r for runs in team_run_dist.values() for r in runs]
    league_obs = defaultdict(float)
    for r in all_runs:
        league_obs[r] += 1 / len(all_runs)

    # Blend observed league data with historical prior
    # At 500+ league games the observed data dominates; early season the prior dominates
    league_weight = min(len(all_runs), 500) / 500
    league_dist = {}
    for r in set(list(league_obs.keys()) + list(HISTORICAL_PRIOR.keys())):
        league_dist[r] = league_weight * league_obs.get(r, 0) + (1 - league_weight) * HISTORICAL_PRIOR.get(r, 0)

    team_dist = {}
    for team, runs in team_run_dist.items():
        n = len(runs)
        # Require a full season (162 games) before trusting team data over the prior.
        # At 25 games, team data only has 15% weight — prevents small samples from
        # wildly inflating/deflating probabilities for rare run totals like 12 or 13.
        team_weight = min(n, 162) / 162
        td = defaultdict(float)
        for r in runs:
            td[r] += 1 / n
        blended = {}
        for r in set(list(td.keys()) + list(league_dist.keys())):
            blended[r] = team_weight * td.get(r, 0) + (1 - team_weight) * league_dist.get(r, 0)
        team_dist[team] = blended

    return team_dist, league_dist


# ------------------------------------------------------------------
# 4. MONTE CARLO SIMULATION
# The race runs until a team fills all 14 slots (no season cap).
# Win probability = fraction of simulations where each team finishes first.
# ------------------------------------------------------------------

def simulate_pool(team_runs, team_dist, all_teams, n_sims=N_SIMS, max_games=MAX_SIM_GAMES):
    print(f"Running {n_sims:,} simulations (up to {max_games} games/team)...", flush=True)

    # Pre-build CDF per team for fast sampling
    team_cdf = {}
    for team in all_teams:
        dist = team_dist.get(team, HISTORICAL_PRIOR)
        sorted_items = sorted(dist.items())
        cdf, cumulative = [], 0.0
        for r, p in sorted_items:
            cumulative += p
            cdf.append((cumulative, r))
        team_cdf[team] = cdf

    def sample(team):
        cdf = team_cdf[team]
        x = random.random()
        for prob, r in cdf:
            if x <= prob:
                return r
        return cdf[-1][1]

    win_count = defaultdict(int)
    finish_games = defaultdict(list)

    for _ in range(n_sims):
        missing = {t: set(SLOTS) - team_runs.get(t, set()) for t in all_teams}
        completion = {}

        for g in range(1, max_games + 1):
            for team in all_teams:
                if team in completion:
                    continue
                r = sample(team)
                if r in missing[team]:
                    missing[team].discard(r)
                    if not missing[team]:
                        completion[team] = g

            if completion:
                # End simulation once the first finisher is found
                # (track all who finish this same "game")
                break

        if completion:
            min_game = min(completion.values())
            # Everyone who finishes at the same game number ties for the win
            winners = [t for t, g in completion.items() if g == min_game]
            for w in winners:
                win_count[w] += 1 / len(winners)

        for t, g in completion.items():
            finish_games[t].append(g)

    total_wins = sum(win_count.values()) or 1
    win_pct = {t: win_count.get(t, 0) / total_wins * 100 for t in all_teams}
    avg_games_to_finish = {
        t: sum(finish_games[t]) / len(finish_games[t]) if finish_games[t] else None
        for t in all_teams
    }
    return win_pct, avg_games_to_finish


# ------------------------------------------------------------------
# 5. HTML GENERATION
# ------------------------------------------------------------------

def team_abbrev(name):
    abbrevs = {
        "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL", "Detroit Tigers": "DET",
        "Houston Astros": "HOU", "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN", "New York Mets": "NYM",
        "New York Yankees": "NYY", "Athletics": "ATH",
        "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD", "San Francisco Giants": "SF",
        "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
    }
    return abbrevs.get(name, name[:3].upper())

TEAM_COLORS = {
    "Arizona Diamondbacks": "#A71930", "Atlanta Braves": "#CE1141",
    "Baltimore Orioles": "#DF4601", "Boston Red Sox": "#BD3039",
    "Chicago Cubs": "#0E3386", "Chicago White Sox": "#27251F",
    "Cincinnati Reds": "#C6011F", "Cleveland Guardians": "#E31937",
    "Colorado Rockies": "#333366", "Detroit Tigers": "#0C2340",
    "Houston Astros": "#002D62", "Kansas City Royals": "#004687",
    "Los Angeles Angels": "#BA0021", "Los Angeles Dodgers": "#005A9C",
    "Miami Marlins": "#00A3E0", "Milwaukee Brewers": "#12284B",
    "Minnesota Twins": "#002B5C", "New York Mets": "#002D72",
    "New York Yankees": "#003087", "Athletics": "#003831",
    "Philadelphia Phillies": "#E81828", "Pittsburgh Pirates": "#27251F",
    "San Diego Padres": "#2F241D", "San Francisco Giants": "#FD5A1E",
    "Seattle Mariners": "#0C2C56", "St. Louis Cardinals": "#C41E3A",
    "Tampa Bay Rays": "#092C5C", "Texas Rangers": "#003278",
    "Toronto Blue Jays": "#134A8E", "Washington Nationals": "#AB0003",
}


def generate_html(team_runs, first_to_score, win_pct, avg_games, team_run_dist, game_log):
    all_teams = sorted(team_runs.keys())

    # Sort by win probability (primary), then slots filled (tiebreak)
    sorted_teams = sorted(
        all_teams,
        key=lambda t: (-win_pct.get(t, 0), -len(team_runs.get(t, set())))
    )

    total_games = len(game_log)
    max_win_pct = max(win_pct.values()) if win_pct else 1
    claimed_count = sum(1 for s in SLOTS if s in first_to_score)

    # Slot difficulty annotation using historical prior
    slot_difficulty = {}
    for s in SLOTS:
        p = HISTORICAL_PRIOR.get(s, 0.001)
        expected_games = round(1 / p)
        if expected_games <= 10:
            label = "common"
        elif expected_games <= 30:
            label = "moderate"
        elif expected_games <= 100:
            label = "rare"
        else:
            label = "very rare"
        slot_difficulty[s] = (expected_games, label)

    # ---- Standings rows ----
    standings_rows = ""
    for rank, team in enumerate(sorted_teams, 1):
        slots_have = team_runs.get(team, set())
        n = len(slots_have)
        missing = [s for s in SLOTS if s not in slots_have]
        pct = win_pct.get(team, 0)
        avg = avg_games.get(team)
        bar_w = round(pct / max_win_pct * 100) if max_win_pct else 0
        color = TEAM_COLORS.get(team, "#555")
        abbr = team_abbrev(team)
        missing_str = (
            ", ".join(str(m) for m in missing)
            if missing else "<span style='color:#22c55e'>COMPLETE!</span>"
        )
        avg_str = f"{avg:.0f}" if avg else "—"
        standings_rows += f"""
        <tr>
          <td class="rank">#{rank}</td>
          <td class="team-name">
            <span class="dot" style="background:{color}"></span>
            <span class="full">{team}</span>
            <span class="abbr">{abbr}</span>
          </td>
          <td class="slots-cell">
            <span class="slots-num">{n}/14</span>
            <div class="slot-pips">{''.join(
              f'<span class="pip filled" title="{s} runs" style="background:{color}"></span>' if s in slots_have
              else f'<span class="pip empty" title="{s} runs — needs {slot_difficulty[s][0]} avg games"></span>'
              for s in SLOTS
            )}</div>
          </td>
          <td class="missing-cell">{missing_str}</td>
          <td class="prob-cell">
            <div class="prob-bar-wrap">
              <div class="prob-bar" style="width:{bar_w}%;background:{color}"></div>
            </div>
            <span class="prob-num">{pct:.1f}%</span>
          </td>
          <td class="avg-cell">{avg_str}</td>
        </tr>"""

    # ---- Grid rows (also sorted by win prob) ----
    grid_rows = ""
    for team in sorted_teams:
        slots_have = team_runs.get(team, set())
        color = TEAM_COLORS.get(team, "#555")
        abbr = team_abbrev(team)
        cells = ""
        for s in SLOTS:
            if s in slots_have:
                claimer, cdate, opp = first_to_score.get(s, ("?", "?", "?"))
                is_first = claimer == team
                tip = f"Scored {s} runs vs {opp} on {cdate}"
                first_badge = " first" if is_first else ""
                cells += (
                    f'<td class="grid-cell claimed{first_badge}" '
                    f'style="background:{color}22;border-color:{color}55" title="{tip}">'
                    f'<span class="check" style="color:{color}">✓</span></td>'
                )
            else:
                exp, lbl = slot_difficulty[s]
                cells += f'<td class="grid-cell empty difficulty-{lbl}" title="{s} runs — {lbl} (~1 in {exp} games)"></td>'
        n = len(slots_have)
        grid_rows += f"""
        <tr>
          <td class="grid-team">
            <span class="dot" style="background:{color}"></span>
            <span class="full">{team}</span>
            <span class="abbr">{abbr}</span>
          </td>
          <td class="grid-total">{n}/14</td>
          {cells}
        </tr>"""

    # ---- Slot difficulty header annotation ----
    slot_headers = ""
    for s in SLOTS:
        exp, lbl = slot_difficulty[s]
        slot_headers += f'<th class="slot-header difficulty-{lbl}" title="{lbl} — ~1 in {exp} games">{s}</th>'

    # ---- First claims rows ----
    claim_rows = ""
    for s in SLOTS:
        exp, lbl = slot_difficulty[s]
        diff_badge = f'<span class="diff-badge {lbl}">{lbl}</span>'
        if s in first_to_score:
            team, cdate, opp = first_to_score[s]
            color = TEAM_COLORS.get(team, "#555")
            abbr = team_abbrev(team)
            claim_rows += f"""
            <tr>
              <td class="slot-num">{s}</td>
              <td>{diff_badge}</td>
              <td class="team-name">
                <span class="dot" style="background:{color}"></span>
                <span class="full">{team}</span>
                <span class="abbr">{abbr}</span>
              </td>
              <td>{cdate}</td>
              <td>vs {opp}</td>
            </tr>"""
        else:
            claim_rows += (
                f'<tr><td class="slot-num">{s}</td><td>{diff_badge}</td>'
                f'<td colspan="3" class="unclaimed">— unclaimed —</td></tr>'
            )

    leader = sorted_teams[0]
    leader_color = TEAM_COLORS.get(leader, "#fff")
    leader_slots = len(team_runs.get(leader, set()))
    leader_pct = win_pct.get(leader, 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>13-Run Pool Tracker — 2026 MLB Season</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #22263a;
    --border: #2e3352;
    --text: #e8eaf6;
    --muted: #8b90a7;
    --green: #22c55e;
    --gold: #f59e0b;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px 16px; max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.7rem; font-weight: 700; letter-spacing: -0.5px; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }}
  .header {{ display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 32px; }}
  .header-meta {{ color: var(--muted); font-size: 0.85rem; text-align: right; line-height: 1.6; }}
  .section {{ margin-bottom: 40px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}

  /* Badges */
  .badges {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
  .badge {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 16px; }}
  .badge-val {{ font-size: 1.4rem; font-weight: 700; }}
  .badge-label {{ font-size: 0.75rem; color: var(--muted); margin-top: 2px; }}

  /* Leader */
  .leader-box {{ background: linear-gradient(135deg, var(--surface2), var(--surface)); border: 1px solid #f59e0b55; border-radius: 10px; padding: 14px 18px; margin-bottom: 20px; display: flex; align-items: center; gap: 12px; }}
  .leader-label {{ color: var(--gold); font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }}
  .leader-team {{ font-size: 1.1rem; font-weight: 700; }}
  .leader-detail {{ font-size: 0.85rem; color: var(--muted); }}

  /* Standings */
  .standings-table {{ width: 100%; border-collapse: collapse; }}
  .standings-table th {{ background: var(--surface2); color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); cursor: default; }}
  .standings-table td {{ padding: 10px 12px; border-bottom: 1px solid #2e335222; vertical-align: middle; font-size: 0.9rem; }}
  .standings-table tr:last-child td {{ border-bottom: none; }}
  .standings-table tr:hover td {{ background: var(--surface2); }}
  .rank {{ color: var(--muted); font-size: 0.8rem; width: 32px; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; flex-shrink: 0; }}
  .team-name {{ display: flex; align-items: center; white-space: nowrap; font-weight: 500; }}
  .abbr {{ display: none; }}
  .slots-cell {{ min-width: 200px; }}
  .slots-num {{ font-weight: 700; font-size: 0.95rem; margin-right: 8px; }}
  .slot-pips {{ display: inline-flex; gap: 3px; vertical-align: middle; }}
  .pip {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
  .pip.filled {{ opacity: 0.9; }}
  .pip.empty {{ background: var(--surface2); border: 1px solid var(--border); }}
  .missing-cell {{ color: var(--muted); font-size: 0.82rem; }}
  .prob-cell {{ min-width: 150px; }}
  .prob-bar-wrap {{ background: var(--surface2); border-radius: 4px; height: 6px; width: 80px; display: inline-block; vertical-align: middle; margin-right: 8px; }}
  .prob-bar {{ height: 6px; border-radius: 4px; min-width: 2px; }}
  .prob-num {{ font-size: 0.88rem; font-weight: 600; }}
  .avg-cell {{ color: var(--muted); font-size: 0.85rem; }}

  /* Grid */
  .grid-wrap {{ overflow-x: auto; }}
  .grid-table {{ border-collapse: collapse; min-width: 700px; width: 100%; }}
  .grid-table th {{ background: var(--surface2); color: var(--muted); font-size: 0.72rem; padding: 8px 4px; text-align: center; border: 1px solid var(--border); min-width: 32px; }}
  .slot-header.difficulty-common {{ color: #22c55e; }}
  .slot-header.difficulty-moderate {{ color: #f59e0b; }}
  .slot-header.difficulty-rare {{ color: #f97316; }}
  .slot-header.difficulty-very.rare {{ color: #ef4444; }}
  .grid-table .grid-team {{ padding: 6px 10px; white-space: nowrap; font-size: 0.82rem; font-weight: 500; border: 1px solid var(--border); }}
  .grid-table td {{ border: 1px solid #2e335233; text-align: center; padding: 0; height: 32px; }}
  .grid-total {{ font-size: 0.75rem; color: var(--muted); padding: 4px 6px !important; width: 42px; }}
  .grid-cell {{ cursor: default; transition: opacity 0.15s; }}
  .grid-cell:hover {{ opacity: 0.65; }}
  .grid-cell.empty {{ background: var(--surface); }}
  .grid-cell.empty.difficulty-rare {{ background: #1a1520; }}
  .grid-cell.empty.difficulty-very {{ background: #1a1118; }}
  .grid-cell.claimed.first .check::after {{ content: '★'; font-size: 7px; vertical-align: super; margin-left: 1px; }}
  .check {{ font-size: 0.95rem; }}

  /* First claims */
  .claims-table {{ width: 100%; border-collapse: collapse; }}
  .claims-table th {{ background: var(--surface2); color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  .claims-table td {{ padding: 9px 12px; border-bottom: 1px solid #2e335222; font-size: 0.88rem; vertical-align: middle; }}
  .claims-table tr:last-child td {{ border-bottom: none; }}
  .slot-num {{ font-weight: 700; font-size: 1rem; width: 40px; }}
  .unclaimed {{ color: var(--muted); font-style: italic; }}

  /* Difficulty badges */
  .diff-badge {{ display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; }}
  .diff-badge.common {{ background: #22c55e22; color: #22c55e; }}
  .diff-badge.moderate {{ background: #f59e0b22; color: #f59e0b; }}
  .diff-badge.rare {{ background: #f9731622; color: #f97316; }}
  .diff-badge.very.rare {{ background: #ef444422; color: #ef4444; }}

  /* Prob note */
  .prob-note {{ background: var(--surface2); border-left: 3px solid var(--border); border-radius: 4px; padding: 10px 14px; font-size: 0.82rem; color: var(--muted); margin-bottom: 14px; }}

  @media (max-width: 640px) {{
    .full {{ display: none; }}
    .abbr {{ display: inline; }}
    .slot-pips .pip {{ width: 9px; height: 9px; }}
    h1 {{ font-size: 1.3rem; }}
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>⚾ 13-Run Pool Tracker</h1>
    <div style="color:var(--muted);margin-top:4px;font-size:0.9rem;">2026 MLB Season · First team to score 0–13 runs in separate games wins</div>
  </div>
  <div class="header-meta">
    Last updated: {TODAY}<br>
    Games processed: {total_games:,}<br>
    Simulations: {N_SIMS:,}
  </div>
</div>

<div class="badges">
  <div class="badge"><div class="badge-val">{leader_slots}/14</div><div class="badge-label">Leader slots</div></div>
  <div class="badge"><div class="badge-val">{claimed_count}/14</div><div class="badge-label">Slots claimed (league)</div></div>
  <div class="badge"><div class="badge-val">{len(all_teams)}</div><div class="badge-label">Teams active</div></div>
  <div class="badge"><div class="badge-val">{total_games:,}</div><div class="badge-label">Games played</div></div>
</div>

<div class="leader-box">
  <div style="font-size:1.6rem">🏆</div>
  <div>
    <div class="leader-label">Current Leader (by win probability)</div>
    <div class="leader-team" style="color:{leader_color}">{leader}</div>
    <div class="leader-detail">{leader_slots}/14 slots · {leader_pct:.1f}% win probability</div>
  </div>
</div>

<!-- STANDINGS -->
<div class="section">
  <h2>Standings — sorted by win probability</h2>
  <div class="prob-note">
    Win % = Monte Carlo simulation ({N_SIMS:,} runs). Probabilities use a historical MLB run distribution as a prior
    (scoring 13 runs ≈ 0.5% chance/game vs. scoring 4 runs ≈ 14%), blended with 2026 team data.
    The race runs until any team completes all 14 slots — it may extend beyond one season.
    "Avg games" = expected games from today until that team finishes.
  </div>
  <div class="card" style="overflow-x:auto">
    <table class="standings-table">
      <thead>
        <tr>
          <th></th>
          <th>Team</th>
          <th>Slots filled</th>
          <th>Still needs</th>
          <th>Win probability</th>
          <th>Avg games to finish</th>
        </tr>
      </thead>
      <tbody>{standings_rows}</tbody>
    </table>
  </div>
</div>

<!-- GRID -->
<div class="section">
  <h2>Progress Grid
    <span style="font-size:0.7rem;font-weight:400;margin-left:8px">
      ✓ = scored &nbsp;·&nbsp; ✓★ = first in league &nbsp;·&nbsp;
      <span style="color:#22c55e">green cols</span> = common &nbsp;·&nbsp;
      <span style="color:#f59e0b">amber</span> = moderate &nbsp;·&nbsp;
      <span style="color:#f97316">orange</span> = rare &nbsp;·&nbsp;
      <span style="color:#ef4444">red</span> = very rare
    </span>
  </h2>
  <div class="card grid-wrap">
    <table class="grid-table">
      <thead>
        <tr>
          <th style="text-align:left;padding-left:10px">Team</th>
          <th>Done</th>
          {slot_headers}
        </tr>
      </thead>
      <tbody>{grid_rows}</tbody>
    </table>
  </div>
</div>

<!-- FIRST CLAIMS -->
<div class="section">
  <h2>First Team to Score Each Total</h2>
  <div class="card">
    <table class="claims-table">
      <thead>
        <tr><th>Runs</th><th>Difficulty</th><th>Team</th><th>Date</th><th>Opponent</th></tr>
      </thead>
      <tbody>{claim_rows}</tbody>
    </table>
  </div>
</div>

<div style="color:var(--muted);font-size:0.78rem;text-align:center;margin-top:32px;padding-bottom:32px">
  Data: MLB Stats API (statsapi.mlb.com) &nbsp;·&nbsp;
  Probabilities: Monte Carlo ({N_SIMS:,} simulations) with historical MLB run distribution prior &nbsp;·&nbsp;
  Refresh: <code>python mlb_pool_tracker.py</code>
</div>

</body>
</html>"""
    return html


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    game_log, team_run_dist = fetch_season_data()
    team_runs, first_to_score = build_tracker(game_log)
    team_dist, league_dist = build_run_distributions(team_run_dist)
    all_teams = sorted(team_runs.keys())
    win_pct, avg_games = simulate_pool(team_runs, team_dist, all_teams)
    html = generate_html(team_runs, first_to_score, win_pct, avg_games, team_run_dist, game_log)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDone! Open {OUTPUT_FILE} in your browser.")
    sorted_teams = sorted(all_teams, key=lambda t: -win_pct.get(t, 0))
    print("\nTop 10 by win probability:")
    for t in sorted_teams[:10]:
        n = len(team_runs.get(t, set()))
        p = win_pct.get(t, 0)
        avg = avg_games.get(t)
        avg_str = f"{avg:.0f}g" if avg else "—"
        print(f"  {t:<32} {n}/14 slots   {p:5.1f}%   {avg_str}")
