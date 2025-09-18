# coding: utf-8
import os, time, math, glob, logging, requests
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
from dateutil.tz import gettz
from rapidfuzz import process, fuzz
import schedule

# ===================== CONFIG =====================
LOCAL_TZ = gettz("Europe/Rome")
ACTIVE_START_HOUR = int(os.environ.get("ACTIVE_START_HOUR", "8"))
ACTIVE_END_HOUR   = int(os.environ.get("ACTIVE_END_HOUR",   "23"))
RUN_EVERY_MIN     = int(os.environ.get("RUN_EVERY_MIN",     "30"))
LOOKAHEAD_HOURS   = float(os.environ.get("LOOKAHEAD_HOURS", "1.0"))

ODDS_API_KEY      = os.environ.get("ODDS_API_KEY")
TELEGRAM_TOKEN    = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID  = os.environ.get("TELEGRAM_CHAT_ID")

EDGE_MIN          = float(os.environ.get("EDGE_MIN", "0.0"))
MIN_PROB_TENNIS   = float(os.environ.get("MIN_PROB_TENNIS", "0.90"))
MIN_PROB_SOCCER   = float(os.environ.get("MIN_PROB_SOCCER", "0.80"))
MIN_PROB_BASKET   = float(os.environ.get("MIN_PROB_BASKET", "0.80"))
MIN_ODDS_ALL      = float(os.environ.get("MIN_ODDS_ALL", "1.70"))
MIN_GAMES         = int(os.environ.get("MIN_GAMES", "20"))

DATA_TENNIS_DIR   = os.path.join("data", "tennis")
DATA_SOCCER_DIR   = os.path.join("data", "soccer")
DATA_BASKET_DIR   = os.path.join("data", "basketball")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ===================== UTILS ======================
def now_local():
    return datetime.now(tz=LOCAL_TZ)

def to_local(dt_str):
    if not dt_str:
        return None
    try:
        dt = dateparser.parse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ)
    except Exception:
        return None

def valid_future_match(start):
    if not start:
        return False
    now = now_local()
    if start <= now:
        return False
    if (start - now) < timedelta(hours=LOOKAHEAD_HOURS):
        return False
    return True

def send_tg(text):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        logging.warning("[TG] Dry-run: %s", text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text}
        )
    except Exception as e:
        logging.error("Telegram error: %s", e)

def fuzzy_one(q, choices, cutoff=65):
    if not choices:
        return (None, 0)
    res = process.extractOne(q, choices, scorer=fuzz.token_sort_ratio)
    if res and res[1] >= cutoff:
        return res[0], res[1]
    return (None, 0)

# ================= SPORT AUTODETECT ===============
SPORTS_AVAILABLE = []
if ODDS_API_KEY:
    try:
        url = f"https://api.the-odds-api.com/v4/sports/?apiKey={ODDS_API_KEY}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        SPORTS_AVAILABLE = resp.json()
    except Exception as e:
        logging.error("Errore fetch lista sport: %s", e)
        SPORTS_AVAILABLE = []

def sports_with(keyword, exclude_keywords=None):
    exclude_keywords = exclude_keywords or []
    sports = []
    for s in SPORTS_AVAILABLE:
        key = s.get("key","")
        if keyword in key and s.get("active"):
            if not any(ex in key for ex in exclude_keywords):
                sports.append(key)
    return sports

# Escludiamo futures/outrights per evitare 422
TENNIS_SPORTS     = sports_with("tennis", exclude_keywords=["winner","championship","outright"])
SOCCER_SPORTS     = sports_with("soccer", exclude_keywords=["winner","championship","outright"])
BASKETBALL_SPORTS = sports_with("basketball", exclude_keywords=["winner","championship","outright"])

logging.info("Sport attivi filtrati: Tennis=%s Soccer=%s Basket=%s",
             TENNIS_SPORTS, SOCCER_SPORTS, BASKETBALL_SPORTS)

# ================== ODDS FETCH ====================
def fetch_odds_for_sport(sport_key, markets="h2h"):
    if not ODDS_API_KEY:
        return []
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions":"eu", "markets":markets, "oddsFormat":"decimal"}
    try:
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error("Odds fetch error %s/%s: %s", sport_key, markets, e)
        return []

# ================= DATE HELPERS ===================
def parse_any_date_column(df, candidates):
    """Restituisce una Serie naive (senza tz) convertita in Europe/Rome per confronti coerenti."""
    for col in candidates:
        if col in df.columns:
            try:
                temp = pd.to_datetime(df[col], errors="coerce", utc=True)
                # Porta in LOCAL_TZ e poi rendi naive (dtype datetime64[ns])
                temp = temp.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
                return temp
            except Exception:
                continue
    return pd.Series([pd.NaT] * len(df))

def naive_utc_now():
    # Timestamp UTC naive per confronti con colonne datetime64[ns] (naive)
    return pd.Timestamp.utcnow().tz_localize(None)

# ==================== LOADERS =====================
def load_all_tennis():
    files = glob.glob(os.path.join(DATA_TENNIS_DIR, "*.csv"))
    if not files:
        logging.info("Nessun CSV tennis trovato in %s", DATA_TENNIS_DIR)
        return pd.DataFrame(columns=["winner_name","loser_name","tourney_date"])
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            logging.warning("Errore lettura tennis %s: %s", f, e)
    if not dfs:
        return pd.DataFrame(columns=["winner_name","loser_name","tourney_date"])
    df = pd.concat(dfs, ignore_index=True, sort=False).fillna("")
    df["tourney_date"] = parse_any_date_column(df, ["tourney_date","date","match_date","Date"])
    return df

def tennis_player_winrate(df, months=12):
    if df.empty:
        return {}
    cutoff = naive_utc_now() - pd.Timedelta(days=30*months)
    recent = df[df["tourney_date"] >= cutoff] if "tourney_date" in df.columns else df
    wins = recent["winner_name"].value_counts() if "winner_name" in recent.columns else pd.Series(dtype=int)
    losses = recent["loser_name"].value_counts() if "loser_name" in recent.columns else pd.Series(dtype=int)
    players = set(wins.index.tolist()) | set(losses.index.tolist())
    rates = {}
    for p in players:
        w = int(wins.get(p, 0)); l = int(losses.get(p, 0)); n = w + l
        if n < MIN_GAMES:
            continue
        rates[p] = w / n if n > 0 else 0.5
    return rates

def load_all_soccer():
    files = glob.glob(os.path.join(DATA_SOCCER_DIR, "*.csv"))
    if not files:
        logging.info("Nessun CSV calcio trovato in %s", DATA_SOCCER_DIR)
        return pd.DataFrame(columns=["HomeTeam","AwayTeam","FTHG","FTAG","Date"])
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            logging.warning("Errore lettura calcio %s: %s", f, e)
    if not dfs:
        return pd.DataFrame(columns=["HomeTeam","AwayTeam","FTHG","FTAG","Date"])
    df = pd.concat(dfs, ignore_index=True, sort=False).fillna("")
    # standardizza nomi squadra
    if "HomeTeam" not in df.columns:
        for alt in ["home_team","Home","HOME","Home Team"]:
            if alt in df.columns:
                df["HomeTeam"] = df[alt].astype(str)
                break
    if "AwayTeam" not in df.columns:
        for alt in ["away_team","Away","AWAY","Away Team"]:
            if alt in df.columns:
                df["AwayTeam"] = df[alt].astype(str)
                break
    # goals numeric
    for target, alts in {"FTHG":["FTHG","HomeGoals","HG","FTHG_home"], "FTAG":["FTAG","AwayGoals","AG","FTAG_away"]}.items():
        col = next((a for a in alts if a in df.columns), None)
        if col:
            df[target] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        else:
            df[target] = 0
    # date naive
    df["Date"] = parse_any_date_column(df, ["Date","date","MatchDate"])
    return df

def soccer_team_rates(df, months=12):
    if df.empty:
        return {}, {}, {}
    cutoff = naive_utc_now() - pd.Timedelta(days=365)
    if "Date" in df.columns:
        df = df[df["Date"] >= cutoff]
    df["BTTS"]   = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
    df["Over25"] = ((df["FTHG"] + df["FTAG"]) >= 3).astype(int)
    teams = pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True).dropna())
    btts_rate = {}; over25_rate = {}; counts = {}
    gf = {}; ga = {}
    for t in teams:
        home = df[df["HomeTeam"]==t]; away = df[df["AwayTeam"]==t]
        n = len(home) + len(away)
        if n < MIN_GAMES:
            continue
        btts = int(home["BTTS"].sum() + away["BTTS"].sum())
        over = int(home["Over25"].sum() + away["Over25"].sum())
        goals_for = int(home["FTHG"].sum() + away["FTAG"].sum())
        goals_ag  = int(home["FTAG"].sum() + away["FTHG"].sum())
        btts_rate[t]   = btts / n
        over25_rate[t] = over / n
        counts[t]      = n
        gf[t] = goals_for / n
        ga[t] = goals_ag  / n
    return btts_rate, over25_rate, {"gf":gf, "ga":ga, "n":counts}

def load_all_basket():
    files = glob.glob(os.path.join(DATA_BASKET_DIR, "*.csv"))
    if not files:
        logging.info("Nessun CSV basket trovato in %s", DATA_BASKET_DIR)
        return pd.DataFrame(columns=["team","off_rating","def_rating","pace"])
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            logging.warning("Errore lettura basket %s: %s", f, e)
    if not dfs:
        return pd.DataFrame(columns=["team","off_rating","def_rating","pace"])
    df = pd.concat(dfs, ignore_index=True, sort=False)
    # team column
    team_col = next((c for c in ["team","Team","TEAM","team_name","squadra"] if c in df.columns), None)
    df["team"] = df[team_col].astype(str) if team_col else "Unknown"
    # numeric columns: crea se mancano, poi to_numeric
    for col, default in [("off_rating",110),("def_rating",110),("pace",99.0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
        else:
            df[col] = default
    return df[["team","off_rating","def_rating","pace"]].dropna()

# ===================== MODELLI ====================
def tennis_best_picks():
    df = load_all_tennis()
    rates = tennis_player_winrate(df, months=12)
    players = list(rates.keys())
    picks=[]
    for sport in TENNIS_SPORTS:
        for ev in fetch_odds_for_sport(sport, markets="h2h"):
            start = to_local(ev.get("commence_time"))
            if not valid_future_match(start):
                continue
            bms = ev.get("bookmakers", [])
            if not bms:
                continue
            # prendi il primo mercato disponibile
            mk=None
            for bm in bms:
                ms=bm.get("markets",[])
                if ms:
                    mk=ms[0]; break
            if not mk:
                continue
            oc = mk.get("outcomes",[])
            if len(oc)<2:
                continue
            est=[]
            for o in oc:
                name=o.get("name",""); price=float(o.get("price",0))
                mapped,_=fuzzy_one(name, players, cutoff=65)
                p=rates.get(mapped, None)
                if p is None: p=0.5
                est.append({"name":name,"price":price,"p":p})
            s=sum(e["p"] for e in est) or 1.0
            for e in est: e["p"]/=s
            for e in est:
                price=e["price"]; implied=1/price if price>0 else 0; edge=e["p"]-implied
                if price>=MIN_ODDS_ALL and e["p"]>=MIN_PROB_TENNIS and edge>EDGE_MIN:
                    picks.append({
                        "sport":"Tennis","tournament": ev.get("sport_title","Tennis"),
                        "home":ev.get("home_team"),"away":ev.get("away_team"),
                        "start":start.isoformat(),"market":"Vincente","selection":e["name"],
                        "odds":price,"model_prob":round(e["p"],3),"implied":round(implied,3),
                        "edge":round(edge,3),"bookmaker":bms[0].get("title","")
                    })
    picks.sort(key=lambda x:(x["edge"], x["model_prob"]), reverse=True)
    return picks

def soccer_best_picks():
    # Calcolo basato su storici: Over/Under 2.5
    df = load_all_soccer()
    btts_rate, over25_rate, agg = soccer_team_rates(df, months=12)
    teams = list(set(list(btts_rate.keys())+list(over25_rate.keys())))
    picks=[]
    for sport in SOCCER_SPORTS:
        for ev in fetch_odds_for_sport(sport, markets="totals"):
            start = to_local(ev.get("commence_time"))
            if not valid_future_match(start):
                continue
            home, away = ev.get("home_team",""), ev.get("away_team","")
            mh,_ = fuzzy_one(home, teams); ma,_ = fuzzy_one(away, teams)
            if not (mh and ma):
                continue
            p_over = np.mean([over25_rate.get(mh,0.5), over25_rate.get(ma,0.5)])
            p_under = 1 - p_over
            # trova mercato totals
            mk=None; bm=None
            for bm_ in ev.get("bookmakers",[]):
                for m in bm_.get("markets",[]):
                    if m.get("key","")=="totals":
                        mk=m; bm=bm_; break
                if mk: break
            if not mk: 
                continue
            oc = mk.get("outcomes",[])
            over = next((o for o in oc if str(o.get("name","")).lower()=="over"), None)
            under= next((o for o in oc if str(o.get("name","")).lower()=="under"), None)
            if over:
                try: line=float(over.get("point", 2.5))
                except: line=2.5
                price=float(over.get("price",0)); implied=1/price if price>0 else 0; edge=p_over-implied
                if price>=MIN_ODDS_ALL and p_over>=MIN_PROB_SOCCER and edge>EDGE_MIN:
                    picks.append({
                        "sport":"Calcio","tournament":ev.get("sport_title","Calcio"),
                        "home":home,"away":away,"start":start.isoformat(),
                        "market":f"Over/Under {line}","selection":f"Over {line}",
                        "odds":price,"model_prob":round(p_over,3),"implied":round(implied,3),
                        "edge":round(edge,3),"bookmaker":bm.get("title","")
                    })
            if under:
                try: line=float(under.get("point", 2.5))
                except: line=2.5
                price=float(under.get("price",0)); implied=1/price if price>0 else 0; edge=p_under-implied
                if price>=MIN_ODDS_ALL and p_under>=MIN_PROB_SOCCER and edge>EDGE_MIN:
                    picks.append({
                        "sport":"Calcio","tournament":ev.get("sport_title","Calcio"),
                        "home":home,"away":away,"start":start.isoformat(),
                        "market":f"Over/Under {line}","selection":f"Under {line}",
                        "odds":price,"model_prob":round(p_under,3),"implied":round(implied,3),
                        "edge":round(edge,3),"bookmaker":bm.get("title","")
                    })
    picks.sort(key=lambda x:(x["edge"], x["model_prob"]), reverse=True)
    return picks

def basket_best_picks():
    df = load_all_basket()
    teams = df["team"].astype(str).unique().tolist() if "team" in df.columns else []
    picks=[]
    # Vincente (H2H)
    for sport in BASKETBALL_SPORTS:
        for ev in fetch_odds_for_sport(sport, markets="h2h"):
            start = to_local(ev.get("commence_time"))
            if not valid_future_match(start):
                continue
            home, away = ev.get("home_team",""), ev.get("away_team","")
            mh,_ = fuzzy_one(home, teams); ma,_ = fuzzy_one(away, teams)
            if not (mh and ma):
                continue
            rh = df[df["team"]==mh]; ra = df[df["team"]==ma]
            if len(rh)<MIN_GAMES or len(ra)<MIN_GAMES:
                continue
            s_home = float(rh["off_rating"].mean() - rh["def_rating"].mean() + 1.5)
            s_away = float(ra["off_rating"].mean() - ra["def_rating"].mean())
            p_home = 1/(1+math.exp(-(s_home - s_away)/10.0)); p_away = 1 - p_home
            bms=ev.get("bookmakers",[]); 
            if not bms: 
                continue
            mk0 = bms[0].get("markets", [])
            if not mk0:
                continue
            oc0 = mk0[0].get("outcomes", []) if mk0 else []
            price_home=price_away=None
            for o in oc0:
                nm=str(o.get("name","")).lower().strip()
                pr=float(o.get("price",0))
                if nm==home.lower().strip(): price_home=pr
                elif nm==away.lower().strip(): price_away=pr
            if price_home:
                implied=1/price_home if price_home>0 else 0; edge=p_home-implied
                if price_home>=MIN_ODDS_ALL and p_home>=MIN_PROB_BASKET and edge>EDGE_MIN:
                    picks.append({
                        "sport":"Basket","tournament":ev.get("sport_title","Basket"),
                        "home":home,"away":away,"start":start.isoformat(),
                        "market":"Vincente","selection":home,
                        "odds":price_home,"model_prob":round(p_home,3),"implied":round(implied,3),
                        "edge":round(edge,3),"bookmaker":bms[0].get("title","")
                    })
            if price_away:
                implied=1/price_away if price_away>0 else 0; edge=p_away-implied
                if price_away>=MIN_ODDS_ALL and p_away>=MIN_PROB_BASKET and edge>EDGE_MIN:
                    picks.append({
                        "sport":"Basket","tournament":ev.get("sport_title","Basket"),
                        "home":home,"away":away,"start":start.isoformat(),
                        "market":"Vincente","selection":away,
                        "odds":price_away,"model_prob":round(p_away,3),"implied":round(implied,3),
                        "edge":round(edge,3),"bookmaker":bms[0].get("title","")
                    })
    # Totali punti (Over/Under)
    for sport in BASKETBALL_SPORTS:
        for ev in fetch_odds_for_sport(sport, markets="totals"):
            start = to_local(ev.get("commence_time"))
            if not valid_future_match(start):
                continue
            home, away = ev.get("home_team",""), ev.get("away_team","")
            mh,_ = fuzzy_one(home, teams); ma,_ = fuzzy_one(away, teams)
            if not (mh and ma):
                continue
            rh = df[df["team"]==mh]; ra = df[df["team"]==ma]
            if len(rh)<MIN_GAMES or len(ra)<MIN_GAMES:
                continue
            expected_total = (rh["off_rating"].mean()+ra["off_rating"].mean())/2 + (rh["pace"].mean()+ra["pace"].mean())*0.1
            sigma = 12.0
            mk=None; bm=None
            for bm_ in ev.get("bookmakers",[]):
                for m in bm_.get("markets",[]):
                    if m.get("key","")=="totals":
                        mk=m; bm=bm_; break
                if mk: break
            if not mk: 
                continue
            oc = mk.get("outcomes",[])
            over = next((o for o in oc if str(o.get("name","")).lower()=="over"), None)
            under= next((o for o in oc if str(o.get("name","")).lower()=="under"), None)
            if over:
                line=float(over.get("point", expected_total))
                z=(expected_total - line)/sigma
                p_over = 1 - 0.5*(1+math.erf(z/np.sqrt(2)))
                price=float(over.get("price",0)); implied=1/price if price>0 else 0; edge=p_over-implied
                if price>=MIN_ODDS_ALL and p_over>=MIN_PROB_BASKET and edge>EDGE_MIN:
                    picks.append({
                        "sport":"Basket","tournament":ev.get("sport_title","Basket"),
                        "home":home,"away":away,"start":start.isoformat(),
                        "market":"Totali punti","selection":f"Over {line}",
                        "odds":price,"model_prob":round(p_over,3),"implied":round(implied,3),
                        "edge":round(edge,3),"bookmaker":bm.get("title","")
                    })
            if under:
                line=float(under.get("point", expected_total))
                z=(expected_total - line)/sigma
                p_under = 0.5*(1+math.erf(z/np.sqrt(2)))
                price=float(under.get("price",0)); implied=1/price if price>0 else 0; edge=p_under-implied
                if price>=MIN_ODDS_ALL and p_under>=MIN_PROB_BASKET and edge>EDGE_MIN:
                    picks.append({
                        "sport":"Basket","tournament":ev.get("sport_title","Basket"),
                        "home":home,"away":away,"start":start.isoformat(),
                        "market":"Totali punti","selection":f"Under {line}",
                        "odds":price,"model_prob":round(p_under,3),"implied":round(implied,3),
                        "edge":round(edge,3),"bookmaker":bm.get("title","")
                    })
    picks.sort(key=lambda x:(x["edge"], x["model_prob"]), reverse=True)
    return picks

# ===================== MAIN =======================
def format_pick(p):
    return (
        f"🏆 {p.get('tournament', p['sport'])} — {p['sport']}\n"
        f"👥 {p.get('home','?')} vs {p.get('away','?')}\n"
        f"⏰ {p['start']}\n"
        f"🛒 Mercato: {p['market']}\n"
        f"✅ Selezione: {p['selection']}\n"
        f"💰 Quota: {p['odds']:.2f}\n"
        f"📈 Prob modello: {p['model_prob']:.3f} | 📉 Implicita: {p['implied']:.3f}\n"
        f"🔺 Edge: {p['edge']:.3f}\n"
        f"🏷️ Book: {p.get('bookmaker','-')}"
    )

def run_once():
    now = now_local()
    if not (ACTIVE_START_HOUR <= now.hour < ACTIVE_END_HOUR):
        logging.info("Fuori fascia oraria. Skip.")
        return

    all_picks = []
    try: all_picks += tennis_best_picks()
    except Exception as e: logging.error("Tennis error: %s", e)
    try: all_picks += soccer_best_picks()
    except Exception as e: logging.error("Soccer error: %s", e)
    try: all_picks += basket_best_picks()
    except Exception as e: logging.error("Basket error: %s", e)

    if not all_picks:
        logging.info("Nessun pick in questa run.")
        return

    top = sorted(all_picks, key=lambda x: (x["edge"], x["model_prob"]), reverse=True)[:5]
    for p in top:
        send_tg(format_pick(p))

if __name__ == "__main__":
    run_once()
    schedule.every(RUN_EVERY_MIN).minutes.do(run_once)
    logging.info("🤖 Scheduler ON — ogni %d min, %02d:00–%02d:00 Europe/Rome",
                 RUN_EVERY_MIN, ACTIVE_START_HOUR, ACTIVE_END_HOUR)
    while True:
        schedule.run_pending()
        time.sleep(20)
