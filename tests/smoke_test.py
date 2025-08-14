import os
import openai
import json
from pathlib import Path
import numpy as np
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def _exists(p):
    return Path(p).exists()


def _load_joblib(p):
    try:
        from joblib import load

        return load(p)
    except Exception as e:
        print(f"[ERR] Nu pot încărca {p}: {e}")
        return None


def test_1_features(symbol="AAPL"):
    print("\n=== TEST 1: LIVE FEATURES ===")
    from app.features import make_live_features

    feats = make_live_features(symbol)
    if not feats:
        print(
            "[FAIL] make_live_features a returnat dict gol (nu sunt suficiente candele?)."
        )
        return None, None, None

    print(f"[OK] features dict cu {len(feats)} chei.")

    fjson = Path("models_store") / "features.json"
    if not fjson.exists():
        print(
            "[WARN] models_store/features.json nu există. Folosesc cheile din live_features ca referință."
        )
        feat_names = sorted(feats.keys())
    else:
        feat_names = json.load(open(fjson, "r"))
        missing = [n for n in feat_names if n not in feats]
        extra = [k for k in feats.keys() if k not in feat_names]
        print(f"[CHK] missing={len(missing)} | extra={len(extra)}")
        if missing:
            print("      MISSING KEYS:", missing)
        if extra:
            print("      EXTRA KEYS  :", extra)

    row = np.array([[feats.get(n, 0.0) for n in feat_names]], dtype=float)
    print(f"[OK] vector live shape = {row.shape}")
    return row, feat_names, feats


def test_2_models(row):
    print("\n=== TEST 2: MODELE LOADING & PREDICT ===")
    reg_p = Path("models_store") / "regressor.pkl"
    clf_p = Path("models_store") / "classifier.pkl"
    cal_p = Path("models_store") / "calibrator.pkl"
    q20_p = Path("models_store") / "reg_q20.pkl"
    q80_p = Path("models_store") / "reg_q80.pkl"

    print("[CHK] files:")
    for p in [reg_p, clf_p, cal_p, q20_p, q80_p]:
        print("    ", p, "OK" if p.exists() else "MISSING")

    REG = _load_joblib(reg_p) if reg_p.exists() else None
    CLF = _load_joblib(clf_p) if clf_p.exists() else None
    CAL = _load_joblib(cal_p) if cal_p.exists() else None
    Q20 = _load_joblib(q20_p) if q20_p.exists() else None
    Q80 = _load_joblib(q80_p) if q80_p.exists() else None

    if REG is not None:
        try:
            reg_pred = float(REG.predict(row)[0])
            print(f"[OK] REG.predict -> {reg_pred:.4f} (%)")
        except Exception as e:
            print("[FAIL] REG.predict:", e)
    else:
        print("[WARN] Regressor absent.")

    if CLF is not None:
        try:
            prob = float(CLF.predict_proba(row)[0, 1] * 100.0)
            print(f"[OK] CLF.predict_proba -> {prob:.2f} (%)")
            if CAL is not None:
                try:
                    # calibratorul tău poate fi sklearn CalibratedClassifier sau un obiect custom;
                    # încercăm pattern comun .predict_proba pe ieșirea CLF sau transform numeric.
                    if hasattr(CAL, "transform"):
                        cal_prob = float(CAL.transform(prob))
                        print(f"[OK] CAL.transform(prob) -> {cal_prob:.2f} (%)")
                    elif hasattr(CAL, "predict_proba"):
                        # dacă e un calibrator sklearn, îi dăm probabilitatea necalibrată în formă [ [1-p, p] ]
                        import numpy as np

                        cal_prob = float(
                            CAL.predict_proba(np.array([[1 - prob / 100, prob / 100]]))[
                                0, 1
                            ]
                            * 100.0
                        )
                        print(f"[OK] CAL.predict_proba -> {cal_prob:.2f} (%)")
                    else:
                        print(
                            "[WARN] Calibrator găsit dar nu are API așteptat (transform/predict_proba)."
                        )
                except Exception as e:
                    print("[WARN] Calibrator call a eșuat:", e)
        except Exception as e:
            print("[FAIL] CLF.predict_proba:", e)
    else:
        print("[WARN] Classifier absent.")

    if Q20 is not None and Q80 is not None:
        try:
            q20 = float(Q20.predict(row)[0])
            q80 = float(Q80.predict(row)[0])
            print(f"[OK] Quantiles -> P20={q20:.2f}%  P80={q80:.2f}%")
        except Exception as e:
            print("[WARN] Quantile predict a eșuat:", e)
    else:
        print("[WARN] Quantile models lipsă (opțional).")


def test_3_pipeline(symbol="AAPL", force_refresh=True):
    print("\n=== TEST 3: PIPELINE END-TO-END (batch_process) ===")
    try:
        from app.stock_analysis import batch_process

        res = batch_process([symbol], force_refresh=force_refresh, batch_size=1)
        item = res.get(symbol)
        if not item:
            print("[FAIL] batch_process nu a întors date pentru simbol.")
            return
        print("[OK] batch_process a întors un dict.")
        # afișăm cheile importante
        for k in [
            "percent",
            "probability",
            "reward_to_risk",
            "recommendation",
            "trade_outcome",
            "actual_price",
        ]:
            print(f"    {k}: {item.get(k)}")
        # checks ușoare
        if item.get("percent") is None:
            print("[WARN] percent este None (verificăm parsarea LLM sau ensemble).")
        if item.get("probability") is None:
            print(
                "[WARN] probability este None (verificăm parsarea LLM sau calibrator)."
            )
        return
    except Exception as e:
        print("[FAIL] batch_process error:", e)


def main():
    sym = "AAPL"
    row, feat_names, feats = test_1_features(sym)
    if row is not None:
        test_2_models(row)
    # rulăm pipeline doar dacă ai OPENAI_API_KEY setat;
    # poți forța să sari testul 3 dacă nu vrei call la LLM.
    if os.getenv("OPENAI_API_KEY"):
        test_3_pipeline(sym, force_refresh=True)
    else:
        print("\n[SKIP] TEST 3: OPENAI_API_KEY nu e setat; sari LLM pipeline.")


if __name__ == "__main__":
    main()
