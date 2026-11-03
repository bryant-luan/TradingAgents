"""Post-trade learning: losses tighten the parameters, wins are explained and recorded."""
import json
import time
from pathlib import Path

from . import strategy as S

MIN_SCORE_CAP, MIN_SCORE_FLOOR = 70, 48
LOSS_SCORE_STEP, WIN_SCORE_STEP = 2, 1
LOSS_COOLDOWN_H = 12
HALF_RISK_AFTER_LOSSES = 3
FACTOR_MIN_N = 3
FACTOR_GAIN, FACTOR_ADJ_MIN, FACTOR_ADJ_MAX = 8.0, -15.0, 10.0
CHASE_R = 0.25
FLAT_R = 0.05

DEFAULTS = {
    "min_score": S.MIN_SCORE,
    "risk_pct": S.RISK_PCT,
    "entry_r_window": list(S.ENTRY_R_WINDOW),
    "factor_adj": {},
    "factor_stats": {},
    "symbol_cooldown": {},
    "consec_losses": 0,
    "chased_losses": 0,
    "shadow_adj": {},
    "calibrated_min_score": None,
    "effective_min_score": S.MIN_SCORE,
    "tp_split": None,
    "trades": 0, "wins": 0, "losses": 0, "sum_r": 0.0, "sum_pnl": 0.0,
}


class Learner:
    def __init__(self, path: Path, insight_log: Path, log_event):
        self.path, self.insight_log, self.log = path, insight_log, log_event
        self.p = {**json.loads(json.dumps(DEFAULTS))}
        if path.exists():
            self.p.update(json.loads(path.read_text()))

    def refresh_effective(self):
        cal = self.p.get("calibrated_min_score")
        own = self.p["min_score"]
        eff = own if cal is None else round((own + cal) / 2)
        self.p["effective_min_score"] = max(MIN_SCORE_FLOOR, min(MIN_SCORE_CAP, eff))

    def save(self):
        now = time.time()
        self.p["symbol_cooldown"] = {k: v for k, v in self.p["symbol_cooldown"].items() if v > now}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.p, ensure_ascii=False, indent=1))
        tmp.replace(self.path)

    def _insight(self, rec: dict):
        self.insight_log.parent.mkdir(parents=True, exist_ok=True)
        with self.insight_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _update_factors(self, factors: dict, r: float, win: bool) -> list:
        changes = []
        for tag in factors:
            st = self.p["factor_stats"].setdefault(tag, {"n": 0, "wins": 0, "sum_r": 0.0})
            st["n"] += 1
            st["wins"] += int(win)
            st["sum_r"] = round(st["sum_r"] + r, 4)
            if st["n"] >= FACTOR_MIN_N:
                new = round(max(FACTOR_ADJ_MIN, min(FACTOR_ADJ_MAX, st["sum_r"] / st["n"] * FACTOR_GAIN)), 1)
                old = self.p["factor_adj"].get(tag, 0.0)
                if abs(new - old) >= 0.5:
                    self.p["factor_adj"][tag] = new
                    changes.append(f"因子 [{tag}] 權重 {old:+.1f} → {new:+.1f} (n={st['n']}, 勝率 {st['wins'] / st['n']:.0%}, 平均 {st['sum_r'] / st['n']:+.2f}R)")
        return changes

    def on_close(self, tr: dict, r: float | None, pnl: float | None, reason: str, tps_filled: list, held_min: float):
        if r is None:
            self.log("LEARN_SKIP", trade_id=tr["id"], reason="無法取得已實現盈虧，此筆不納入學習")
            return
        factors = tr.get("factors") or {}
        entry_r = tr.get("entry_cur_r") or 0.0
        win, loss = r > FLAT_R, r < -FLAT_R
        p = self.p
        p["trades"] += 1
        p["sum_r"] = round(p["sum_r"] + r, 4)
        p["sum_pnl"] = round(p["sum_pnl"] + (pnl or 0), 4)
        changes = self._update_factors(factors, r, win)
        base = dict(trade_id=tr["id"], symbol=tr["tsym"], side=tr["side"], r_result=r, pnl_usdt=pnl,
                    path=tr.get("path"), source=tr.get("source"), score=tr.get("score"),
                    regime_at_entry=tr.get("regime"), entry_at_r=entry_r, tps_filled=tps_filled,
                    held_min=held_min, exit_reason=reason)

        if tr.get("tier") == "HIGH":
            p["high_consec_losses"] = p.get("high_consec_losses", 0) + 1 if loss else 0 if win else p.get("high_consec_losses", 0)
            if p["high_consec_losses"] >= 2 and p.get("high_tier_enabled", True):
                p["high_tier_enabled"] = False
                changes.append("高信心單連續虧損 2 筆，暫停放大部位 (恢復一般風險)")
            elif win and not p.get("high_tier_enabled", True):
                p["high_tier_enabled"] = True
                changes.append("高信心單獲利，恢復放大部位")
        if loss:
            p["losses"] += 1
            p["consec_losses"] += 1
            diagnosis = []
            reg = tr.get("regime") or [0, "?"]
            if "regime:aligned" not in factors:
                diagnosis.append(f"進場時大盤 {reg[1]} ({reg[0]:+}) 未與方向一致")
            if entry_r > CHASE_R:
                diagnosis.append(f"進場已在 {entry_r:+.2f}R，偏追價")
                p["chased_losses"] += 1
            if not tps_filled and held_min < 30:
                diagnosis.append(f"{held_min:.0f} 分鐘內直接止損，進場時機/止損過近")
            if not tps_filled and held_min >= 30:
                diagnosis.append("未達 TP1 即止損，訊號方向判斷失準")
            negs = [f"{k}({v:+})" for k, v in factors.items() if v < 0]
            if negs:
                diagnosis.append("進場時已有扣分因子: " + ", ".join(negs))
            if tps_filled:
                diagnosis.append(f"曾達 {tps_filled[-1]} 但回吐，移動止損後仍虧 (多為手續費/滑價)")

            old = p["min_score"]
            p["min_score"] = min(MIN_SCORE_CAP, old + LOSS_SCORE_STEP)
            if p["min_score"] != old:
                changes.append(f"開倉門檻 {old} → {p['min_score']}")
            sym = tr["tsym"].replace("-SWAP-USDT", "USDT")
            p["symbol_cooldown"][sym] = time.time() + LOSS_COOLDOWN_H * 3600
            changes.append(f"{sym} 冷卻 {LOSS_COOLDOWN_H} 小時")
            if p["consec_losses"] >= HALF_RISK_AFTER_LOSSES and p["risk_pct"] > S.RISK_PCT / 2:
                p["risk_pct"] = S.RISK_PCT / 2
                changes.append(f"連續虧損 {p['consec_losses']} 筆，單筆風險降為 {p['risk_pct']:.1%}")
            if entry_r > CHASE_R and p["chased_losses"] >= 2 and p["entry_r_window"][1] > 0.2:
                old_hi = p["entry_r_window"][1]
                p["entry_r_window"][1] = round(max(0.2, old_hi - 0.1), 2)
                p["chased_losses"] = 0
                changes.append(f"追價虧損累積，進場上限 {old_hi}R → {p['entry_r_window'][1]}R")
            rec = {**base, "type": "LOSS_OPTIMIZE", "diagnosis": diagnosis, "changes": changes}
            self.log("OPTIMIZE", **rec)
        elif win:
            p["wins"] += 1
            p["consec_losses"] = 0
            if p["risk_pct"] < S.RISK_PCT:
                p["risk_pct"] = S.RISK_PCT
                changes.append(f"獲利後單筆風險恢復 {S.RISK_PCT:.0%}")
            old = p["min_score"]
            p["min_score"] = max(MIN_SCORE_FLOOR, old - WIN_SCORE_STEP)
            if p["min_score"] != old:
                changes.append(f"開倉門檻 {old} → {p['min_score']}")
            pos = sorted(((k, v) for k, v in factors.items() if v > 0), key=lambda x: -x[1])
            why = []
            if pos:
                why.append("主要加分因子: " + "、".join(f"{k}({v:+})" for k, v in pos))
            reg = tr.get("regime") or [0, "?"]
            why.append(f"進場時大盤 {reg[1]} ({reg[0]:+})" + ("，順勢" if "regime:aligned" in factors else ""))
            why.append(f"進場位置 {entry_r:+.2f}R" + ("，未追價" if entry_r <= CHASE_R else "，雖有追價仍獲利"))
            if tps_filled:
                why.append(f"打到 {', '.join(tps_filled)}，分批止盈" + ("並移動止損鎖利" if len(tps_filled) < 5 else "全部完成"))
            else:
                why.append(f"未打到 TP，提前獲利出場: {reason}")
            why.append(f"持倉 {held_min:.0f} 分鐘，路徑 {tr.get('path')}，來源 {tr.get('source')}，綜合分數 {tr.get('score')}")
            rec = {**base, "type": "WIN_ANALYSIS", "why_profitable": why, "changes": changes}
            self.log("WIN_ANALYSIS", **rec)
        else:
            rec = {**base, "type": "FLAT", "note": "接近保本出場 (多為 TP1 後保本止損)", "changes": changes}
            self.log("FLAT", **rec)

        self.refresh_effective()
        p["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._insight(rec)
        self.save()

    def summary(self) -> dict:
        p = self.p
        return {k: p.get(k) for k in ("trades", "wins", "losses", "sum_r", "sum_pnl", "min_score",
                                      "calibrated_min_score", "effective_min_score", "risk_pct",
                                      "entry_r_window", "consec_losses", "tp_split", "factor_adj", "shadow_adj")}
