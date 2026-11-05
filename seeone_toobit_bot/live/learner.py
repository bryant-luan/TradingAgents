"""Post-trade learning: losses tighten the parameters, wins are explained and recorded."""
import json
import time
from pathlib import Path

from . import strategy as S

MIN_SCORE_CAP, MIN_SCORE_FLOOR = 70, 48
LOSS_SCORE_STEP, WIN_SCORE_STEP = 2, 1
LOSS_COOLDOWN_H = 12
STREAK_PAUSE_LOSSES, STREAK_PAUSE_H = 3, 6
HIGH_OFF_H, HIGH_REARM_WINS = 24, 2
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


def _aligned(tr: dict, factors: dict) -> bool:
    label = (tr.get("regime") or [0, "?"])[1]
    return "regime:aligned" in factors or (label, tr["side"]) in (("BULL", "LONG"), ("BEAR", "SHORT"))


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

    def high_tier_on(self) -> bool:
        if self.p.get("high_tier_enabled", True):
            return True
        off_at = self.p.setdefault("high_disabled_at", time.time())
        if time.time() - off_at >= HIGH_OFF_H * 3600:
            self._rearm_high(f"高信心單暫停已滿 {HIGH_OFF_H} 小時")
            return True
        return False

    def _rearm_high(self, why: str):
        self.p["high_tier_enabled"] = True
        self.p["high_consec_losses"] = 0
        self.p.pop("high_disabled_at", None)
        self.p.pop("wins_since_high_off", None)
        self.save()
        self.log("HIGH_TIER_REARM", reason=why)

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
                p["high_disabled_at"] = time.time()
                p["wins_since_high_off"] = 0
                changes.append(f"高信心單連續虧損 2 筆，暫停放大部位 {HIGH_OFF_H} 小時或待一般單連贏 {HIGH_REARM_WINS} 筆")
        if not p.get("high_tier_enabled", True):
            p["wins_since_high_off"] = p.get("wins_since_high_off", 0) + 1 if win else 0 if loss else p.get("wins_since_high_off", 0)
            if p["wins_since_high_off"] >= HIGH_REARM_WINS:
                self._rearm_high(f"高信心單暫停後連贏 {HIGH_REARM_WINS} 筆")
                changes.append(f"連贏 {HIGH_REARM_WINS} 筆，恢復高信心放大部位")
        if loss:
            p["losses"] += 1
            p["consec_losses"] += 1
            diagnosis = []
            reg = tr.get("regime") or [0, "?"]
            if not _aligned(tr, factors):
                diagnosis.append(f"進場時大盤 {reg[1]} ({reg[0]:+}) 未與方向一致")
            if entry_r > CHASE_R:
                diagnosis.append(f"進場已在 {entry_r:+.2f}R，偏追價")
                p["chased_losses"] += 1
            if not tps_filled and "重評" in (reason or ""):
                diagnosis.append(f"{held_min:.0f} 分鐘內分析驗算重評轉向，提前出場 (未打到止損，減少虧損)")
            elif not tps_filled and held_min < 30:
                diagnosis.append(f"{held_min:.0f} 分鐘內直接止損，進場時機/止損過近")
            elif not tps_filled:
                diagnosis.append("未達 TP1 即止損，訊號方向判斷失準")
            m15, sgn = tr.get("m15_at_entry"), 1 if tr["side"] == "LONG" else -1
            if m15 is not None and m15 * sgn >= self.p.get("m15_max", S.M15_MAX_DEFAULT):
                diagnosis.append(f"進場時 See One 15m 方向已達 {m15:+.1f}，順勢追在走勢末端")
            htf = tr.get("htf_at_entry")
            if htf is not None and htf * sgn < 0:
                diagnosis.append(f"進場時 1h/4h 大方向 {htf:+.1f}，屬逆大勢交易")
            rpos = tr.get("range_pos_at_entry")
            if rpos is not None and rpos > 1.0:
                diagnosis.append(f"進場價已超出近 3 小時區間 (位置 {rpos:.2f})，追在突破延伸段")
            rsi = tr.get("rsi15_at_entry")
            if rsi is not None and rsi < 50:
                diagnosis.append(f"進場時 15m RSI 在交易方向只有 {rsi:.0f}，短線動能與方向相反")
            h4 = (tr.get("dirs_at_entry") or {}).get("4h")
            if h4 is not None and h4 * sgn < 0:
                diagnosis.append(f"See One 4h 方向 {h4:+.1f} 與{'做多' if sgn > 0 else '做空'}相反")
            ext = tr.get("ext_r_at_entry")
            if ext is not None and ext >= 2.0:
                diagnosis.append(f"進場前 2 小時本幣已順方向走 {ext:+.1f}R，追在走勢末端")
            if tr.get("same_side_at_entry"):
                diagnosis.append(f"進場時已有 {tr['same_side_at_entry']} 筆同方向持倉，風險集中")
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
            if p["consec_losses"] >= STREAK_PAUSE_LOSSES:
                p["pause_until"] = time.time() + STREAK_PAUSE_H * 3600
                changes.append(f"連續虧損 {p['consec_losses']} 筆，暫停開新倉 {STREAK_PAUSE_H} 小時 (不縮小部位)")
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
            old = p["min_score"]
            p["min_score"] = max(MIN_SCORE_FLOOR, old - WIN_SCORE_STEP)
            if p["min_score"] != old:
                changes.append(f"開倉門檻 {old} → {p['min_score']}")
            pos = sorted(((k, v) for k, v in factors.items() if v > 0), key=lambda x: -x[1])
            why = []
            if pos:
                why.append("主要加分因子: " + "、".join(f"{k}({v:+})" for k, v in pos))
            reg = tr.get("regime") or [0, "?"]
            why.append(f"進場時大盤 {reg[1]} ({reg[0]:+})" + ("，順勢" if _aligned(tr, factors) else ""))
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
