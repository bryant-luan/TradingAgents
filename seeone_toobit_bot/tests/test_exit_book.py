"""Exit bookkeeping. A vanished take-profit order is not a fill, and a flat venue is not a new loss."""
import tempfile
import threading
import unittest
from pathlib import Path

from live import bot as botmod
from live import strategy as S
from live.bot import Bot, filled_margin, position_gone, position_pnl, remaining_contracts, tp_fill_verdict, tp_still_ahead
from live.gate import Gate, GateError, price_order_view
from live.toobit import ToobitError


class FullLive(unittest.TestCase):
    def test_measured_short_stays_live(self):
        self.assertTrue(S.self_short_full_live("SHORT", 0.4, 0.008, 0.62, 1.0, 0.05))

    def test_three_losses_raise_the_live_bar(self):
        losses = [{"ev": 0.44}, {"ev": 0.44}, {"ev": 0.50}]
        self.assertEqual(S.roll_ev_floor(0.50, losses), 0.51)
        # All three were already under the line. The bar still steps up.
        self.assertEqual(S.roll_ev_floor(0.50, [{"ev": 0.32}, {"ev": 0.27}, {"ev": 0.37}]), 0.55)
        self.assertEqual(S.stuck_floor_step(0.50, [{"ev": 0.37}, {"ev": 0.27}, {"ev": 0.32}], 0.50), 0.55)
        self.assertIsNone(S.stuck_floor_step(0.50, [{"ev": 0.44}, {"ev": 0.50}, {"ev": 0.73}], 0.74))
        self.assertFalse(S.self_short_full_live("SHORT", 0.4, 0.008, 0.50, 1.0, 0.05, 0.51))
        self.assertTrue(S.self_short_full_live("SHORT", 0.4, 0.008, 0.73, 1.0, 0.05, 0.51))

    def test_three_straight_losses_stop_that_rule(self):
        rules = {}
        key = S.rule_key("IGNITION 15m", "LONG", True)
        self.assertIsNone(S.roll_rule(rules, key, -1.0, "AGT"))
        self.assertIsNone(S.roll_rule(rules, key, 0.4, "KAVA"))
        self.assertEqual(rules[key]["streak"], 0)
        self.assertIsNone(S.roll_rule(rules, key, -1.0, "TON"))
        self.assertIsNone(S.roll_rule(rules, key, -1.0, "BTW"))
        note = S.roll_rule(rules, key, -1.0, "BULLA")
        self.assertIn("停止新開倉", note)
        self.assertFalse(S.rule_open(rules, key))
        self.assertIsNone(S.roll_rule(rules, key, 1.2, "EVAA"))
        self.assertFalse(S.rule_open(rules, key))

    def test_a_live_self_short_keeps_its_bar_instead_of_stopping(self):
        closes = [
            {"paper": False, "path": "SELF ANALYSIS", "side": "SHORT", "symbol": "LTC", "r": -0.93},
            {"paper": False, "path": "SELF ANALYSIS", "side": "SHORT", "symbol": "BONK", "r": -1.48},
            {"paper": False, "path": "SELF ANALYSIS", "side": "SHORT", "symbol": "COTI", "r": -0.87},
            {"paper": True, "path": "SELF ANALYSIS", "side": "SHORT", "symbol": "CRV", "r": -0.42},
            {"paper": True, "path": "SELF ANALYSIS", "side": "SHORT", "symbol": "BMT", "r": -0.53},
            {"paper": True, "path": "SELF ANALYSIS", "side": "SHORT", "symbol": "0G", "r": -0.59},
        ]
        rules = S.replay_rule_streaks(closes)
        self.assertNotIn(S.rule_key("SELF ANALYSIS", "SHORT", False), rules)
        self.assertFalse(S.rule_open(rules, S.rule_key("SELF ANALYSIS", "SHORT", True)))
        reopened = S.replay_rule_streaks(closes + [
            {"paper": True, "path": "SELF ANALYSIS", "side": "SHORT", "symbol": "KAVA", "r": 1.5},
        ])
        self.assertTrue(S.rule_open(reopened, S.rule_key("SELF ANALYSIS", "SHORT", True)))

    def test_a_win_steps_the_bar_back(self):
        self.assertEqual(S.ease_ev_floor(0.51, 0.50), 0.50)
        self.assertEqual(S.ease_ev_floor(0.62, 0.50), 0.57)

    def test_gate_book_is_larger_and_toobit_stays(self):
        gate, toobit = S.book_limits("gate"), S.book_limits("toobit")
        self.assertEqual(gate["risk"], 0.12)
        self.assertEqual(gate["short_risk"], 0.12)
        self.assertGreaterEqual(gate["margin"], 0.48)
        self.assertGreater(gate["total_margin"], toobit["total_margin"])
        self.assertEqual(toobit["risk"], S.MIN_RISK_PCT)
        self.assertEqual(toobit["short_risk"], S.SELF_SHORT_RISK_PCT)
        spec = {"multiplier": 1.0, "min_contracts": 1, "max_leverage": 25}
        sized = S.size_position(1000, 100, 99, spec, gate["risk"], gate["margin"])
        self.assertIsNotNone(sized)
        self.assertGreaterEqual(sized["risk_usdt"], 1000 * 0.12 * 0.95)

    def test_open_paper_scales_once_without_changing_r(self):
        trades = {"a": {
            "tsym": "WLD-SWAP-USDT", "paper": True, "paper_probe": True,
            "contracts": 595, "risk_usdt": 7.7945, "margin": 14.07,
            "tps": [{"qty": 59}, {"qty": 59}, {"qty": 119}, {"qty": 178}, {"qty": 180}],
        }, "b": {
            "tsym": "LIVE-SWAP-USDT", "paper": False, "contracts": 100, "risk_usdt": 8.0,
        }}
        names = S.scale_open_paper(trades, 1.6)
        self.assertEqual(names, ["WLD-SWAP-USDT"])
        self.assertEqual(trades["a"]["contracts"], 952)
        self.assertEqual(sum(tp["qty"] for tp in trades["a"]["tps"]), 952)
        self.assertAlmostEqual(trades["a"]["risk_usdt"] / 7.7945, 952 / 595, places=4)
        self.assertEqual(trades["b"]["contracts"], 100)
        self.assertEqual(S.scale_open_paper(trades, 1.0), [])

    def test_a_narrow_stop_below_the_floor_is_not_a_live_order(self):
        self.assertFalse(S.self_short_sample_live(0.008))
        self.assertFalse(S.self_short_sample_live(0.012))
        self.assertEqual(S.live_ev_floor(0.55), S.SELF_SHORT_FULL_RISK_EV)
        self.assertEqual(S.live_ev_floor(0.74), 0.74)
        self.assertFalse(S.self_short_full_live("SHORT", 0.4, 0.008, 0.55, 1.0, 0.05))
        self.assertTrue(S.self_short_full_live("SHORT", 0.4, 0.008, 0.62, 1.0, 0.05))
        self.assertTrue(S.ignition_15m_open("toobit"))
        self.assertTrue(S.ignition_15m_open("gate"))
        self.assertFalse(S.ignition_live("gate", "15m"))
        self.assertFalse(S.ignition_live("gate", "1m"))
        self.assertFalse(S.ignition_live("toobit", "15m"))
        self.assertFalse(S.ignition_live("toobit", "1m"))
        self.assertTrue(S.IGNITION_1M)
        self.assertTrue(S.BREAKOUT_ENTRIES)

    def test_a_below_floor_short_papers_until_it_is_positive(self):
        self.assertTrue(S.SELF_ANALYSIS_PAPER_OPEN)
        rules = {S.rule_key("SELF FLUSH LONG", "LONG", False): {"open": False, "streak": 3, "last": []}}
        self.assertTrue(S.paper_for_stopped_live(rules, "SELF FLUSH LONG", "LONG", False))
        self.assertFalse(S.paper_for_stopped_live(rules, "SELF FLUSH LONG", "LONG", True))
        self.assertFalse(S.flush_live("toobit"))
        self.assertFalse(S.flush_live("gate"))
        self.assertIn("2%", S.gate_flush_wide_stop("gate", 0.045))
        self.assertIsNone(S.gate_flush_wide_stop("gate", 0.015))
        self.assertIsNone(S.gate_flush_wide_stop("toobit", 0.045))
        self.assertTrue(S.gate_flush_goes_live("gate", 0.015))
        self.assertFalse(S.gate_flush_goes_live("gate", 0.02))
        self.assertFalse(S.gate_flush_goes_live("toobit", 0.015))
        self.assertTrue(S.toobit_mid_short_live("toobit", "SHORT", 0.57))
        self.assertFalse(S.toobit_mid_short_live("toobit", "SHORT", 0.54))
        self.assertFalse(S.toobit_mid_short_live("toobit", "SHORT", 0.65))
        self.assertFalse(S.toobit_mid_short_live("gate", "SHORT", 0.57))
        self.assertTrue(S.toobit_long_live("toobit", "LONG", 0.03))
        self.assertFalse(S.toobit_long_live("toobit", "LONG", 0.029))
        self.assertFalse(S.toobit_long_live("gate", "LONG", 0.04))
        self.assertTrue(S.toobit_wide_long_review("toobit", 0.04, 0.20))
        self.assertTrue(S.toobit_wide_long_review("toobit", 0.04, 0.70))
        self.assertFalse(S.toobit_wide_long_review("toobit", 0.04, 0.76))
        self.assertEqual(S.GATE_SOFT_SHORT_MIN_PROB, 50.0)
        self.assertEqual(S.TOOBIT_LONG_LIVE_MIN_EV, 0.25)
        self.assertEqual(S.LIVE_BAND_CONFIRM_MIN_PROB, 45.0)
        self.assertEqual(S.SELF_SHORT_MAX_SAME_SIDE, 5)
        self.assertEqual(S.MAX_SAME_SIDE, 4)
        self.assertEqual(S.MAX_POSITIONS, 5)
        self.assertEqual(S.BTC1H_CHASE_MAX, 0.40)
        self.assertTrue(S.gate_soft_short_live("gate", "SHORT", 0.20))
        self.assertFalse(S.gate_soft_short_live("gate", "SHORT", 0.14))
        self.assertFalse(S.gate_soft_short_live("gate", "SHORT", 0.25))
        self.assertFalse(S.gate_soft_short_live("toobit", "SHORT", 0.20))
        self.assertIsNone(S.gate_measured_loss("gate", "SELF ANALYSIS", "SHORT", False, 0.20))
        self.assertFalse(S.toobit_wide_long_review("toobit", 0.07, 0.20))
        self.assertFalse(S.toobit_wide_long_review("gate", 0.04, 0.20))
        self.assertIsNone(S.gate_measured_loss("toobit", "SELF ANALYSIS", "SHORT", False))
        self.assertIsNone(S.gate_measured_loss("gate", "SELF ANALYSIS", "SHORT", True))
        self.assertIsNone(S.gate_measured_loss("gate", "QUIET SHORT", "SHORT", False))
        self.assertIsNone(S.gate_measured_loss("gate", "SELF FLUSH LONG", "LONG", False))
        self.assertIn("不再開", S.gate_measured_loss("gate", "SELF ANALYSIS", "SHORT", False, 0.40))
        self.assertIn("不再開", S.gate_measured_loss("gate", "SELF ANALYSIS", "LONG", False))
        self.assertIn("不再開", S.gate_measured_loss("gate", "IGNITION 1m", "LONG", False))
        self.assertIsNone(S.gate_measured_loss("gate", "IGNITION 15m", "LONG", False))
        self.assertTrue(S.NEW_ENTRIES_PAPER)
        self.assertFalse(S.toobit_open_now_live("toobit", "SHORT", 0.20))
        self.assertFalse(S.toobit_open_now_live("toobit", "SHORT", 0.20, 54))
        self.assertFalse(S.toobit_open_now_live("toobit", "SHORT", 0.20, 55))
        self.assertIsNone(S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.20, False, 54))
        self.assertIn("不再開", S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.20, False, 56))
        self.assertFalse(S.toobit_open_now_live("toobit", "SHORT", 0.40))
        self.assertFalse(S.toobit_open_now_live("toobit", "SHORT", 0.52))
        self.assertFalse(S.toobit_open_now_live("toobit", "SHORT", 0.55))
        self.assertFalse(S.toobit_open_now_live("gate", "SHORT", 0.20))
        self.assertIsNone(S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.20, False))
        self.assertIn("不再開", S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.40, False))
        self.assertIn("不再開", S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.52, False))
        self.assertIsNone(S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.55, False))
        self.assertIn("不再開", S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.65, False))
        self.assertIsNone(S.self_short_chased("SHORT", 0.99))
        self.assertIn("不再開", S.self_short_chased("SHORT", 1.0))
        self.assertIsNone(S.self_short_chased("LONG", 2.0))
        self.assertIsNone(S.self_short_chased("SHORT", None))
        self.assertIn("不再開", S.self_short_chased("SHORT", 1.5))
        slots = [
            {"paper_probe": True, "path": "IGNITION 1m"},
            {"paper_probe": True, "path": "IGNITION 1m"},
            {"paper_probe": True, "path": "IGNITION 15m"},
            {"paper_probe": True, "path": "IGNITION 15m", "exit_reason": "done"},
        ]
        self.assertEqual(S.ignition_open_slots(slots, "IGNITION 15m"), 1)
        self.assertEqual(S.ignition_open_slots(slots, "IGNITION 1m"), 2)
        self.assertIn("不再開", S.self_short_chased("SHORT", 1.72))
        self.assertIn("不再開", S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.65, True))
        self.assertIsNone(S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.70, True))
        self.assertIsNone(S.toobit_measured_loss("toobit", "SELF ANALYSIS", "SHORT", 0.40, True))
        self.assertIsNone(S.toobit_measured_loss("toobit", "QUIET SHORT", "SHORT", 0.2, False))
        self.assertIn("不再開", S.toobit_measured_loss("toobit", "IGNITION 1m", "LONG", None, False))
        self.assertIn("不再開", S.toobit_measured_loss("toobit", "IGNITION 15m", "LONG", None, False))
        self.assertIn("不再開", S.toobit_measured_loss("toobit", "SELF FLUSH LONG", "LONG", None, False))
        self.assertIn("不再開", S.toobit_measured_loss("toobit", "BREAKOUT", "LONG", None, False))
        self.assertIsNone(S.toobit_measured_loss("toobit", "BREAKOUT", "SHORT", None, False))
        self.assertIsNone(S.toobit_measured_loss("gate", "IGNITION 1m", "LONG", None, False))

    def test_other_slices_paper(self):
        self.assertFalse(S.self_short_full_live("LONG", 0.4, 0.008, 0.62, 1.0, 0.05))
        self.assertFalse(S.self_short_full_live("SHORT", -0.4, 0.008, 0.62, 1.0, 0.05))
        self.assertFalse(S.self_short_full_live("SHORT", None, 0.008, 0.62, 1.0, 0.05))
        self.assertFalse(S.self_short_full_live("SHORT", 0.4, 0.012, 0.62, 1.0, 0.05))
        self.assertFalse(S.self_short_full_live("SHORT", 0.4, 0.008, 0.30, 1.0, 0.05))
        self.assertFalse(S.self_short_full_live("SHORT", 0.4, 0.008, 0.62, 4.0, 0.05))
        self.assertFalse(S.self_short_full_live("SHORT", 0.4, 0.008, 0.62, 1.0, 0.20))


class Verdict(unittest.TestCase):
    def test_filled_only_when_the_order_says_so(self):
        self.assertEqual(tp_fill_verdict("FILLED", 0, 10), "filled")
        self.assertEqual(tp_fill_verdict("CANCELED", 10, 10), "filled")
        self.assertEqual(tp_fill_verdict("CANCELED", 0, 10), "cancelled")
        self.assertEqual(tp_fill_verdict("REJECTED", 0, 10), "cancelled")
        self.assertEqual(tp_fill_verdict("NEW", 0, 10), "pending")
        self.assertEqual(tp_fill_verdict(None, None, 10), "pending")

    def test_position_gone_matches_both_venues(self):
        self.assertTrue(position_gone(ToobitError("{'code': -3145, 'msg': 'No position, Please confirm'}")))
        self.assertTrue(position_gone(ToobitError("{'label': 'AUTO_USER_NO_SHORT_POSITION'}")))
        self.assertFalse(position_gone(ToobitError("ReadTimeout")))

    def test_restore_only_while_the_target_is_still_ahead(self):
        self.assertTrue(tp_still_ahead("SHORT", 1.0, 0.97))
        self.assertFalse(tp_still_ahead("SHORT", 0.90, 0.97))
        self.assertTrue(tp_still_ahead("LONG", 1.0, 1.03))
        self.assertFalse(tp_still_ahead("LONG", 1.05, 1.03))


class FakeBook:
    def __init__(self):
        self.orders = {}
        self.rows = []
        self.stops = []
        self.closed = []
        self.placed = []
        self.px = 1.0
        self._n = 0

    def order(self, oid):
        return self.orders[str(oid)]

    def open_orders(self, symbol, kind="STOP"):
        return [{"orderId": oid} for oid, k in self.placed if k == kind]

    def post_only(self, symbol, side, contracts, price, tag):
        self._n += 1
        oid = f"new{self._n}"
        self.placed.append((oid, "LIMIT"))
        self.orders[oid] = {"orderId": oid, "status": "NEW", "executedQty": 0}
        return {"orderId": oid}

    def positions(self, symbol):
        return list(self.rows)

    def history_positions(self, symbol, limit=5):
        return list(getattr(self, "history", []))

    def user_trades(self, symbol, limit=100):
        return []

    def set_position_stop(self, symbol, side, stop):
        if not any(p.get("side") == side for p in self.rows):
            raise ToobitError("{'code': -3145, 'msg': 'No position'}")
        self.stops.append(stop)

    def fmt_price(self, symbol, price):
        return f"{float(price):.6f}"

    def price(self, symbol):
        return self.px

    def flash_close(self, symbol, side):
        self.closed.append((symbol, side))

    def cancel_order(self, order_id, kind="STOP"):
        return None


def bot(book):
    b = Bot.__new__(Bot)
    b.tb = book
    b.st = {"trades": {}, "closed": [], "day_lock_hit": False}
    b._protect_lock = threading.Lock()
    b.paper = False
    b.htf = None
    return b


def trade():
    return {
        "id": "CGPT-SWAP-USDT:SHORT:1", "tsym": "CGPT-SWAP-USDT", "side": "SHORT",
        "entry": 1.0, "stop": 1.01, "init_dist": 0.01, "contracts": 100, "risk_usdt": 8.0,
        "mfe_r": 0.0, "path": "SELF ANALYSIS", "source": "測試", "signal_id": "self:test",
        "tp_migrated": True, "tps": [
            {"label": "TP1", "kind": "LIMIT", "order_id": "a", "qty": 10, "price": 0.99, "r": 1.0, "filled": False},
            {"label": "TP3", "kind": "LIMIT", "order_id": "c", "qty": 30, "price": 0.97, "r": 2.0, "filled": False},
            {"label": "TP5", "kind": "LIMIT", "order_id": "e", "qty": 40, "price": 0.95, "r": 4.0, "filled": False},
        ],
    }


class Manage(unittest.TestCase):
    def setUp(self):
        self._prev = botmod.TRADE_LOG
        self._tmp = tempfile.NamedTemporaryFile(prefix="exit-book-", suffix=".jsonl", delete=False)
        botmod.TRADE_LOG = Path(self._tmp.name)

    def tearDown(self):
        botmod.TRADE_LOG = self._prev
        Path(self._tmp.name).unlink(missing_ok=True)

    def test_cancelled_targets_are_not_fills(self):
        book = FakeBook()
        book.rows = [{"side": "SHORT", "position": "100"}]
        book.orders = {
            "a": {"status": "FILLED", "executedQty": 10},
            "c": {"status": "CANCELED", "executedQty": 0},
            "e": {"status": "NEW", "executedQty": 0},
        }
        b = bot(book)
        tr = trade()
        tr["exchange_flat"] = True
        b.st["trades"] = {"1": tr}
        b.manage({("CGPT-SWAP-USDT", "SHORT"): book.rows[0]}, {})
        self.assertFalse(tr.get("exchange_flat"))
        self.assertTrue(tr["tps"][0]["filled"])
        # Cancelled, still ahead of the price, and posted once. It is not a fill.
        self.assertFalse(tr["tps"][1].get("cancelled"))
        self.assertFalse(tr["tps"][1].get("filled"))
        self.assertEqual(tr["tps"][1].get("restore_n"), 1)
        self.assertEqual(tr["tps"][1].get("order_id"), "new1")
        self.assertFalse(tr["tps"][2].get("filled"))
        self.assertFalse(tr["tps"][2].get("cancelled"))
        # One real TP1 does move the post-TP stop. The cancelled targets do not count as a full take-profit.
        self.assertTrue(book.stops)
        self.assertFalse(all(tp.get("filled") for tp in tr["tps"]))
        b.manage({("CGPT-SWAP-USDT", "SHORT"): book.rows[0]}, {})
        self.assertEqual([oid for oid, _ in book.placed], ["new1"])

    def test_a_live_name_is_not_kept_when_the_venue_is_flat(self):
        b = bot(FakeBook())
        live = trade()
        paper = trade()
        paper.update(id="2", tsym="LDO-SWAP-USDT", paper=True)
        b.st["trades"] = {"1": live, "2": paper}
        self.assertEqual(list(b._with_paper({})), [("LDO-SWAP-USDT", "SHORT")])

    def test_a_flat_live_book_is_recorded_from_the_fill(self):
        book = FakeBook()
        book.history = [{"side": "SHORT", "closeTime": 1_700_000_000_000, "realizedPnL": 1.07, "closeAvgPrice": 0.99}]
        b = bot(book)
        tr = trade()
        tr["opened_ts"] = 1_700_000_000
        tr["tps"] = []
        b.st["trades"] = {"1": tr}
        b.learner = type("L", (), {"on_close": staticmethod(lambda *a, **k: None)})()
        saved = botmod.save_state
        botmod.save_state = lambda st: None
        try:
            b.manage({}, {})
        finally:
            botmod.save_state = saved
        self.assertNotIn("1", b.st["trades"])
        self.assertEqual(b.st["closed"][-1]["r_result"], 0.134)
        self.assertEqual(book.closed, [])

    def test_a_missed_snapshot_does_not_close_a_live_position(self):
        book = FakeBook()
        book.rows = [{"side": "SHORT", "position": "100"}]
        b = bot(book)
        tr = trade()
        tr["exchange_flat"] = True
        b.st["trades"] = {"1": tr}
        b.manage({}, {})
        self.assertIn("1", b.st["trades"])
        self.assertFalse(tr.get("exchange_flat"))

    def test_a_previous_round_trip_is_not_this_trades_pnl(self):
        opened = 1_700_000_000
        rows = [
            {"side": "SHORT", "openTime": (opened - 3600) * 1000, "closeTime": (opened - 30) * 1000,
             "realizedPnL": 20.0, "closeAvgPrice": 0.9},
            {"side": "SHORT", "openTime": (opened - 8) * 1000, "closeTime": (opened + 40) * 1000,
             "realizedPnL": 1.07, "closeAvgPrice": 0.99},
            {"side": "LONG", "openTime": (opened - 8) * 1000, "closeTime": (opened + 40) * 1000,
             "realizedPnL": 9.0, "closeAvgPrice": 1.1},
        ]
        pnl, px = position_pnl(rows, "SHORT", opened)
        self.assertEqual(pnl, 1.07)
        self.assertEqual(px, 0.99)

    def test_adopted_position_is_not_dropped(self):
        opened = 1_700_000_000
        rows = [
            {"side": "SHORT", "openTime": (opened - 3600) * 1000, "closeTime": (opened - 30) * 1000,
             "realizedPnL": 20.0, "closeAvgPrice": 0.9},
            {"side": "SHORT", "openTime": (opened - 7200) * 1000, "closeTime": (opened + 90) * 1000,
             "realizedPnL": 1.07, "closeAvgPrice": 0.99},
        ]
        pnl, px = position_pnl(rows, "SHORT", opened)
        self.assertEqual(pnl, 1.07)
        self.assertEqual(px, 0.99)

    def test_partial_closes_of_the_same_position_add_up(self):
        opened = 1_700_000_000
        open_ms = (opened - 5) * 1000
        rows = [
            {"side": "SHORT", "openTime": open_ms, "closeTime": (opened + 10) * 1000,
             "realizedPnL": 0.4, "closeAvgPrice": 0.99},
            {"side": "SHORT", "openTime": open_ms + 500, "closeTime": (opened + 80) * 1000,
             "realizedPnL": 0.67, "closeAvgPrice": 0.97},
        ]
        pnl, px = position_pnl(rows, "SHORT", opened)
        self.assertEqual(round(pnl, 2), 1.07)
        self.assertEqual(px, 0.97)

    def test_filled_targets_leave_the_book_risk(self):
        tr = trade()
        tr["tps"][0]["filled"] = True
        self.assertEqual(remaining_contracts(tr), 90)

    def test_partial_fill_keeps_its_share_of_margin(self):
        self.assertEqual(filled_margin(353265, 32791, 80.0), round(80.0 * 32791 / 353265, 4))
        self.assertEqual(filled_margin(100, 100, 8.0), 8.0)

    def test_a_gate_trigger_counts_when_it_fired(self):
        self.assertEqual(price_order_view({"id": "9", "status": "finished", "finish_as": "succeeded",
                                            "initial": {"size": -30}})["status"], "FILLED")
        self.assertEqual(price_order_view({"id": "9", "status": "finished", "finish_as": "cancelled",
                                            "initial": {"size": -30}})["status"], "CANCELED")
        self.assertEqual(price_order_view({"id": "9", "status": "open", "initial": {"size": 30}})["status"], "NEW")

    def test_a_rejected_stop_puts_the_previous_one_back(self):
        g = Gate.__new__(Gate)
        g._hedge = False
        seen = []

        def open_orders(symbol, kind="STOP"):
            return [{"orderId": "old", "orderType": "STOP_SHORT_LOSS", "stopPrice": "1.02"}]

        def cancel_order(oid, kind="STOP"):
            seen.append(("cancel", oid))

        def _request(method, path, params=None, body=None):
            price = body["trigger"]["price"]
            seen.append(("post", price))
            if price == "1.01":
                raise GateError("POST /futures/usdt/price_orders -> 400 rejected")
            return {"id": "new"}

        g.open_orders = open_orders
        g.cancel_order = cancel_order
        g._request = _request
        with self.assertRaises(GateError):
            g.set_position_stop("X-SWAP-USDT", "SHORT", "1.01")
        self.assertEqual(seen, [("cancel", "old"), ("post", "1.01"), ("post", "1.02")])

    def test_flat_book_does_not_invent_a_loss(self):
        book = FakeBook()
        book.px = 1.009  # -0.9R on the short, past the -0.85 line
        b = bot(book)
        tr = trade()
        tr["tps"] = []
        b.protect(tr)
        self.assertIsNone(tr.get("exit_reason"))
        self.assertEqual(book.closed, [])
        book.stops.clear()
        b.replace_stop(tr, 1.008, "again")
        self.assertTrue(tr.get("exchange_flat"))
        self.assertEqual(book.stops, [])

    def test_open_loser_keeps_the_exchange_stop(self):
        book = FakeBook()
        book.px = 1.009  # -0.9R on the short, past the old -0.85 market line
        book.rows = [{"side": "SHORT", "position": "100"}]
        b = bot(book)
        tr = trade()
        tr["tps"] = []
        b.protect(tr)
        self.assertIsNone(tr.get("exit_reason"))
        self.assertEqual(book.closed, [])

    def test_live_spike_keeps_the_exchange_stop(self):
        book = FakeBook()
        book.px = 1.0045  # -0.45R, a 0.56R drop from the previous print
        book.rows = [{"side": "SHORT", "position": "100"}]
        b = bot(book)
        tr = trade()
        tr["_prev_r"] = 0.11
        tr["tps"] = []
        b.protect(tr)
        self.assertIsNone(tr.get("exit_reason"))
        self.assertEqual(book.closed, [])

    def test_paper_spike_stays_with_the_stop(self):
        book = FakeBook()
        book.px = 1.0045
        b = bot(book)
        tr = trade()
        tr["paper"] = True
        tr["_prev_r"] = 0.11
        tr["tps"] = []
        b.protect(tr)
        self.assertIsNone(tr.get("exit_reason"))
        self.assertEqual(book.closed, [])

    def test_a_wick_through_the_deep_line_waits_for_the_next_read(self):
        book = FakeBook()
        book.px = 1.009  # -0.9R, a 0.7R drop from the previous print
        b = bot(book)
        tr = trade()
        tr["paper"] = True
        tr["_prev_r"] = -0.2
        tr["tps"] = []
        b.protect(tr)
        self.assertIsNone(tr.get("exit_reason"))
        tr["_prev_r"] = -0.9
        b.protect(tr)
        self.assertTrue(str(tr.get("exit_reason") or "").startswith("浮虧"))

    def test_paper_loser_still_flattens(self):
        book = FakeBook()
        book.px = 1.009
        b = bot(book)
        tr = trade()
        tr["paper"] = True
        tr["tps"] = []
        b.protect(tr)
        self.assertTrue(str(tr.get("exit_reason") or "").startswith("浮虧"))

    def test_close_of_an_already_flat_book_drops_the_sentence(self):
        book = FakeBook()
        b = bot(book)
        tr = trade()
        tr["exit_reason"] = "浮虧 -0.53R 已深於 -0.5R，先平倉，避免插針滑過止損"
        tr["lock_r"] = 1.0
        b.maker_close(tr, urgent=True)
        self.assertIsNone(tr.get("exit_reason"))
        self.assertTrue(tr.get("exchange_flat"))
        self.assertEqual(book.closed, [])

    def test_a_live_ignition_keeps_the_base_stop_and_does_not_chase(self):
        book = FakeBook()
        b = bot(book)
        sig = {"symbol": "AAAUSDT", "entry_price": 1.0, "stop_price": 0.90,
               "levels": [{"label": "TP1", "r_multiple": 1.0, "price": 1.1}]}
        book.px = 1.05
        filled, why = b._ignition_live_fill(sig)
        self.assertIsNone(why)
        self.assertEqual(filled["entry_price"], 1.05)
        self.assertEqual(filled["stop_price"], 0.90)
        self.assertAlmostEqual(filled["levels"][0]["price"], 1.05 + 0.15)
        self.assertFalse(filled["paper_only"])
        book.px = 1.20
        filled, why = b._ignition_live_fill(sig)
        self.assertIn("不追", why)
        book.px = 0.89
        filled, why = b._ignition_live_fill(sig)
        self.assertIn("穿越止損", why)

    def test_a_stock_order_uses_the_cash_price(self):
        names = frozenset({"NANYA", "NVDA", "TQQQX"})
        self.assertTrue(S.is_stock_contract("NANYA-SWAP-USDT", names))
        self.assertTrue(S.is_stock_contract("TQQQX_USDT", names))
        self.assertFalse(S.is_stock_contract("BTC-SWAP-USDT", names))
        short = {"path": "QUIET SHORT", "side": "SHORT", "r_result": 1.0, "closed_at": "2026-11-14T00:00:00+00:00"}
        ready = S.path_sample([dict(short, r_result=0.6, closed_at=f"2026-11-{i:02d}T00:00:00+00:00")
                               for i in range(1, 21)], "QUIET SHORT", "SHORT")
        self.assertTrue(ready["ready"])
        self.assertTrue(S.clears_live_target(ready))
        exact = dict(ready, win=0.55, mean_r=0.30)
        self.assertFalse(S.clears_live_target(exact))
        thin = dict(ready, n=7)
        self.assertFalse(S.clears_live_target(thin))
        stocked = S.path_sample([dict(short, stock=True)], "QUIET SHORT", "SHORT")
        self.assertEqual(stocked["n"], 0)
        anchored, why = S.anchor_stock_prices(
            9.861, 10.302, [{"label": "TP1", "price": 9.42}], 9.971, 9.90)
        self.assertIsNone(why)
        self.assertAlmostEqual(anchored["entry"], 9.971)
        self.assertAlmostEqual(anchored["stop"], 10.302 + (9.971 - 9.861))
        self.assertAlmostEqual(anchored["levels"][0]["price"], 9.42 + (9.971 - 9.861))
        self.assertIn("9.971", anchored["note"])
        missed, why = S.anchor_stock_prices(9.861, 10.302, [], 9.971, 9.80)
        self.assertIsNone(missed)
        self.assertIn("偏離股票價格", why)
        missed, why = S.anchor_stock_prices(9.861, 10.302, [], None, 9.90)
        self.assertIsNone(missed)
        self.assertIn("讀不到", why)
        self.assertIsNone(S.ignition_setup(self._ignition_bars(102.0), "15m", now_ms=10**15))
        hit = S.ignition_setup(self._ignition_bars(100.5), "15m", now_ms=10**15)
        self.assertIsNotNone(hit)
        self.assertLess(hit["break_pct"], 0.01)
        self.assertLess(hit["vol_mult"], 4.5)
        self.assertIsNone(S.ignition_setup(self._ignition_bars(100.5, volume=45.0), "15m", now_ms=10**15))
        self.assertTrue(S.STOCK_PAPER)
        slots = [
            {"paper_probe": True, "path": "IGNITION 15m", "stock": True},
            {"paper_probe": True, "path": "IGNITION 15m"},
            {"paper_probe": True, "path": "DAILY TREND", "stock": True, "exit_reason": "done"},
        ]
        self.assertEqual(S.stock_open_slots(slots), 1)
        self.assertEqual(S.ignition_open_slots(slots, "IGNITION 15m"), 1)
        shorts = [{"paper_probe": True, "stock": True, "side": "SHORT", "path": "DAILY TREND"} for _ in range(4)]
        self.assertIn("同一方向", S.stock_slot_block(shorts, "SHORT"))
        self.assertIsNone(S.stock_slot_block(shorts, "LONG"))
        self.assertIsNone(S.stock_slot_block(shorts + [{"paper_probe": True, "side": "LONG"}], "LONG"))
        both = shorts[:2] + [{"paper_probe": True, "stock": True, "side": "LONG"} for _ in range(2)]
        self.assertIn("同一方向", S.stock_slot_block(both, "LONG"))
        self.assertIn("同一方向", S.stock_slot_block(both, "SHORT"))
        self.assertIsNone(S.stock_slot_block([{**t, "exit_reason": "done"} for t in shorts], "SHORT"))
        self.assertEqual(S.LIVE_WIN, 0.55)
        self.assertEqual(S.LIVE_R, 0.30)
        self.assertEqual(S.QUIET_LONG_READY_N, 8)
        self.assertEqual(S.IGNITION_PAPER_MAX, 12)
        self.assertEqual(S.TREND_PAPER_MAX, 8)

    @staticmethod
    def _ignition_bars(close: float, volume: float = 40.0):
        bars = []
        for i in range(8):
            bars.append([i * 900_000, 99.5, 100.0, 99.0, 99.6, 10.0])
        bars.append([8 * 900_000, 100.0, close, 99.4, close, volume])
        return bars

    def test_a_cleared_paper_book_goes_live(self):
        book = FakeBook()
        book.px = 100.0
        b = bot(book)
        b.note_skip = lambda *a, **k: None
        b.st["traded_ids"] = []
        opened = []
        b.open_trade = lambda sig, ev, src, reg, eq: opened.append(bool(sig.get("paper_only"))) or True
        prev = S.stock_bases
        S.stock_bases = lambda now=None: frozenset()

        def closes(path, side, n):
            return [{"path": path, "side": side, "r_result": 0.8,
                     "closed_at": f"2026-10-{i:02d}T00:00:00+00:00"} for i in range(1, n + 1)]

        try:
            b.st["closed"] = closes("QUIET LONG", "LONG", 20)
            sig = {"paper_only": True, "symbol": "BTCUSDT", "side": "LONG", "id": "q",
                   "signal_path": "QUIET LONG"}
            b._open_quiet_long_paper(sig, {"reasons": []}, "安靜期強勢多", (0.0, "NEUTRAL"), 100)
            self.assertEqual(opened, [False])
            opened.clear()
            b.st["closed"] = closes("QUIET LONG", "LONG", 7)
            thin = {"paper_only": True, "symbol": "ETHUSDT", "side": "LONG", "id": "q2",
                    "signal_path": "QUIET LONG"}
            b._open_quiet_long_paper(thin, {"reasons": []}, "安靜期強勢多", (0.0, "NEUTRAL"), 100)
            self.assertEqual(opened, [True])
            opened.clear()
            b.st["closed"] = closes("QUIET LONG", "LONG", 20)
            stock = {"paper_only": True, "stock": True, "symbol": "NVDAUSDT", "side": "LONG", "id": "qs",
                     "signal_path": "QUIET LONG"}
            b._open_quiet_long_paper(stock, {"reasons": []}, "安靜期強勢多", (0.0, "NEUTRAL"), 100)
            self.assertEqual(opened, [True])
            opened.clear()
            b.st["closed"] = closes(S.TREND_PATH, "LONG", 20)
            trend = {"trend": True, "paper_only": True, "symbol": "SOLUSDT", "side": "LONG", "id": "t",
                     "signal_path": S.TREND_PATH, "entry_price": 100, "stop_price": 90}
            b._open_trend_paper(trend, {"reasons": []}, S.TREND_PATH, (0.0, "NEUTRAL"), 100)
            self.assertEqual(opened, [False])
        finally:
            S.stock_bases = prev

    def test_a_daily_trend_paper_keeps_its_volatility_stop(self):
        book = FakeBook()
        book.px = 1.009
        b = bot(book)
        tr = trade()
        tr["paper"] = True
        tr["paper_probe"] = True
        tr["path"] = S.TREND_PATH
        tr["opened_ts"] = 1_700_000_000
        tr["tps"] = []
        b.protect(tr)
        self.assertIsNone(tr.get("exit_reason"))
        self.assertEqual(tr["stop"], 1.01)
        book.px = 0.97
        b.protect(tr)
        self.assertIsNone(tr.get("exit_reason"))
        self.assertEqual(tr["stop"], 1.01)
        self.assertGreater(tr["mfe_r"], 2)


class DailyTrend(unittest.TestCase):
    now = 1_800_000_000_000

    def bars(self, closes):
        start = self.now - 86_400_000 * (len(closes) + 2)
        return [[start + i * 86_400_000, c, c, c, c, 1] for i, c in enumerate(closes)]

    def walk(self, steps):
        px, out = 100.0, [100.0]
        for step in steps:
            px *= 1 + step
            out.append(px)
        return out

    def test_both_horizons_up_is_a_long_and_the_stop_is_clamped(self):
        quiet = self.walk([0.004 if i % 5 else 0.003 for i in range(21)])
        hit = S.daily_trend_setup(self.bars(quiet), self.now, 0.1)
        self.assertEqual(hit["side"], "LONG")
        self.assertAlmostEqual(hit["stop_pct"], 0.02)
        self.assertLess(hit["stop"], hit["entry"])
        wild = self.walk([0.25 if i % 2 == 0 else -0.10 for i in range(21)])
        hit = S.daily_trend_setup(self.bars(wild), self.now, 0.0)
        self.assertEqual(hit["side"], "LONG")
        self.assertAlmostEqual(hit["stop_pct"], 0.15)

    def test_a_mild_short_waits_while_btc_is_up_or_unknown(self):
        mild = self.walk([
            0.05, -0.06, 0.04, -0.05, 0.03, -0.04, 0.05, -0.06, 0.04, -0.03,
            0.02, -0.03, 0.04, -0.05, 0.03, -0.02, 0.04, -0.03, 0.02, -0.025,
        ])
        self.assertIsNone(S.daily_trend_setup(self.bars(mild), self.now, 0.1))
        self.assertIsNone(S.daily_trend_setup(self.bars(mild), self.now, None))
        hit = S.daily_trend_setup(self.bars(mild), self.now, -0.05)
        self.assertEqual(hit["side"], "SHORT")
        self.assertGreater(hit["stop"], hit["entry"])

    def test_a_one_sigma_short_still_opens_while_btc_is_up(self):
        down = self.walk([ -0.03 if i % 3 else -0.01 for i in range(21)])
        hit = S.daily_trend_setup(self.bars(down), self.now, 0.2)
        self.assertEqual(hit["side"], "SHORT")
        self.assertTrue(hit["btc_up"])

    def test_disagreement_flat_and_a_forming_bar_do_not_enter(self):
        mix = self.walk([0.03 if i < 14 else -0.015 for i in range(21)])
        self.assertIsNone(S.daily_trend_setup(self.bars(mix), self.now, -0.2))
        self.assertIsNone(S.daily_trend_setup(self.bars([100.0] * 21), self.now, -0.1))
        self.assertIsNone(S.daily_trend_setup(self.bars(self.walk([0.01] * 10)), self.now, 0.1))
        up = self.walk([0.012 if i % 2 == 0 else 0.008 for i in range(21)])
        rows = self.bars(up)
        hit = S.daily_trend_setup(rows, self.now, 0.1)
        rows.append([self.now - 1000, 1, 1, 1, 1, 1])
        again = S.daily_trend_setup(rows, self.now, 0.1)
        self.assertEqual(again["side"], "LONG")
        self.assertEqual(again["entry"], hit["entry"])
        self.assertEqual(again["bar_open"], hit["bar_open"])

    def test_the_exit_is_the_sign_change_and_a_missing_candle_holds(self):
        up = self.walk([0.012 if i % 2 == 0 else 0.008 for i in range(21)])
        down = self.walk([ -0.03 if i % 3 else -0.01 for i in range(21)])
        self.assertTrue(S.daily_trend_holds(self.bars(up), self.now, "LONG"))
        self.assertFalse(S.daily_trend_holds(self.bars(down), self.now, "LONG"))
        self.assertTrue(S.daily_trend_holds(self.bars(down), self.now, "SHORT"))
        self.assertIsNone(S.daily_trend_holds([], self.now, "LONG"))
        self.assertIsNone(S.daily_trend_holds(self.bars(up[:5]), self.now, "LONG"))


if __name__ == "__main__":
    unittest.main()
