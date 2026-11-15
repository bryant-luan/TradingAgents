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
        self.assertEqual(S.roll_ev_floor(0.50, [{"ev": 0.32}, {"ev": 0.27}, {"ev": 0.37}]), 0.50)
        self.assertFalse(S.self_short_full_live("SHORT", 0.4, 0.008, 0.50, 1.0, 0.05, 0.51))
        self.assertTrue(S.self_short_full_live("SHORT", 0.4, 0.008, 0.73, 1.0, 0.05, 0.51))

    def test_a_win_steps_the_bar_back(self):
        self.assertEqual(S.ease_ev_floor(0.51, 0.50), 0.50)
        self.assertEqual(S.ease_ev_floor(0.62, 0.50), 0.57)

    def test_gate_book_is_larger_and_toobit_stays(self):
        gate, toobit = S.book_limits("gate"), S.book_limits("toobit")
        self.assertEqual(gate["risk"], 0.08)
        self.assertEqual(gate["short_risk"], 0.12)
        self.assertGreater(gate["total_margin"], toobit["total_margin"])
        self.assertEqual(toobit["risk"], S.MIN_RISK_PCT)
        self.assertEqual(toobit["short_risk"], S.SELF_SHORT_RISK_PCT)

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


if __name__ == "__main__":
    unittest.main()
