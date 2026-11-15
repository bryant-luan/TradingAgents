"""Exit bookkeeping. A vanished take-profit order is not a fill, and a flat venue is not a new loss."""
import tempfile
import threading
import unittest
from pathlib import Path

from live import bot as botmod
from live.bot import Bot, position_gone, tp_fill_verdict, tp_still_ahead
from live.toobit import ToobitError


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

    def test_flat_book_does_not_invent_a_loss(self):
        book = FakeBook()
        book.px = 1.009  # -0.9R on the short, past the -0.85 line
        b = bot(book)
        tr = trade()
        tr["tps"] = []
        b.protect(tr)
        self.assertTrue(tr.get("exchange_flat"))
        self.assertIsNone(tr.get("exit_reason"))
        self.assertEqual(book.closed, [])
        book.stops.clear()
        b.replace_stop(tr, 1.008, "again")
        self.assertEqual(book.stops, [])

    def test_open_loser_still_flattens(self):
        book = FakeBook()
        book.px = 1.009
        book.rows = [{"side": "SHORT", "position": "100"}]
        b = bot(book)
        tr = trade()
        tr["tps"] = []
        b.protect(tr)
        self.assertTrue(str(tr.get("exit_reason") or "").startswith("浮虧"))
        self.assertEqual(book.closed, [("CGPT-SWAP-USDT", "SHORT")])

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
