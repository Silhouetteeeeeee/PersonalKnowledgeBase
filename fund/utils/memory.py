"""Fund decision log with deferred reflection (3-phase memory)."""

import logging
from datetime import date, datetime
from typing import Optional

from agent.utils.llm import LLM
from storage.database import get_connection
from fund.utils.fund_data_tools import get_fund_nav, get_index_data

logger = logging.getLogger(__name__)


class FundMemory:
    """Three-phase memory: A) store → B) resolve → C) inject."""

    # ── Phase A: Store pending decision ──

    def store_decision(self, user_id: str, fund_code: str,
                       rating: str, decision_text: str, nav: float):
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO fund_decisions
                   (user_id, fund_code, decision_date, rating, decision_text,
                    nav_at_decision, status, created_at)
                   VALUES (?, ?, date('now'), ?, ?, ?, 'pending', datetime('now', 'localtime'))""",
                (user_id, fund_code, rating, decision_text, nav),
            )
            conn.commit()
            logger.info("Memory Phase A: stored decision for %s %s", user_id, fund_code)
        except Exception as e:
            conn.rollback()
            logger.error("store_decision failed: %s", e)
        finally:
            conn.close()

    # ── Phase B: Resolve pending decisions (compare outcome) ──

    def resolve_pending(self, user_id: str, fund_code: str):
        """Called before next analysis of same fund. Reflects on past decisions."""
        conn = get_connection()
        try:
            pending = conn.execute(
                """SELECT * FROM fund_decisions
                   WHERE fund_code = ? AND user_id = ? AND status = 'pending'
                   ORDER BY created_at ASC""",
                (fund_code, user_id),
            ).fetchall()
        finally:
            conn.close()

        if not pending:
            return

        today_nav_records = get_fund_nav(fund_code, days=5)
        if not today_nav_records:
            logger.warning("resolve_pending: no NAV data for %s", fund_code)
            return
        today_nav = today_nav_records[0]["nav"]

        for p in pending:
            raw_return = (today_nav - p["nav_at_decision"]) / p["nav_at_decision"] if p["nav_at_decision"] else 0.0
            reflection = self._reflect(p["decision_text"], raw_return)

            conn = get_connection()
            try:
                conn.execute(
                    """UPDATE fund_decisions
                       SET status = 'resolved', raw_return = ?, reflection = ?,
                           resolved_at = datetime('now', 'localtime')
                       WHERE id = ?""",
                    (raw_return, reflection, p["id"]),
                )
                conn.commit()
                logger.info("Memory Phase B: resolved decision %d (return: %+.1f%%)",
                           p["id"], raw_return * 100)
            except Exception as e:
                conn.rollback()
                logger.error("resolve_pending update failed: %s", e)
            finally:
                conn.close()

    def _reflect(self, decision_text: str, raw_return: float) -> str:
        """Reflector prompt — 2-4 sentences of structured reflection."""
        prompt = (
            "You are a fund analyst reviewing your own past decision now that the outcome is known.\n"
            "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
            "Cover in order:\n"
            "1. Was the rating call correct? (cite the return figure)\n"
            "2. Which part of the analysis held or failed?\n"
            "3. One concrete lesson to apply to the next similar fund analysis.\n\n"
            "Be specific and terse. Every word must earn its place."
        )
        messages = [
            ("system", prompt),
            ("human",
             f"Raw return: {raw_return:+.1%}\n\n"
             f"Decision:\n{decision_text}"),
        ]
        try:
            llm = LLM.get_model()
            return llm.invoke(messages).content
        except Exception as e:
            logger.error("Reflection failed: %s", e)
            return "Reflection unavailable."

    # ── Phase C: Inject context for Portfolio Manager ──

    def get_past_context(self, user_id: str, fund_code: str,
                         n_same: int = 3, n_cross: int = 2) -> str:
        """Get resolved decisions as context for the PM prompt."""
        conn = get_connection()
        try:
            resolved = conn.execute(
                """SELECT * FROM fund_decisions
                   WHERE user_id = ? AND status = 'resolved'
                   ORDER BY resolved_at DESC""",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()

        same = [dict(r) for r in resolved if r["fund_code"] == fund_code][:n_same]
        cross = [dict(r) for r in resolved if r["fund_code"] != fund_code][:n_cross]

        parts = []
        if same:
            parts.append("**过去对 {fund_code} 的分析（近期优先）:**")
            for s in same:
                ret = f"{s['raw_return']:+.1%}" if s['raw_return'] is not None else "N/A"
                parts.append(
                    f"- [{s['decision_date']} | {s['rating']} | {ret}]\n"
                    f"  反思: {s['reflection']}"
                )
        if cross:
            parts.append("\n**其他基金的经验教训:**")
            for c in cross:
                parts.append(f"- [{c['fund_code']} | {c['decision_date']} | {c['rating']}] {c['reflection']}")

        return "\n\n".join(parts) if parts else ""
