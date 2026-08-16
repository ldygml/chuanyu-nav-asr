#!/usr/bin/env python3
"""
CTC Prefix Beam Search Decoder (no LM required, high-performance).

Based on the CTC prefix search decoding algorithm from Graves' thesis.
Compared to Viterbi (argmax) decoding, beam search maintains multiple
candidate prefixes at each time step, avoiding local optima.

Optimized for speed:
  - math.log/exp instead of torch scalar ops
  - Reduced default topk (20 vs 100)
  - Pre-allocated lists, no temporary tuples in hot path
  - String-based prefix keys (faster than tuple hashing)
"""

import math
from typing import List, Dict

import torch

from .base_decoder import BaseDecoder


def _logsumexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b)), C-math only — no torch overhead."""
    if a < b:
        a, b = b, a
    if b == float("-inf"):
        return a
    diff = b - a
    if diff < -50.0:
        return a
    return a + math.log1p(math.exp(diff))


def _prefix_str(tokens: list) -> str:
    """Compact string key for a prefix token list. Avoids tuple allocation."""
    return ",".join(map(str, tokens))


class CTCBeamSearchDecoder(BaseDecoder):
    """CTC prefix beam search decoder.

    Args:
        cfg: FlashlightDecoderConfig (reuses beam, beamsizetoken params)
        tgt_dict: target dictionary
    """

    def __init__(self, cfg, tgt_dict):
        super().__init__(tgt_dict)
        self.beam_size = (
            int(cfg.beam) if hasattr(cfg, "beam") and cfg.beam is not None else 3
        )
        # Use beamsizetoken as topk_candidates if set, else default to 20
        self.topk_candidates = (
            int(cfg.beamsizetoken)
            if hasattr(cfg, "beamsizetoken") and cfg.beamsizetoken is not None
            else min(20, self.vocab_size)
        )
        self.beam_threshold = (
            float(cfg.beamthreshold)
            if hasattr(cfg, "beamthreshold") and cfg.beamthreshold is not None
            else 50.0
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def decode(
        self,
        emissions: torch.FloatTensor,
    ) -> List[List[Dict[str, torch.LongTensor]]]:
        B, T, V = emissions.shape
        if T < 2:
            return self._fallback_greedy(emissions)

        topk = min(self.topk_candidates, V)
        _, topk_idx = torch.topk(emissions, k=topk, dim=-1)      # (B, T, K)
        topk_val = emissions.gather(dim=-1, index=topk_idx)       # (B, T, K)

        results = []
        for b in range(B):
            hypo = self._beam_search(
                T, V,
                topk_idx[b].tolist(),
                topk_val[b].tolist(),
            )
            results.append([hypo])
        return results

    # ── Core beam search (single utterance) ─────────────────────────────────

    def _beam_search(self, T: int, V: int,
                     tk_idx: List[List[int]],
                     tk_val: List[List[float]]) -> dict:
        """
        Run CTC prefix beam search on one utterance.

        Beam state: [prefix_key, log_p_b, log_p_nb, last_token, prefix_list]
        We track both p_b (prefix ending with blank) and p_nb (not ending with blank).

        CTC extensions at step t:
          - blank:  p_b_new  = log_p(blank) + logsumexp(p_b, p_nb)
          - same as last token:
              p_nb_new = logsumexp(p_nb + log_p(c), p_b + log_p(c))
          - new token:
              p_nb_new(prefix+c) = log_p(c) + logsumexp(p_b, p_nb)
        """
        NEG_INF = -1e30

        # beam entries: [prefix_list, p_b, p_nb, last_tok]
        # Use list of lists (not dict) for faster iteration with small beam sizes
        beams = [([], 0.0, NEG_INF, -1)]  # empty prefix

        for t in range(T):
            t_toks = tk_idx[t]
            t_vals = tk_val[t]

            # Find blank log-prob in top-k
            blank_lp = NEG_INF
            for i, tok in enumerate(t_toks):
                if tok == self.blank:
                    blank_lp = t_vals[i]
                    break

            # Accumulate into dict for dedup, then prune
            new_map: Dict[str, list] = {}  # key -> [prefix_list, p_b, p_nb, last_tok]

            for prefix_lst, p_b, p_nb, last in beams:
                base_sum = _logsumexp(p_b, p_nb) if p_nb > NEG_INF else p_b

                # (1) Emit blank → prefix unchanged
                if blank_lp > NEG_INF:
                    nb = blank_lp + base_sum
                    key = _prefix_str(prefix_lst)
                    self._acc(new_map, key, prefix_lst, nb, NEG_INF, last)

                # (2) Emit non-blank tokens
                for i, (tok, lp) in enumerate(zip(t_toks, t_vals)):
                    if tok == self.blank:
                        continue

                    if last == tok:
                        # Repeat last token — prefix unchanged
                        nb = _logsumexp(p_nb + lp, p_b + lp)
                        key = _prefix_str(prefix_lst)
                        self._acc(new_map, key, prefix_lst, NEG_INF, nb, last)
                    else:
                        # New token — extend prefix
                        new_lst = prefix_lst + [tok]
                        nb = lp + base_sum
                        key = _prefix_str(new_lst)
                        self._acc(new_map, key, new_lst, NEG_INF, nb, tok)

            # Score & prune
            scored = []
            for key, (lst, pb, pnb, last) in new_map.items():
                score = _logsumexp(pb, pnb)
                scored.append((lst, pb, pnb, last, score))
            scored.sort(key=lambda x: x[4], reverse=True)

            # Beam pruning
            beams = []
            if scored:
                best_score = scored[0][4]
                thr = best_score - self.beam_threshold
                for lst, pb, pnb, last, sc in scored:
                    if len(beams) >= self.beam_size:
                        break
                    if sc >= thr:
                        beams.append([lst, pb, pnb, last])

            if not beams:
                beams = [([], 0.0, NEG_INF, -1)]

        # Best beam
        if not beams:
            return {"tokens": torch.LongTensor([]), "score": 0.0}

        best = max(beams, key=lambda b: _logsumexp(b[1], b[2]))
        tokens = torch.LongTensor(best[0])
        score = _logsumexp(best[1], best[2])
        return {"tokens": tokens, "score": score}

    @staticmethod
    def _acc(d: Dict[str, list], key: str, lst: list,
             pb: float, pnb: float, last: int):
        """Merge a beam candidate into the dict (logsumexp for each term)."""
        if key in d:
            old = d[key]
            d[key] = [
                old[0],
                _logsumexp(old[1], pb),
                _logsumexp(old[2], pnb),
                old[3],
            ]
        else:
            d[key] = [lst, pb, pnb, last]

    # ── Fallback ────────────────────────────────────────────────────────────

    def _fallback_greedy(self, emissions):
        B, T, V = emissions.shape
        results = []
        for b in range(B):
            toks = emissions[b].argmax(dim=-1).unique_consecutive()
            toks = toks[toks != self.blank]
            score = emissions[b].max(dim=-1)[0].sum().item()
            results.append([{"tokens": toks, "score": score}])
        return results
