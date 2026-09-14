import re
from collections import defaultdict


class NERMetric:
    _ENTITY_RE = re.compile(r"^(.+?):\s*GENE\s*$", re.IGNORECASE)

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp_dict = defaultdict(int)
        self.pred_sum_dict = defaultdict(int)
        self.true_sum_dict = defaultdict(int)

    def extract_entities(self, text: str):
        ents = set()
        if not text:
            return ents
        text = text.strip()
        if text == "无实体":
            return ents

        for line in text.split("\n"):
            line = line.strip()
            if not line or line == "无实体":
                continue

            # 先整行匹配
            m = self._ENTITY_RE.match(line)
            if m and m.group(1).strip():
                ents.add((m.group(1).strip(), "GENE"))
                continue

            # ,;，；分隔都算对
            for part in re.split(r"[,;，；]", line):
                m = self._ENTITY_RE.match(part.strip())
                if m and m.group(1).strip():
                    ents.add((m.group(1).strip(), "GENE"))
        return ents

    def update(self, pred, true):
        pred_entities = self.extract_entities(pred)
        true_entities = self.extract_entities(true)

        # 按实体类型分组
        true_by_type = defaultdict(set)
        pred_by_type = defaultdict(set)
        for name, typ in true_entities:
            true_by_type[typ].add((name, typ))
        for name, typ in pred_entities:
            pred_by_type[typ].add((name, typ))

        all_types = set(true_by_type.keys()) | set(pred_by_type.keys())
        for typ in all_types:
            t = true_by_type.get(typ, set())
            p = pred_by_type.get(typ, set())
            self.tp_dict[typ]        += len(p & t)
            self.pred_sum_dict[typ]  += len(p)
            self.true_sum_dict[typ]  += len(t)

    def update_batch(self, preds, trues):
        assert len(preds) == len(trues)
        for p, r in zip(preds, trues):
            self.update(p, r)

    def compute(self):
        all_types = sorted(
            set(self.tp_dict.keys())
            | set(self.pred_sum_dict.keys())
            | set(self.true_sum_dict.keys())
        )

        per_type = {}
        total_tp = total_pred = total_true = 0
        macro_p = macro_r = macro_f1 = 0.0

        for typ in all_types:
            tp = self.tp_dict.get(typ, 0)
            pred_sum = self.pred_sum_dict.get(typ, 0)
            true_sum = self.true_sum_dict.get(typ, 0)

            total_tp += tp
            total_pred += pred_sum
            total_true += true_sum

            p = self._prf_divide(tp, pred_sum)
            r = self._prf_divide(tp, true_sum)
            f1 = self._prf_divide(2 * p * r, p + r)
            per_type[typ] = {
                "precision": p, "recall": r, "f1": f1, "support": true_sum,
            }
            macro_p += p
            macro_r += r
            macro_f1 += f1

        num_types = len(all_types)
        if num_types > 0:
            macro_p /= num_types
            macro_r /= num_types
            macro_f1 /= num_types

        micro_p = self._prf_divide(total_tp, total_pred)
        micro_r = self._prf_divide(total_tp, total_true)
        micro_f1 = self._prf_divide(2 * micro_p * micro_r, micro_p + micro_r)

        return {
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
            "tp": total_tp, "fp": total_pred - total_tp, "fn": total_true - total_tp,

            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f1,
            "per_type": per_type,
        }


    @staticmethod
    def _prf_divide(num, den, zero_division=0.0):
        return zero_division if den == 0 else num / den