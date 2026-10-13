"""
07_write_results.py -- write the results of the GDL extension into the READMEs, every number read from the JSON
outputs (verdict_<sample>.json, baselines_<sample>.json). Idempotent: it replaces its own blocks.

Usage: python 07_write_results.py
"""
import json
import re
from pathlib import Path

import gdl_common as G

ROOT = G.HERE.parent
MINUS = "−"


def load(name):
    p = G.OUT / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def f3(x):
    return f"{x:.3f}"


def summarise(sample):
    v, b = load(f"verdict_{sample}.json"), load(f"baselines_{sample}.json")
    if v is None or b is None:
        return None
    m, c = v["targets"]["mean"], v["targets"]["ceiling"]
    gm, gc = list(m["gnn"].values()), list(c["gnn"].values())
    comp_mean = {k: x for k, x in b["comparisons"].items() if k.split("|")[0] in ("ridge", "rf", "gbm")}
    comp_ceil = {k: x for k, x in b["comparisons"].items() if k.split("|")[0] == "gbmq"}

    def tally(comp):
        helps = sum(x["boot90_hi"] < 0 for x in comp.values())
        hurts = sum(x["boot90_lo"] > 0 for x in comp.values())
        return helps, hurts, len(comp) - helps - hurts, len(comp)

    train_mean = m["trivial"]["training mean"]
    cb_name, cb = min(c["gnn"].items(), key=lambda kv: kv[1]["per_seed_mean_score"])
    kept = (cb["placebo_gain_over_mlp"] / cb["real_gain_over_mlp"]) if cb["real_gain_over_mlp"] > 0 else None
    return {
        "cb_name": cb_name.rsplit("|", 1)[0].replace("|", " "), "cb_score": cb["per_seed_mean_score"],
        "cb_diff": cb["vs"]["non_graph"]["diff"], "cb_lo": cb["vs"]["non_graph"]["boot90"][0], "cb_hi": cb["vs"]["non_graph"]["boot90"][1],
        "cb_placebo": cb["placebo_mean_score"], "cb_kept": kept,
        "sample": sample, "n_economies": b["n_economies"], "n_rows": v["n_test_rows"], "verdict": v["verdict"],
        "n_success": v["n_successful_settings"],
        "train_mean": train_mean, "best_ng": m["best_non_graph"], "best_ng_score": m["best_non_graph_score"],
        "best_slx": m["best_slx"], "best_slx_score": m["best_slx_score"], "mlp": m["twin_mlp_score"],
        "gain_vs_mean_pct": 100 * (train_mean - m["best_non_graph_score"]) / train_mean,
        "gnn_mean_lo": min(r["per_seed_mean_score"] for r in gm), "gnn_mean_hi": max(r["per_seed_mean_score"] for r in gm),
        "n_mean_sig_worse": sum(r["vs"]["non_graph"]["boot90"][0] > 0 for r in gm),
        "n_mean_real_beats_placebo": sum(r["per_seed_mean_score"] < r["placebo_mean_score"] for r in gm),
        "n_settings": len(gm),
        "slx_mean": tally(comp_mean), "slx_ceiling": tally(comp_ceil),
        "c_best_ng": c["best_non_graph"], "c_best_ng_score": c["best_non_graph_score"], "c_best_slx": c["best_slx"],
        "c_best_slx_score": c["best_slx_score"], "c_mlp": c["twin_mlp_score"],
        "gnn_c_lo": min(r["per_seed_mean_score"] for r in gc), "gnn_c_hi": max(r["per_seed_mean_score"] for r in gc),
        "n_c_ci_includes_zero": sum(r["vs"]["non_graph"]["boot90"][0] <= 0 <= r["vs"]["non_graph"]["boot90"][1] for r in gc),
        "n_c_real_beats_placebo": sum(r["per_seed_mean_score"] < r["placebo_mean_score"] for r in gc),
        "n_c_below_mlp": sum(r["per_seed_mean_score"] < c["twin_mlp_score"] for r in gc),
        "placebo_c_lo": min(r["placebo_mean_score"] for r in gc), "placebo_c_hi": max(r["placebo_mean_score"] for r in gc),
        "n_beating_mean": len(m.get("learned_models_beating_training_mean", [])),
    }


P, S = summarise("primary"), summarise("nohom")
if P is None:
    raise SystemExit("run 03-05 for the primary sample first")


def block(lang):
    en = lang == "en"
    t = (lambda e, s: e if en else s)
    lines = []
    lines.append(t(
        f"**Verdict: {P['verdict']}.** On the primary sample ({P['n_economies']} economies, {P['n_rows']} out-of-sample country-years in four rolling-origin blocks, 2012–2023), {P['n_success']} of {2 * P['n_settings']} GNN settings (GCN and GAT × four graphs × mean and ceiling) meet the criterion fixed beforehand.",
        f"**Veredicto: {'resultado nulo: ningún GNN cumple el criterio fijado de antemano' if P['n_success'] == 0 else 'al menos un GNN cumple el criterio fijado de antemano'}.** En la muestra primaria ({P['n_economies']} economías, {P['n_rows']} países-año fuera de muestra en cuatro bloques de origen móvil, 2012–2023), {P['n_success']} de {2 * P['n_settings']} configuraciones de GNN (GCN y GAT × cuatro grafos × media y techo) cumplen el criterio."))
    lines.append("")
    lines.append(t(
        f"- **The variables carry little signal.** RMSE of the demeaned next-year growth: training mean {f3(P['train_mean'])}; best learned model without graph ({P['best_ng'].split('|')[0]}) {f3(P['best_ng_score'])}, only {P['gain_vs_mean_pct']:.1f}% better; {P['n_beating_mean']} learned models beat the mean, none by much.",
        f"- **Las variables aportan poca señal.** RMSE del crecimiento del año siguiente ajustado por año: media de entrenamiento {f3(P['train_mean'])}; mejor modelo aprendido sin grafo ({P['best_ng'].split('|')[0]}) {f3(P['best_ng_score'])}, solo {P['gain_vs_mean_pct']:.1f}% mejor; {P['n_beating_mean']} modelos aprendidos superan la media, ninguno por mucho."))
    lines.append(t(
        f"- **Graphs do not help the mean forecast.** GNNs {f3(P['gnn_mean_lo'])}–{f3(P['gnn_mean_hi'])} against {f3(P['best_ng_score'])} (best non-graph) and {f3(P['mlp'])} (the MLP twin); {P['n_mean_sig_worse']} of {P['n_settings']} GNN settings are significantly worse than the best non-graph model (90% country-block bootstrap). Graph-as-features (SLX) models: the graph helps in {P['slx_mean'][0]} of {P['slx_mean'][3]} learner–graph pairs, hurts in {P['slx_mean'][1]} and is indistinguishable in {P['slx_mean'][2]}.",
        f"- **Los grafos no ayudan a predecir la media.** GNN {f3(P['gnn_mean_lo'])}–{f3(P['gnn_mean_hi'])} frente a {f3(P['best_ng_score'])} (mejor modelo sin grafo) y {f3(P['mlp'])} (el MLP gemelo); {P['n_mean_sig_worse']} de {P['n_settings']} GNN son significativamente peores que el mejor modelo sin grafo (bootstrap por países al 90%). Modelos con el grafo como variables (SLX): el grafo ayuda en {P['slx_mean'][0]} de {P['slx_mean'][3]} pares aprendiz–grafo, perjudica en {P['slx_mean'][1]} y es indistinguible en {P['slx_mean'][2]}."))
    lines.append(t(
        f"- **Growth ceiling (pinball loss at q=0.90).** Best non-graph ({P['c_best_ng'].split('|')[0]}) {f3(P['c_best_ng_score'])}, best SLX {f3(P['c_best_slx_score'])}, GNNs {f3(P['gnn_c_lo'])}–{f3(P['gnn_c_hi'])}; the bootstrap interval of the GNN-minus-best-non-graph difference includes zero in {P['n_c_ci_includes_zero']} of {P['n_settings']} settings. The MLP twin is worse ({f3(P['c_mlp'])}), so the GNNs look better than it in {P['n_c_below_mlp']} of {P['n_settings']} settings, but the node-permuted placebo graphs reach {f3(P['placebo_c_lo'])}–{f3(P['placebo_c_hi'])}: message passing acts as a regulariser, not as use of the true topology.",
        f"- **Techo de crecimiento (pérdida pinball en q=0.90).** Mejor sin grafo ({P['c_best_ng'].split('|')[0]}) {f3(P['c_best_ng_score'])}, mejor SLX {f3(P['c_best_slx_score'])}, GNN {f3(P['gnn_c_lo'])}–{f3(P['gnn_c_hi'])}; el intervalo bootstrap de la diferencia GNN menos mejor modelo sin grafo incluye el cero en {P['n_c_ci_includes_zero']} de {P['n_settings']} configuraciones. El MLP gemelo es peor ({f3(P['c_mlp'])}), así que los GNN parecen mejores que él en {P['n_c_below_mlp']} de {P['n_settings']}, pero los grafos placebo con nodos permutados llegan a {f3(P['placebo_c_lo'])}–{f3(P['placebo_c_hi'])}: el paso de mensajes actúa como regularizador, no como uso de la topología real."))
    if S is not None:
        lines.append(t(
            f"- **Sensitivity sample without homicide features ({S['n_economies']} economies, {S['n_rows']} test country-years):** {S['verdict']}; best non-graph {f3(S['best_ng_score'])}, GNNs {f3(S['gnn_mean_lo'])}–{f3(S['gnn_mean_hi'])}, training mean {f3(S['train_mean'])}.",
            f"- **Muestra de sensibilidad sin variables de homicidios ({S['n_economies']} economías, {S['n_rows']} países-año de prueba):** {'resultado nulo' if S['n_success'] == 0 else 'hay configuraciones que cumplen el criterio'}; mejor sin grafo {f3(S['best_ng_score'])}, GNN {f3(S['gnn_mean_lo'])}–{f3(S['gnn_mean_hi'])}, media de entrenamiento {f3(S['train_mean'])}."))
    def call(X):
        kept = "n/a" if X["cb_kept"] is None else f"{100 * X['cb_kept']:.0f}%"
        return (f"{X['cb_name']} scores {f3(X['cb_score'])} against {f3(X['c_best_ng_score'])} for the best non-graph model (difference {X['cb_diff']:+.3f}, 90% interval [{X['cb_lo']:+.3f}, {X['cb_hi']:+.3f}]); its placebo graphs score {f3(X['cb_placebo'])} and keep {kept} of its gain over the MLP twin (the criterion allows at most 25%)",
                f"{X['cb_name']} obtiene {f3(X['cb_score'])} frente a {f3(X['c_best_ng_score'])} del mejor modelo sin grafo (diferencia {X['cb_diff']:+.3f}, intervalo al 90% [{X['cb_lo']:+.3f}, {X['cb_hi']:+.3f}]); sus grafos placebo obtienen {f3(X['cb_placebo'])} y conservan {kept} de su ganancia sobre el MLP gemelo (el criterio permite como m\u00e1ximo 25%)")
    if S is not None:
        lines.append(t(
            f"- **Closest call (upper tail).** Primary sample: {call(P)[0]}. Sensitivity sample: {call(S)[0]}. Neither sample meets the criterion: the intervals include zero and the placebo graphs recover a large share of the gain, so any information that trade partners carry about the upper tail is at most a weak hint.",
            f"- **El caso m\u00e1s cercano (cola superior).** Muestra primaria: {call(P)[1]}. Muestra de sensibilidad: {call(S)[1]}. Ninguna muestra cumple el criterio: los intervalos incluyen el cero y los grafos placebo recuperan una parte grande de la ganancia, as\u00ed que la informaci\u00f3n que los socios comerciales aporten sobre la cola superior es, a lo sumo, un indicio d\u00e9bil."))
    lines.append("")
    lines.append(t(
        "These are predictive results on a small panel with noisy governance scores (Module 14); they say that country graphs add no detectable out-of-sample information about next-year growth or its upper tail beyond the countries' own features, not that spillovers do not exist. The design, its deviations log and the limits are in [`gdl_extension/README.md`](gdl_extension/README.md).",
        "Son resultados predictivos sobre un panel pequeño con puntajes de gobernanza ruidosos (Módulo 14); dicen que los grafos entre países no agregan información fuera de muestra detectable sobre el crecimiento del año siguiente ni sobre su cola superior más allá de las variables propias de cada país, no que los contagios no existan. El diseño, su registro de desviaciones y los límites están en [`gdl_extension/README.md`](gdl_extension/README.md)."))
    return "\n".join(lines)


def top_section(lang):
    en = lang == "en"
    head = ("## Geometric Deep Learning Extension (October 2026)" if en else "## Extensión de Aprendizaje Profundo Geométrico (Octubre 2026)")
    intro = (
        "Motivated by four papers on graph neural networks in economics and finance, [`gdl_extension/`](gdl_extension/) asks whether the graph between countries (trade 2000–2008, distance, borders) improves out-of-sample prediction of next-year GDP growth and of its upper tail, using a global panel of up to 141 economies. The design, including the success criterion, was written before any model was fitted (README there, with a deviations log); it needs PyTorch and PyTorch Geometric in a separate environment (`requirements-gdl.txt`)."
        if en else
        "Motivada por cuatro artículos sobre redes neuronales de grafos en economía y finanzas, [`gdl_extension/`](gdl_extension/) pregunta si el grafo entre países (comercio 2000–2008, distancia, fronteras) mejora la predicción fuera de muestra del crecimiento del PIB del año siguiente y de su cola superior, con un panel global de hasta 141 economías. El diseño, incluido el criterio de éxito, se escribió antes de ajustar ningún modelo (README allí, con un registro de desviaciones); requiere PyTorch y PyTorch Geometric en un entorno aparte (`requirements-gdl.txt`).")
    return f"{head}\n\n{intro}\n\n{block(lang)}\n\n"


def replace_between(text, begin, end, new):
    pat = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.S)
    return pat.sub(lambda m: begin + "\n" + new + "\n" + end, text)


# 1. gdl_extension/README.md
p = G.HERE / "README.md"
t = p.read_text(encoding="utf-8")
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
if BEGIN not in t:
    t = t.replace("## Steps", f"## Results\n\n{BEGIN}\n{END}\n\n## Steps")
t = replace_between(t, BEGIN, END, block("en").replace("(gdl_extension/README.md)", "(README.md)").replace("`gdl_extension/README.md`", "this file"))
t = re.sub(r"\| `02_fetch_trade_edges.py` \| bilateral trade edges, 2000-2008 \| run \|.*", "| `02_fetch_trade_edges.py` | bilateral trade edges, 2000-2008 | done |\n| `gdl_common.py` | shared data, folds, graphs and metrics | done |\n| `03_baselines.py` | no-graph, SLX and placebo baselines | done |\n| `04_gnn.py` | MLP, GCN and GAT (mean and ceiling), placebo graphs | done |\n| `05_verdict.py` | applies the pre-registered success criterion | done |\n| `06_figure.py` | verdict figure | done |\n| `07_write_results.py` | writes these results into the READMEs | done |", t, flags=re.S)
p.write_text(t, encoding="utf-8", newline="")

# 2. main READMEs
for name, lang, marker in (("README.md", "en", "## Repository Cleanup Notes"), ("README_ES.md", "es", "## Notas para el Usuario")):
    p = ROOT / name
    t = p.read_text(encoding="utf-8")
    first = "## Geometric Deep Learning Extension" if lang == "en" else "## Extensión de Aprendizaje Profundo Geométrico"
    if first in t:
        i, j = t.index(first), t.index(marker)
        assert i < j
        t = t[:i] + t[j:]
    assert t.count(marker) == 1
    t = t.replace(marker, top_section(lang) + marker)
    p.write_text(t, encoding="utf-8", newline="")
print("results written;", "sensitivity sample included" if S else "sensitivity sample pending")
