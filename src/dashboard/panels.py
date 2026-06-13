"""Metrik panelleri ve log konsolu bileşenleri.

Saf veri biçimlendirme (``metric_rows``) ile Dash bileşeni üretimini ayırır;
biçimlendirme test edilebilir, bileşen üreticileri ince kabuktur.
"""

from __future__ import annotations

from collections import Counter

import plotly.graph_objects as go
from dash import dcc, html

from src.simulation.metrics import SimulationMetrics

# Plotly figürleri için koyu tema (dashboard ile uyumlu).
_DARK = dict(paper_bgcolor="#15151c", plot_bgcolor="#1a1a22", font=dict(color="#e0e0e8"),
             margin=dict(l=40, r=20, t=40, b=40))

# Doluluk bu oranı aşınca "doyuma yakın" görsel uyarısı verilir (D2). Doyumda
# yerleştirme kalitesi düşer (acil olmayan bobinler kapı civarında takılı kalır);
# bu bir tasarım sınırıdır, gizlenmez — savunmada dürüstlük + analiz fırsatı.
SATURATION_THRESHOLD: float = 0.85


def metric_rows(
    metrics: SimulationMetrics,
    policy_name: str,
    clock_hours: float,
    fill_ratio: float,
    delay_mae: float | None = None,
) -> list[tuple[str, str]]:
    """Panelde gösterilecek (etiket, değer) çiftlerini üretir (saf, test edilebilir)."""
    avg_loading = (
        metrics.total_loading_time_min / metrics.n_dispatches
        if metrics.n_dispatches > 0
        else 0.0
    )
    # Rehandling birikme hızı (sim saati başına) — anlık verimliliğin canlı ölçüsü.
    reh_rate = metrics.rehandling_count / clock_hours if clock_hours > 0 else 0.0
    rows = [
        ("Politika", policy_name),
        ("Sim saati", f"{clock_hours:.1f} s"),
        ("Doluluk", f"{fill_ratio * 100:.0f}%"),
        ("Rehandling", str(metrics.rehandling_count)),
        ("Rehandling/saat", f"{reh_rate:.2f}"),
        ("Yerleştirme", str(metrics.n_placements)),
        ("Sevkiyat", str(metrics.n_dispatches)),
        ("Vinç mesafesi", f"{metrics.total_crane_distance_m:.0f} m"),
        ("Ort. yükleme", f"{avg_loading:.1f} dk"),
    ]
    if delay_mae is not None:
        rows.append(("Gecikme MAE", f"{delay_mae:.1f} dk"))
    return rows


def build_metric_panel(rows: list[tuple[str, str]], fill_ratio: float | None = None) -> html.Div:
    """Metrik çiftlerini kart dizisi olarak render eder.

    ``fill_ratio`` verilir ve ``SATURATION_THRESHOLD``'u aşarsa (D2): üstte bir
    "DOYUMA YAKIN" uyarı bandı gösterilir ve "Doluluk" kartı uyarı rengine boyanır.
    """
    saturated = fill_ratio is not None and fill_ratio >= SATURATION_THRESHOLD
    cards = []
    for label, value in rows:
        # Doyumda Doluluk kartını vurgula (kırmızımsı), diğerleri nötr.
        warn = saturated and label == "Doluluk"
        cards.append(
            html.Div(
                [html.Div(label, className="metric-label"),
                 html.Div(value, className="metric-value",
                          style={"color": "#ff6b6b", "fontWeight": "bold"} if warn else {})],
                className="metric-card",
                style={
                    "border": ("1px solid #ff6b6b" if warn else "1px solid #444"),
                    "borderRadius": "6px", "padding": "8px 12px",
                    "margin": "4px", "minWidth": "120px", "display": "inline-block",
                    "background": ("#2a1a1a" if warn else "#1e1e26"),
                },
            )
        )
    children = []
    if saturated:
        children.append(
            html.Div(
                f"⚠ DOYUMA YAKIN (doluluk %{fill_ratio * 100:.0f}) — boş prime slotlar "
                f"tükeniyor; yerleştirme kalitesi düşebilir (tasarım sınırı).",
                style={
                    "background": "#3a1212", "color": "#ff9b9b", "padding": "8px 12px",
                    "borderRadius": "6px", "marginBottom": "6px", "fontWeight": "bold",
                },
            )
        )
    children.append(html.Div(cards, style={"display": "flex", "flexWrap": "wrap"}))
    return html.Div(children)


def build_heldout_badge(policy_name: str, heldout: dict[str, float] | None) -> html.Div:
    """Seçili politikanın HELD-OUT ortalama rehandling'ini rozet olarak gösterir (#1).

    ``heldout``: {politika: ortalama_rehandling} (comparison.json'dan, görülmemiş
    test havuzu). Canlı koşum tek senaryodur; bu rozet o politikanın 30 görülmemiş
    popülasyondaki genel performansını hatırlatır (ezber değil, genelleşme). Veri
    yoksa boş döner.
    """
    if not heldout or policy_name not in heldout:
        return html.Div()
    val = heldout[policy_name]
    # En iyi (en düşük) politikayı altın, diğerlerini nötr çerçevele.
    is_best = val == min(heldout.values())
    color = "#f0c419" if is_best else "#7fd0ff"
    return html.Div(
        [
            html.Span("Held-out ort. rehandling: ", style={"color": "#9aa0b0"}),
            html.Span(f"{val:.2f}", style={"color": color, "fontWeight": "bold"}),
            html.Span(" (30 görülmemiş popülasyon)" + ("  ★ en iyi" if is_best else ""),
                      style={"color": "#777", "fontSize": "11px"}),
        ],
        style={"padding": "4px 10px", "background": "#1a1a22", "borderRadius": "6px",
               "border": "1px solid #444", "display": "inline-block", "fontSize": "12px"},
    )


def build_approaches_panel() -> html.Div:
    """Dört politikanın felsefesini özetleyen "Yaklaşımlar" bilgi kutusu (C3).

    "Neden birden fazla felsefe?" — aynı problemi farklı kuşak çözümlerle (rastgele
    referans → kural-tabanlı → ML-destekli → öğrenen) kıyaslamak, her birinin
    katkısını izole eder. Bu kutu savunmada "neden 4 politika" sorusunu yanıtlar.
    """
    def _row(name: str, color: str, text: str) -> html.Div:
        return html.Div(
            [
                html.Span(name, style={"color": color, "fontWeight": "bold", "minWidth": "110px",
                                       "display": "inline-block"}),
                html.Span(text),
            ],
            style={"margin": "5px 0"},
        )

    return html.Div(
        [
            html.Div("Yaklaşımlar — Neden 4 Politika?", style={"fontWeight": "bold", "marginBottom": "6px"}),
            _row("Random", "#888", "Referans (alt sınır): geçerli konumlardan rastgele seçer. "
                 "Diğerlerinin ne kadar kazandırdığını ölçmek için zemin."),
            _row("Heuristic", "#3498db", "Kural-tabanlı, mesafe-açgözlü: aciliyet–kapı eşleşmesi + "
                 "istif disiplini + affinity. Açıklanabilir klasik sezgisel."),
            _row("MLHeuristic", "#9b59b6", "Heuristic + LightGBM gecikme tahmini: geç gelecek aracın "
                 "bobinini daha derine koyup yer açar (tahmin-destekli)."),
            _row("PPO", "#e67e22", "Öğrenen ajan (MaskablePPO + CNN): deneme-yanılmayla rehandling'i "
                 "minimize eder. Projenin yıldızı; en düşük rehandling."),
            html.Div("Aynı problemi farklı kuşak çözümlerle kıyaslamak, ML/RL'in klasik sezgisele "
                     "net katkısını istatistiksel olarak izole eder.",
                     style={"marginTop": "8px", "fontSize": "11px", "color": "#9aa0b0"}),
        ],
        style={
            "border": "1px solid #444", "borderRadius": "6px", "padding": "10px 12px",
            "background": "#1a1a22", "fontSize": "12px", "lineHeight": "1.35",
        },
    )


def _decision_icon(reason: str) -> tuple[str, str]:
    """Karar gerekçesine göre (ikon, renk) döndürür — günlükte tür ayrımı için.

    SWAP/reposition (turuncu elmas), rehandling oluşan yerleştirme (kırmızı ok),
    normal yerleştirme (gri kare). Gerekçe metnine bakarak sınıflar.
    """
    r = reason.lower()
    if "swap" in r:
        return "◆", "#e67e22"          # swap / yer açma
    if "rehandling" in r and "+0" not in r:
        return "⬆", "#ff6b6b"          # bu yerleştirme rehandling tetikledi
    return "▪", "#5a5a6a"              # normal yerleştirme


def build_decision_log(decisions: list[dict], max_rows: int = 20) -> html.Div:
    """Son N kararı (yerleştirme/swap) gerekçeleriyle listeleyen "Karar Günlüğü" (B2).

    Her satır: zaman, bobin, seçilen slot, kısa gerekçe (rehandling deltası / swap
    denklemi). "Sistem her adımda ne düşündü" şeffaflığını verir — savunmada güçlü.
    ``decisions`` öğeleri controller'ın ürettiği sözlüklerdir.
    """
    if not decisions:
        body = [html.Div("(henüz karar yok)", style={"color": "#777"})]
    else:
        body = [
            html.Div(
                [
                    html.Span(_decision_icon(d["reason"])[0],
                              style={"color": _decision_icon(d["reason"])[1], "width": "16px",
                                     "display": "inline-block"}),
                    html.Span(f"t={d['t']:.1f}s ", style={"color": "#7fd0ff"}),
                    html.Span(f"{d['coil']} → ", style={"color": "#e0e0e8"}),
                    html.Span(f"Z{d['zone']} B{d['bay']} K{d['layer']}",
                              style={"color": "#9b59b6"}),
                    html.Span(f"  {d['reason']}", style={"color": "#bbb"}),
                ],
                style={"padding": "2px 0", "borderBottom": "1px solid #2a2a33",
                       "whiteSpace": "nowrap", "overflow": "hidden", "textOverflow": "ellipsis"},
            )
            for d in decisions[-max_rows:][::-1]  # en yeni üstte
        ]
    return html.Div(
        body,
        style={
            "background": "#101015", "padding": "8px 10px", "height": "220px",
            "overflowY": "auto", "fontSize": "12px", "borderRadius": "6px",
            "fontFamily": "monospace",
        },
    )


def build_comparison_metrics(lane_a: dict, lane_b: dict) -> html.Div:
    """İki politikanın metriklerini yan yana kıyaslayan tablo (C1).

    ``lane_a``/``lane_b``: {"name", "rehandling", "crane_m", "loading_min", "fill"}.
    Daha iyi (düşük) değer her satırda yeşil vurgulanır — kontrol odası kıyaslaması.
    """
    metrics = [
        ("Rehandling", "rehandling", "{:.0f}"),
        ("Vinç mesafesi", "crane_m", "{:.0f} m"),
        ("Yükleme süresi", "loading_min", "{:.0f} dk"),
        ("Doluluk", "fill", "{:.0%}"),
    ]

    def _cell(value: str, better: bool) -> html.Td:
        return html.Td(value, style={
            "padding": "4px 10px", "textAlign": "right",
            "color": "#5fd98a" if better else "#e0e0e8",
            "fontWeight": "bold" if better else "normal",
        })

    header = html.Tr([
        html.Th("Metrik", style={"padding": "4px 10px", "textAlign": "left"}),
        html.Th(lane_a["name"], style={"padding": "4px 10px", "color": "#3498db"}),
        html.Th(lane_b["name"], style={"padding": "4px 10px", "color": "#e67e22"}),
        html.Th("Δ (A−B)", style={"padding": "4px 10px", "color": "#9aa0b0"}),
    ])
    body_rows = []
    for label, key, fmt in metrics:
        a, b = lane_a[key], lane_b[key]
        # Doluluk hariç düşük = iyi; doluluk bilgi amaçlı (kıyas yok).
        a_better = key != "fill" and a < b
        b_better = key != "fill" and b < a
        # Delta: A−B. Düşük=iyi metriklerde negatif Δ → A daha iyi (yeşil), pozitif → B iyi.
        delta = a - b
        if key == "fill":
            delta_txt, delta_color = "—", "#666"
        else:
            sign = "+" if delta > 0 else ""
            delta_txt = f"{sign}{fmt.format(delta).replace(' m','').replace(' dk','')}"
            delta_color = "#5fd98a" if delta < 0 else ("#ff6b6b" if delta > 0 else "#9aa0b0")
        body_rows.append(html.Tr([
            html.Td(label, style={"padding": "4px 10px"}),
            _cell(fmt.format(a), a_better),
            _cell(fmt.format(b), b_better),
            html.Td(delta_txt, style={"padding": "4px 10px", "textAlign": "right",
                                       "color": delta_color, "fontFamily": "monospace"}),
        ]))
    return html.Table([header] + body_rows,
                      style={"borderCollapse": "collapse", "background": "#1e1e26",
                             "borderRadius": "6px", "marginTop": "6px"})


def build_legend() -> html.Div:
    """Sabit "Gösterge Açıklaması" kutusu (A1): renk/şekil/eksen semantiği.

    Figürün yanında durur; bir mühendis bakıp "sarı ne demek, elmas neden, Bay ne"
    sorularını kod okumadan yanıtlayabilsin. İçerik A4 (başlık) ile aynı sözlüğü
    kullanır (tek terim sözlüğü: aciliyet = sevkiyata kalan süreye göre 0–1 skor).
    """

    def _row(swatch: html.Span, text) -> html.Div:
        return html.Div(
            [swatch, html.Span(text, style={"marginLeft": "8px"})],
            style={"display": "flex", "alignItems": "center", "margin": "4px 0"},
        )

    def _dot(color: str, symbol: str = "●") -> html.Span:
        return html.Span(symbol, style={"color": color, "fontSize": "16px", "width": "18px"})

    return html.Div(
        [
            html.Div("Gösterge Açıklaması", style={"fontWeight": "bold", "marginBottom": "6px"}),
            # Renk (aciliyet) — sürekli skala.
            html.Div("Renk = Aciliyet (sevkiyata kalan süreye göre 0–1; 1 = en acil):",
                     style={"marginTop": "4px"}),
            _row(_dot("#f0f921"), "sarı → yüksek aciliyet (yakında sevk edilecek)"),
            _row(_dot("#0d0887"), "mor/mavi → düşük aciliyet (bekleyebilir)"),
            # Şekiller.
            html.Div("Şekiller:", style={"marginTop": "8px"}),
            _row(_dot("#2ecc71", "■"), "yeşil kare = yükleme kapısı (zone önü, bay 0)"),
            _row(_dot("#ff3b3b", "◆"), "kırmızı kenarlı elmas = yeri değişen bobin "
                 "(swap ile taşınan veya sevkiyatta engel olan)"),
            _row(_dot("rgba(160,160,180,0.6)", "□"), "soluk kare = boş slot"),
            _row(_dot("#9b59b6"), "normal nokta = yerleşik bobin"),
            # Eksenler.
            html.Div("Eksenler:", style={"marginTop": "8px"}),
            html.Div("• Bay = kapıdan depo içine uzaklık (m)", style={"margin": "2px 0"}),
            html.Div("• Zone = enine bölge/koridor (m)", style={"margin": "2px 0"}),
            html.Div("• Kat = istif yüksekliği (m)", style={"margin": "2px 0"}),
            html.Div("İpucu: bir bobine gelin (hover) → tip, ağırlık, bekleme süresi ve "
                     "(taşındıysa) swap gerekçesi görünür.",
                     style={"marginTop": "8px", "fontSize": "11px", "color": "#9aa0b0"}),
        ],
        style={
            "border": "1px solid #444", "borderRadius": "6px", "padding": "10px 12px",
            "background": "#1a1a22", "fontSize": "12px", "lineHeight": "1.3", "height": "100%",
        },
    )


def build_log_console(lines: list[str], max_lines: int = 25) -> html.Pre:
    """Son olayları satır satır gösteren log konsolu."""
    text = "\n".join(lines[-max_lines:]) if lines else "(henüz olay yok)"
    return html.Pre(
        text,
        style={
            "background": "#101015", "color": "#b8c0d0", "padding": "10px",
            "height": "220px", "overflowY": "auto", "fontSize": "12px",
            "borderRadius": "6px", "whiteSpace": "pre-wrap",
        },
    )


def build_data_exploration(scenario) -> html.Div:
    """Üretilen verinin keşif grafikleri — gecikme örüntüleri + veri kompozisyonu.

    Gecikme modelinin (LightGBM) ÖĞRENDİĞİ gizli örüntüleri görsel kanıtlar: gecikme
    dağılımı, gecikme↔mesafe, hava↔gecikme. Ayrıca bobin tipi dağılımı. Tüm grafikler
    senaryo verisinden bir kez hesaplanır (statik). Tezde/sunumda doğrudan kullanılabilir.
    """
    vehicles = list(scenario.vehicles.values())
    delays = [v.delay_minutes for v in vehicles]
    dists = [v.distance_km for v in vehicles]
    weathers = [v.weather.value for v in vehicles]

    # 1) Gecikme histogramı.
    fig_hist = go.Figure(go.Histogram(x=delays, nbinsx=40, marker_color="#e67e22"))
    fig_hist.update_layout(title="Araç Gecikme Dağılımı (dk)", **_DARK,
                           xaxis_title="gecikme (dk)", yaxis_title="araç sayısı", height=300)

    # 2) Gecikme vs mesafe (renk = hava) — mesafe/hava → gecikme örüntüsü.
    fig_scatter = go.Figure()
    for w in sorted(set(weathers)):
        xs = [d for d, ww in zip(dists, weathers) if ww == w]
        ys = [d for d, ww in zip(delays, weathers) if ww == w]
        fig_scatter.add_trace(go.Scattergl(x=xs, y=ys, mode="markers", name=w,
                                           marker=dict(size=4, opacity=0.5)))
    fig_scatter.update_layout(title="Gecikme vs Mesafe (renk: hava)", **_DARK,
                              xaxis_title="mesafe (km)", yaxis_title="gecikme (dk)", height=300)

    # 3) Havaya göre ortalama gecikme.
    by_weather: dict[str, list[float]] = {}
    for w, d in zip(weathers, delays):
        by_weather.setdefault(w, []).append(d)
    w_names = sorted(by_weather)
    w_means = [sum(by_weather[w]) / len(by_weather[w]) for w in w_names]
    fig_weather = go.Figure(go.Bar(x=w_names, y=w_means, marker_color="#3498db"))
    fig_weather.update_layout(title="Havaya Göre Ort. Gecikme (dk)", **_DARK,
                              xaxis_title="hava", yaxis_title="ort. gecikme (dk)", height=300)

    # 4) Bobin tipi dağılımı.
    type_counts = Counter(c.coil_type.value for c in scenario.coils.values())
    fig_types = go.Figure(go.Bar(x=list(type_counts.keys()), y=list(type_counts.values()),
                                 marker_color="#9b59b6"))
    fig_types.update_layout(title="Bobin Tipi Dağılımı", **_DARK,
                            xaxis_title="tip", yaxis_title="bobin sayısı", height=300)

    def _cell(fig):
        return html.Div(dcc.Graph(figure=fig, config={"displaylogo": False}),
                        style={"flex": "1 1 420px"})

    return html.Div(
        [
            html.Div("Bu grafikler gecikme modelinin öğrendiği gizli örüntüleri gösterir: "
                     "kötü hava ve uzun mesafe gecikmeyi artırır. LightGBM bunları yakalar "
                     "(test MAE ≈ 6.95 dk).", style={"color": "#9aa0b0", "fontSize": "12px",
                     "marginBottom": "8px"}),
            html.Div([_cell(fig_hist), _cell(fig_scatter)],
                     style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
            html.Div([_cell(fig_weather), _cell(fig_types)],
                     style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
        ]
    )


def build_delay_panel(delays: list[dict]) -> html.Div:
    """Aktif siparişlerin araçlarının TAHMİNİ (ML) ve GERÇEK gecikmelerini canlı gösterir.

    Her öğe: {order, vehicle, predicted, actual, line}. ML tahmini ile gerçeği yan yana
    göstermek, gecikme modelinin canlı doğruluğunu kanıtlar. Büyük gecikmeler vurgulanır.
    """
    if not delays:
        body = [html.Div("(aktif sevkiyat yok)", style={"color": "#777"})]
    else:
        header = html.Tr([html.Th(h, style={"padding": "3px 8px", "textAlign": "left"})
                          for h in ["Sipariş", "Araç", "Hat", "ML tahmini", "Gerçek"]])
        rows = [header]
        for d in delays[:12]:
            big = d["actual"] >= 60.0  # büyük gecikme vurgusu
            rows.append(html.Tr([
                html.Td(d["order"], style={"padding": "3px 8px"}),
                html.Td(d["vehicle"], style={"padding": "3px 8px"}),
                html.Td(d["line"], style={"padding": "3px 8px", "color": "#7fd0ff"}),
                html.Td(f"{d['predicted']:.0f} dk", style={"padding": "3px 8px", "color": "#f0c419"}),
                html.Td(f"{d['actual']:.0f} dk" + ("  ⚠" if big else ""),
                        style={"padding": "3px 8px",
                               "color": "#ff6b6b" if big else "#e0e0e8",
                               "fontWeight": "bold" if big else "normal"}),
            ]))
        body = [html.Table(rows, style={"borderCollapse": "collapse", "fontSize": "12px"})]
    return html.Div(body, style={"background": "#101015", "padding": "8px 10px",
                                 "height": "220px", "overflowY": "auto", "borderRadius": "6px"})
