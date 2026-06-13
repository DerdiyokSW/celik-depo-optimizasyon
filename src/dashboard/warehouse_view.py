"""Depo durumunu 3B Plotly figürüne çeviren görselleştirme.

``WarehouseState``'i bir ``Scatter3d`` figürüne dönüştürür. GERÇEK metre
koordinatları kullanılır (bay×3m, zone×15m, layer×1.5m) — böylece sahne gerçek
depo oranlarında geniş bir alana yayılır, ince şerit gibi sıkışmaz. Yükleme
KAPISI (bay 0 önü) görünür çizilir; dolu konumlar aciliyete göre renklenir; yeri
değişen (rehandled) bobinler kırmızı kenarlı elmasla vurgulanır.

Kamera/zoom akış sırasında korunur (``uirevision``): her güncellemede figür
yeniden kurulsa bile kullanıcının döndürme/yakınlaştırması sıfırlanmaz.
"""

from __future__ import annotations

import plotly.graph_objects as go

from src.domain import SlotCoord, WarehouseLayout
from src.simulation.metrics import BAY_SPACING_M, LAYER_HEIGHT_M, ZONE_SPACING_M
from src.simulation.warehouse_state import WarehouseState

FILLED_TRACE_NAME = "Dolu"
EMPTY_TRACE_NAME = "Boş"
REHANDLED_TRACE_NAME = "Yeri Değişen"
DOOR_TRACE_NAME = "Yükleme Kapısı"
URGENCY_COLORSCALE = "Plasma"  # düşük (soğuk) -> yüksek aciliyet (sıcak/sarı)


def _coords(bay: int, zone: int, layer: int) -> tuple[float, float, float]:
    """Izgara indekslerini gerçek metre koordinatlarına çevirir (x=bay, y=zone, z=kat)."""
    return bay * BAY_SPACING_M, zone * ZONE_SPACING_M, layer * LAYER_HEIGHT_M


def _dwell_text(coil, now: float | None) -> str:
    """Bobinin bekleme süresini (dwell time) hover satırı olarak biçimlendirir.

    Bekleme = şu anki saat − depoya giriş saati (``stored_at``). Sim saati saat
    cinsindendir; okunabilirlik için saat olarak gösterilir. ``now`` veya
    ``stored_at`` yoksa boş döner (eski test çağrıları ``now`` geçmez).
    """
    if now is None or coil.stored_at is None:
        return ""
    dwell_h = max(0.0, now - coil.stored_at)
    return f"<br>Bekleme: {dwell_h:.1f} sa"


def _swap_text(coil) -> str:
    """Taşınan bobin için kararın gerekçesini hover'a yazar (B1 swap veya B3 reposition).

    Swap (yerleştirmede yer açma): "Şu acil bobin için açıldı; taşıma Xm < alternatif Ym".
    Reposition (B3, aciliyet artışında): "Aciliyet arttı → daha iyi konuma taşındı".
    """
    r = coil.swap_reason
    if not r:
        return ""
    if r.get("kind") == "reposition":
        return (
            "<br><b>YENİDEN KONUMLANDI (aciliyet arttı)</b>"
            f"<br>aciliyet {r['trigger_urgency']} → daha erişilebilir konuma taşındı"
            f"<br>{tuple(r['moved_from'])} → {tuple(r['moved_to'])}"
        )
    return (
        "<br><b>YERİ DEĞİŞTİ (SWAP)</b>"
        f"<br>{r['trigger_coil']} (aciliyet {r['trigger_urgency']}) için açıldı"
        f"<br>taşıma {r['swap_cost_m']}m < alternatif {r['alt_cost_m']}m"
    )


def render_warehouse(
    state: WarehouseState,
    layout: WarehouseLayout,
    now: float | None = None,
    color_mode: str = "urgency",
) -> go.Figure:
    """Anlık depo durumundan 3B Plotly figürü üretir (gerçek metre ölçeğinde).

    ``now`` verilirse (anlık sim saati) hover'a bobin bekleme süresi (dwell time)
    eklenir. Swap ile taşınan veya sevkiyatta yer değiştiren bobinler kırmızı
    kenarlı elmasla işaretlenir; hover'larında gerekçe gösterilir.

    ``color_mode`` (A3): "urgency" → renk aciliyete göre (varsayılan); "dwell" →
    renk bekleme süresine göre (uzun bekleyen = vurgulu). Bekleme modu ``now``
    gerektirir ve depodaki en uzun beklemeye normalize edilir [0,1].
    """
    # Bekleme modunda renk normalizasyonu için depodaki en uzun beklemeyi bul.
    max_dwell = 0.0
    if color_mode == "dwell" and now is not None:
        dwells = [now - c.stored_at for c in state.stored_coils() if c.stored_at is not None]
        max_dwell = max(dwells) if dwells else 0.0

    def _color(coil) -> float:
        """Bobinin renk değeri (0..1): moda göre aciliyet veya normalize bekleme."""
        if color_mode == "dwell" and now is not None and coil.stored_at is not None and max_dwell > 0:
            return (now - coil.stored_at) / max_dwell
        return coil.urgency_score

    filled_x: list[float] = []; filled_y: list[float] = []; filled_z: list[float] = []
    filled_color: list[float] = []; filled_hover: list[str] = []

    reh_x: list[float] = []; reh_y: list[float] = []; reh_z: list[float] = []
    reh_color: list[float] = []; reh_hover: list[str] = []

    empty_x: list[float] = []; empty_y: list[float] = []; empty_z: list[float] = []

    for zone in range(layout.n_zones):
        line = layout.zone_logistics.get(zone)
        line_label = line.value if line is not None else "-"
        for bay in range(layout.n_bays):
            for layer in range(layout.n_layers):
                x, y, z = _coords(bay, zone, layer)
                coil = state.coil_at(SlotCoord(zone, bay, layer))
                if coil is None:
                    empty_x.append(x); empty_y.append(y); empty_z.append(z)
                    continue
                # "Yeri değişen" = swap ile taşınmış VEYA sevkiyatta engel olup taşınmış.
                moved = coil.rehandled or coil.swap_reason is not None
                # Taşınma açıklaması: önce swap gerekçesi (varsa), yoksa sevkiyat engeli.
                if coil.swap_reason is not None:
                    moved_text = _swap_text(coil)
                elif coil.rehandled:
                    moved_text = "<br><b>YERİ DEĞİŞTİ (sevkiyat engeli)</b>"
                else:
                    moved_text = ""
                hover = (
                    f"{coil.coil_id}<br>{coil.coil_type.value}<br>{coil.weight_ton:.1f} ton"
                    f"<br>zone {zone} · bay {bay} · kat {layer}<br>hat: {line_label}"
                    f"<br>aciliyet: {coil.urgency_score:.2f}"
                    + _dwell_text(coil, now)
                    + moved_text
                )
                if moved:
                    reh_x.append(x); reh_y.append(y); reh_z.append(z)
                    reh_color.append(_color(coil)); reh_hover.append(hover)
                else:
                    filled_x.append(x); filled_y.append(y); filled_z.append(z)
                    filled_color.append(_color(coil)); filled_hover.append(hover)

    figure = go.Figure()

    # Boş konumlar: soluk arka plan ızgarası.
    figure.add_trace(
        go.Scatter3d(
            x=empty_x, y=empty_y, z=empty_z, mode="markers",
            marker=dict(size=2, color="rgba(120,120,140,0.12)", symbol="square"),
            name=EMPTY_TRACE_NAME, hoverinfo="skip",
        )
    )
    # Yükleme kapıları: her zone'un önünde (bay 0 hizasında, biraz ileride), yeşil.
    door_x = [-BAY_SPACING_M] * layout.n_zones
    door_y = [zone * ZONE_SPACING_M for zone in range(layout.n_zones)]
    door_z = [0.0] * layout.n_zones
    door_hover = [
        f"Yükleme Kapısı — zone {zone} ({layout.zone_logistics.get(zone).value if layout.zone_logistics.get(zone) else '-'})"
        for zone in range(layout.n_zones)
    ]
    figure.add_trace(
        go.Scatter3d(
            x=door_x, y=door_y, z=door_z, mode="markers",
            marker=dict(size=10, color="#2ecc71", symbol="square"),
            text=door_hover, hoverinfo="text", name=DOOR_TRACE_NAME,
        )
    )
    # Dolu konumlar: seçili moda göre renk skalasıyla (aciliyet veya bekleme).
    colorbar_title = "Bekleme" if color_mode == "dwell" else "Aciliyet"
    figure.add_trace(
        go.Scatter3d(
            x=filled_x, y=filled_y, z=filled_z, mode="markers",
            marker=dict(
                size=6, color=filled_color, colorscale=URGENCY_COLORSCALE,
                cmin=0.0, cmax=1.0,
                # Colorbar'ı sağ kenara, kısa ve ince konumlandır — lejantla (alt yatay)
                # ve eksenlerle çakışmaz.
                colorbar=dict(
                    title=dict(text=colorbar_title, side="right"),
                    len=0.7, thickness=14, x=1.0, xanchor="left",
                    y=0.5, yanchor="middle",
                ),
            ),
            text=filled_hover, hoverinfo="text", name=FILLED_TRACE_NAME,
        )
    )
    # Yeri değişen (rehandled) bobinler: elmas + kırmızı kenarlık.
    figure.add_trace(
        go.Scatter3d(
            x=reh_x, y=reh_y, z=reh_z, mode="markers",
            marker=dict(
                size=8, symbol="diamond", color=reh_color, colorscale=URGENCY_COLORSCALE,
                cmin=0.0, cmax=1.0, line=dict(width=5, color="#ff3b3b"),
            ),
            text=reh_hover, hoverinfo="text", name=REHANDLED_TRACE_NAME,
        )
    )

    figure.update_layout(
        scene=dict(
            # Eksen başlıkları okunabilir/açıklayıcı (A2): mühendis "Bay ne demek"i
            # bilmeden de eksenden anlasın.
            xaxis_title="Bay → kapıdan uzaklık (m)",
            yaxis_title="Zone → bölge/koridor (m)",
            zaxis_title="Kat → istif (m)",
            # Gerçek oran çok yassı olurdu; tabanı geniş tutup yüksekliği görünür kıl.
            aspectmode="manual",
            aspectratio=dict(x=2.2, y=1.3, z=0.45),
            camera=dict(eye=dict(x=1.9, y=-1.9, z=1.3)),
            # KRİTİK: 3B kamera/zoom kalıcılığı SCENE seviyesinde tutulur. Yalnızca
            # layout.uirevision yeterli değil — sahne kamerası için scene.uirevision
            # gerekir; aksi hâlde her figür güncellemesinde (akış/adım) kullanıcının
            # döndürme/yakınlaştırması sıfırlanır (bu hata buydu).
            uirevision="warehouse-view",
        ),
        margin=dict(l=0, r=40, t=34, b=0),
        # Başlık kısa ve sol üstte; renk semantiği lejant + colorbar'da (çakışma yok).
        title=dict(text="Depo Durumu (3B) — kapı: ön (bay 0) · sarı=acil, mavi=bekleyebilir",
                   x=0.01, xanchor="left", y=0.99, yanchor="top", font=dict(size=13)),
        showlegend=True,
        # Lejantı YATAY olarak en üste taşı (sağ üstteki dikey lejant colorbar ile
        # çakışıyordu); plot alanının üstünde tek satır.
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        # Akış sırasında kullanıcı etkileşimi (kamera dışı) da korunur.
        uirevision="warehouse-view",
    )
    return figure
