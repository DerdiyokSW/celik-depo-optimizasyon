"""GEÇİCİ — Gerçek dashboard 3B dijital-ikiz görünümünü poster için PNG'ye render eder.

Ana (2 kat) ve raf (tek kat) senaryolarında PPO ile ~%25 dolulukta render. TNR font.
Bitince silinebilir. Çıktı: runs/evaluation/poster/dash_*.png
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.dashboard.app import _load_controller
from src.dashboard.warehouse_view import render_warehouse

OUT = Path("runs/evaluation/poster")
ctrl, _cmp, _mae = _load_controller(rack_mode=False)


def render(scenario_rack: bool, policy: str, title: str, fname: str, target_fill: float = 0.50):
    """Senaryoyu kurar, hedef doluluğa kadar koşar, 3B görünümü PNG'ye basar."""
    ctrl.set_rack_mode(scenario_rack)
    ctrl.set_policy(policy)
    guard = 0
    while ctrl.sim.state.fill_ratio() < target_fill and not ctrl.sim.is_done() and guard < 5000:
        ctrl.step()
        guard += 1
    ctrl.stamp_all_urgency()
    fig = render_warehouse(ctrl.sim.state, ctrl.sim.layout, now=ctrl.sim.clock)
    fig.update_layout(
        title=title, width=1150, height=720,
        font=dict(family="Times New Roman", size=15),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_image(str(OUT / fname), scale=2)
    print(f"{fname}: doluluk=%{ctrl.sim.state.fill_ratio()*100:.0f}, "
          f"yerleştirme={ctrl.sim.metrics.n_placements}, vinç={ctrl.sim.metrics.total_crane_distance_m:.0f}m")


render(False, "PPO", "Dijital İkiz — Ana Senaryo (2 kat + affinity) · PPO", "dash_main_ppo.png")
render(True, "PPO", "Dijital İkiz — Raf Senaryosu (tek kat, rota) · PPO", "dash_rack_ppo.png")
render(False, "Heuristic", "Dijital İkiz — Ana Senaryo (2 kat) · Heuristic", "dash_main_heu.png")
print("DONE")
