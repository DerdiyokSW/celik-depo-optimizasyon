"""Üretilen veri setinin docs/01 §10'daki 8 doğrulama kuralını uygulayan modül.

Veri üretici çıktıyı yazmadan ÖNCE bu kontroller koşar; ihlal varsa ``ValueError``
fırlatılır ("uydurma metrik / geçersiz veri yok" ilkesi). Aynı modül ileride
simülasyon çekirdeği tarafından veri yüklenirken de kullanılacaktır (tek doğruluk
kaynağı, tekrarsız).
"""

from __future__ import annotations

import pandas as pd

from src.domain import CoilType, SlotCoord, WarehouseLayout, max_stack_layer_for


def validate_all(
    coils: pd.DataFrame,
    vehicles: pd.DataFrame,
    orders: pd.DataFrame,
    layout: dict,
    initial_state: dict,
) -> None:
    """Tüm veri setini 8 kurala göre doğrular; ihlalde ``ValueError`` fırlatır.

    Kurallar sırayla denetlenir; ilk ihlalde anlamlı bir Türkçe mesajla durur.
    Hepsi geçerse sessizce döner.
    """
    wl = WarehouseLayout.from_dict(layout)
    _check_id_uniqueness(coils, vehicles, orders)          # kural 1
    _check_referential_integrity(coils, vehicles, orders)  # kural 2
    _check_delay_non_negative(vehicles)                    # kural 7
    _check_type_layer_consistency(coils)                   # kural 8
    _check_initial_state(coils, wl, initial_state)         # kural 3,4,5,6


def _check_id_uniqueness(
    coils: pd.DataFrame, vehicles: pd.DataFrame, orders: pd.DataFrame
) -> None:
    """Kural 1: Tüm coil_id, vehicle_id, order_id benzersiz olmalı."""
    if coils["coil_id"].duplicated().any():
        raise ValueError("Doğrulama (kural 1): coil_id değerleri benzersiz değil.")
    if vehicles["vehicle_id"].duplicated().any():
        raise ValueError("Doğrulama (kural 1): vehicle_id değerleri benzersiz değil.")
    if orders["order_id"].duplicated().any():
        raise ValueError("Doğrulama (kural 1): order_id değerleri benzersiz değil.")


def _check_referential_integrity(
    coils: pd.DataFrame, vehicles: pd.DataFrame, orders: pd.DataFrame
) -> None:
    """Kural 2: Sipariş–araç–bobin bağları geçerli ve çift yönlü tutarlı olmalı."""
    valid_vehicle_ids = set(vehicles["vehicle_id"])
    valid_coil_ids = set(coils["coil_id"])
    # Bobin -> ait olduğu sipariş haritası (coils tablosundan).
    coil_to_order = dict(zip(coils["coil_id"], coils["order_id"]))

    # Sipariş tarafından beyan edilen bobin üyelikleri.
    order_to_coils: dict[str, set[str]] = {}
    for row in orders.itertuples():
        order_id = row.order_id
        if row.vehicle_id not in valid_vehicle_ids:
            raise ValueError(
                f"Doğrulama (kural 2): {order_id} geçersiz araca işaret ediyor: {row.vehicle_id}."
            )
        coil_ids = list(row.coil_ids)
        order_to_coils[order_id] = set(coil_ids)
        for coil_id in coil_ids:
            if coil_id not in valid_coil_ids:
                raise ValueError(
                    f"Doğrulama (kural 2): {order_id} geçersiz bobine işaret ediyor: {coil_id}."
                )
            # İleri yön: bobinin order_id'si bu siparişe eşleşmeli.
            if coil_to_order.get(coil_id) != order_id:
                raise ValueError(
                    f"Doğrulama (kural 2): {coil_id} bobini {order_id} listesinde ama "
                    f"order_id'si '{coil_to_order.get(coil_id)}' (çift yönlü bağ kopuk)."
                )

    # Ters yön: order_id'si dolu her bobin, o siparişin coil_ids listesinde olmalı.
    for coil_id, order_id in coil_to_order.items():
        if order_id is None:
            continue
        if order_id not in order_to_coils or coil_id not in order_to_coils[order_id]:
            raise ValueError(
                f"Doğrulama (kural 2): {coil_id} bobini '{order_id}' siparişine bağlı "
                f"ama o siparişin coil_ids listesinde yok (çift yönlü bağ kopuk)."
            )


def _check_delay_non_negative(vehicles: pd.DataFrame) -> None:
    """Kural 7: Hiçbir delay_minutes negatif olamaz."""
    if (vehicles["delay_minutes"] < 0).any():
        raise ValueError("Doğrulama (kural 7): negatif delay_minutes değeri var.")


def _check_type_layer_consistency(coils: pd.DataFrame) -> None:
    """Kural 8: max_stack_layer, bobin tipinin izin verdiği değerle eşleşmeli."""
    expected = coils["coil_type"].map(lambda v: max_stack_layer_for(CoilType(v)))
    mismatch = coils["max_stack_layer"] != expected
    if mismatch.any():
        bad = coils.loc[mismatch, "coil_id"].iloc[0]
        raise ValueError(
            f"Doğrulama (kural 8): {bad} bobininde tip–kat tutarsızlığı var."
        )


def _check_initial_state(
    coils: pd.DataFrame, wl: WarehouseLayout, initial_state: dict
) -> None:
    """Kural 3,4,5,6: Başlangıç yerleşiminin konum/süreklilik/ağırlık/kapasite geçerliliği."""
    weight_by_coil = dict(zip(coils["coil_id"], coils["weight_ton"]))
    valid_coil_ids = set(coils["coil_id"])

    # Konum -> ağırlık eşlemesi ve doluluk kümesi kur.
    occupied: dict[tuple[int, int, int], float] = {}
    for p in initial_state["placements"]:
        coil_id = p["coil_id"]
        if coil_id not in valid_coil_ids:
            raise ValueError(
                f"Doğrulama (kural 2): başlangıç durumunda geçersiz bobin: {coil_id}."
            )
        coord = SlotCoord(p["zone"], p["bay"], p["layer"])
        # Kural 3: konum depo sınırları içinde olmalı.
        if not wl.is_valid_coord(coord):
            raise ValueError(
                f"Doğrulama (kural 3): geçersiz konum {coord} (bobin {coil_id})."
            )
        occupied[(coord.zone, coord.bay, coord.layer)] = weight_by_coil[coil_id]

    # Kural 4 (süreklilik) ve Kural 5 (ağırlık): her dolu üst kat için alt kat dolu
    # olmalı ve üstteki bobin alttakinden hafif olmalı.
    for (zone, bay, layer), weight in occupied.items():
        if layer > 0:
            below_key = (zone, bay, layer - 1)
            if below_key not in occupied:
                raise ValueError(
                    f"Doğrulama (kural 4): ({zone},{bay},{layer}) dolu ama altı boş "
                    f"(süreklilik ihlali)."
                )
            if not (weight < occupied[below_key]):
                raise ValueError(
                    f"Doğrulama (kural 5): ({zone},{bay},{layer}) bobini alttakinden "
                    f"hafif değil (ağırlık kuralı ihlali)."
                )

    # Kural 6: her zone'daki toplam ağırlık zone tonaj limitini aşmamalı.
    zone_total: dict[int, float] = {}
    for (zone, _bay, _layer), weight in occupied.items():
        zone_total[zone] = zone_total.get(zone, 0.0) + weight
    for zone, total in zone_total.items():
        if total > wl.zone_max_weight_ton[zone]:
            raise ValueError(
                f"Doğrulama (kural 6): zone {zone} toplam ağırlığı {total:.1f} ton, "
                f"limit {wl.zone_max_weight_ton[zone]:.1f} ton aşıldı."
            )
