"""WarehouseSimulator — simülasyonun ana orkestratörü ve zaman döngüsü.

İki kullanım modu aynı ilkelleri paylaşır (kod tekrarı yok):
  1) Politika-güdümlü: ``run(policy, horizon)`` uçtan uca koşturur.
  2) Dışarıdan-güdümlü (RL): ``pending_coil() / valid_actions() / apply_placement()``
     adım adım çağrılır.

Zaman modeli (docs/03 §9): sürekli zaman (saat). Bir öncelik kuyruğunda olaylar
(EventGenerator) ve zamanlanmış sevkiyatlar zaman sırasıyla işlenir. Yerleştirme
kararları "anlık"tır (saat yalnızca olaylarla ilerler); bir NEW_ORDER olayı
siparişin bobinlerini yerleştirme kuyruğuna sokar, zamanlanmış bir DISPATCH ise
sevkiyatı (rehandling sayımıyla) tetikler.
"""

from __future__ import annotations

import copy
import heapq
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from src.domain import (
    CoilStatus,
    Event,
    EventType,
    Order,
    OrderPriority,
    OrderStatus,
    SlotCoord,
    SteelCoil,
    Vehicle,
    WarehouseLayout,
)

from .constraints import can_place
from .dispatch import dispatch_order
from .event_generator import EventGenerator
from .metrics import SimulationMetrics, crane_distance, crane_move_time
from .warehouse_state import WarehouseState

if TYPE_CHECKING:
    # Politika sözleşmesi src/policies/base.py'de ABC olarak tanımlıdır; burada
    # yalnızca tip ipucu için (çalışma zamanında içe aktarılmaz -> döngü yok).
    from src.policies.base import PlacementPolicy


@dataclass
class StepResult:
    """Tek bir yerleştirme adımının sonucu (RL adım dönüşü)."""

    rehandling_delta: int = 0
    distance_delta: float = 0.0
    events_occurred: list[Event] = field(default_factory=list)
    done: bool = False


# Yerleştirme mesafesi için üretim bandı çıkış referansı (docs/01 §4: bay=0, zone=0).
_ENTRY_SLOT = SlotCoord(0, 0, 0)

# Sipariş başına PLANLANAN teslim süresi aralığı (saat). Her sipariş etkinleştiğinde
# kamyonu bu aralıktan bir süre sonrasına programlanır; böylece aciliyet saat
# ölçeğinde anlamlı varyans kazanır (kimi sevkiyat yakın, kimi uzak). Gerçek varış
# bunun üzerine aracın GERÇEK gecikmesi eklenerek bulunur (geç kamyon = geç sevk).
MIN_SERVICE_LEAD_H: float = 1.0
MAX_SERVICE_LEAD_H: float = 3.0


class WarehouseSimulator:
    """Depo simülasyonunun ana orkestratörü."""

    def __init__(
        self,
        coils: dict[str, SteelCoil],
        orders: list[Order],
        layout: WarehouseLayout,
        initial_placements: list[tuple[str, SlotCoord]],
        event_generator: EventGenerator,
        seed: int = 0,
        horizon_hours: float = 24.0,
        dispatch_lead_hours: float = 3.0,
        vehicles: dict[str, Vehicle] | None = None,
        enforce_affinity: bool = True,
        single_layer: bool = False,
    ) -> None:
        # Simülatör coil/order nesnelerini mutasyona uğrattığından (location, status,
        # priority) kendi bağımsız kopyalarını alır. Böylece girdi Scenario'su temiz
        # kalır ve aynı senaryodan kurulan farklı simülatörler birbirini etkilemez
        # (determinizm için kritik).
        self._coils = copy.deepcopy(coils)
        self._orders = copy.deepcopy(orders)
        # Araçlar yalnızca okunur (gecikme tahmini/affinity için); kopyalanmaz.
        self._vehicles = vehicles if vehicles is not None else {}
        # Bobin->sipariş->araç aramaları için id sözlüğü (kopyalanmış siparişlere işaret eder).
        self._order_by_id = {order.order_id: order for order in self._orders}
        self.layout = layout
        self._initial_placements = initial_placements
        self._event_gen = event_generator
        self._seed = seed
        self._horizon = horizon_hours
        self._dispatch_lead = dispatch_lead_hours
        # Senaryo modu bayrakları. Varsayılan = gerçekçi ağır-sanayi deposu (2 kat +
        # affinity). PPO'ya özel "raf benzeri" senaryo için: single_layer=True (yalnız
        # zemin → istif/rehandling yok) + enforce_affinity=False (kapı/hat ayrımı yok,
        # amaç saf vinç-mesafesi/rota optimizasyonu — RL'in doğal çalışma alanı).
        self._enforce_affinity = enforce_affinity
        self._single_layer = single_layer

        # Sipariş orijinal durum/önceliğini sakla; reset() bunları geri yükler ki
        # ardışık koşular birbirini etkilemesin (determinizm).
        self._orig_order = {o.order_id: (o.priority, o.status) for o in orders}

        # reset() ile dolacak çalışma-zamanı durumu.
        self.state: WarehouseState = WarehouseState(layout)
        self.metrics = SimulationMetrics()
        self._clock = 0.0
        self._rng = np.random.default_rng(seed)
        self._pending: deque[SteelCoil] = deque()
        self._queue: list[tuple[float, int, str, object]] = []
        self._seq = 0
        self._active: dict[str, dict] = {}
        self._dispatched: set[str] = set()
        self._pool_idx = 0
        self._done = False
        # B3 (opt-in): aciliyet değişiminde yeniden konumlandırmayı yapacak politika.
        # Yalnızca dashboard ``set_reposition_policy`` ile takar; eval'da None kalır →
        # değerlendirme sonuçları etkilenmez.
        self._reposition_policy: "PlacementPolicy | None" = None

        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        event_generator: EventGenerator | None = None,
        seed: int | None = None,
    ) -> None:
        """Depoyu ve tüm çalışma durumunu başlangıca alır; olay akışını yeniden kurar.

        Bobinler ve siparişler orijinal durumlarına döndürülür, başlangıç yerleşimi
        (initial_state) uygulanır, EventGenerator akışı ufuk boyunca kuyruğa yüklenir.
        Aynı seed ile reset sonrası koşu birebir tekrarlanabilir.

        RL için ucuz episode çeşitliliği: ``event_generator``/``seed`` verilirse coil
        ve sipariş havuzu yeniden derin kopyalanmadan (pahalı işlem) yalnızca olay
        akışı ve karar rng'si değiştirilir.
        """
        if event_generator is not None:
            self._event_gen = event_generator
        if seed is not None:
            self._seed = seed

        # Bobinleri sıfırla.
        for coil in self._coils.values():
            coil.location = None
            coil.status = CoilStatus.PENDING_PLACEMENT
            coil.rehandled = False
            # Bekleme süresi ve swap gerekçesi de çalışma-zamanı alanları: sıfırla.
            coil.stored_at = None
            coil.swap_reason = None
        # Siparişleri orijinal öncelik/duruma döndür.
        for order in self._orders:
            priority, status = self._orig_order[order.order_id]
            order.priority = priority
            order.status = status

        # Boş depoyu kurup başlangıç yerleşimini uygula.
        self.state = WarehouseState(self.layout)
        for coil_id, slot in self._initial_placements:
            coil = self._coils[coil_id]
            self.state.place(coil, slot)
            # Başlangıç envanteri t=0'dan beri depoda sayılır (bekleme süresi referansı).
            coil.stored_at = 0.0

        # Çalışma durumunu sıfırla.
        self.metrics = SimulationMetrics()
        self._clock = 0.0
        self._rng = np.random.default_rng(self._seed)
        self._pending = deque()
        self._queue = []
        self._seq = 0
        self._active = {}
        self._dispatched = set()
        self._pool_idx = 0
        self._done = False

        # Olay akışını kuyruğa yükle (her olay zaman damgasıyla).
        for event in self._event_gen.stream(self._horizon):
            self._push(event.timestamp, "EVENT", event)

    # ------------------------------------------------- dışarıdan-güdümlü mod
    def pending_coil(self) -> SteelCoil | None:
        """Yerleştirilmeyi bekleyen sıradaki bobin. Kuyruk boşsa olayları işleyerek
        bir bobin doğana (ya da simülasyon bitene) kadar zamanı ilerletir."""
        if not self._pending and not self._done:
            self.advance_to_next_decision()
        return self._pending[0] if self._pending else None

    def allowed_zones(self, coil: SteelCoil) -> set[int] | None:
        """Bir bobinin AFFINITY kısıtıyla konabileceği zone'lar (hat → bölge eşlemesi).

        GERÇEK ÜRÜN KISITI: bir bobin yalnızca kendi lojistik hattının (gemisinin/
        treninin) hizmet ettiği zone'lara konabilir — aksi hâlde sevkiyatta yanlış
        rıhtıma götürülür (fiziksel olarak geçersiz). Hattı/araç bilgisi yoksa (ör.
        siparişe atanmamış başlangıç bobini) None döner = tüm zone'lar serbest.
        """
        vehicle = self.vehicle_of(self.order_of(coil))
        if vehicle is None:
            return None
        line = vehicle.target_logistics_line
        return {z for z, lg in self.layout.zone_logistics.items() if lg == line}

    def valid_actions(self) -> list[SlotCoord]:
        """Bekleyen bobinin tüm kısıtları (fizik + mod bayrakları) sağlayarak konabileceği
        konumlar. RL'de bu liste action mask'e dönüştürülür.

        Mod bayrakları: ``single_layer`` ise yalnız zemin (layer 0) → istif yok;
        ``enforce_affinity`` ise yalnız bobinin hattının zone'ları.
        """
        coil = self.pending_coil()
        if coil is None:
            return []
        slots = self.state.valid_slots(coil)
        if self._single_layer:
            slots = [s for s in slots if s.layer == 0]
        if self._enforce_affinity:
            allowed = self.allowed_zones(coil)
            if allowed is not None:
                slots = [s for s in slots if s.zone in allowed]
        return slots

    def apply_placement(self, slot: SlotCoord) -> StepResult:
        """Bekleyen bobini verilen konuma yerleştirir, sonra bir sonraki karar
        noktasına kadar olayları işleyerek zamanı ilerletir.

        Geçersiz konum verilirse istisna fırlatır (geçerli kullanım: önce
        ``valid_actions()``'a bakmak)."""
        if not self._pending:
            raise RuntimeError("Yerleştirilecek bekleyen bobin yok.")
        coil = self._pending[0]
        if not can_place(self.state, coil, slot):
            raise ValueError(f"Geçersiz yerleştirme konumu: {slot}")
        # Affinity kısıtı (gerçek ürün): bobin yalnızca hattının zone'larına konabilir.
        # PPO "raf benzeri" senaryosunda (enforce_affinity=False) bu kapalıdır.
        if self._enforce_affinity:
            allowed = self.allowed_zones(coil)
            if allowed is not None and slot.zone not in allowed:
                raise ValueError(
                    f"Affinity ihlali: {coil.coil_id} zone {slot.zone}'a konamaz (izinli: {sorted(allowed)})"
                )

        place_dist = crane_distance(_ENTRY_SLOT, slot)
        self.state.place(coil, slot)
        # Bobinin depoya giriş anını işaretle (bekleme süresi/dwell time için).
        coil.stored_at = self._clock
        self.metrics.n_placements += 1
        self.metrics.total_crane_distance_m += place_dist
        self._pending.popleft()

        # Sonraki karar noktasına ilerle; bu sırada gerçekleşen sevkiyat/olayları topla.
        rehandling_delta, dispatch_dist, events = self._advance()
        return StepResult(
            rehandling_delta=rehandling_delta,
            distance_delta=place_dist + dispatch_dist,
            events_occurred=events,
            done=self.is_done(),
        )

    def relocate(self, coil: SteelCoil, new_slot: SlotCoord) -> float:
        """Bir bobini mevcut konumdan başka bir geçerli slota TAŞIR (1 vinç hamlesi).

        SWAP MEKANİZMASI için kullanılır: Heuristic/MLHeuristic, kapıya yakın "prime"
        bir slotta daha az acil bir bobin bulunca onu uygun bir alternatife taşıyıp
        prime'ı boşaltır, ardından yeni acil bobini prime'a yerleştirir. Bu rehandling
        DEĞİLDİR (sevkiyat sırasında zorunlu engel kaldırma değil) ama vinç maliyeti
        (mesafe + süre) sayılır.

        Çağıran, ``new_slot``'un ``can_place`` koşullarını sağladığını garanti etmelidir
        (politikanın swap karar mantığı bunu kontrol eder). Dönüş: yapılan mesafe (m).

        Not: ``stored_at`` KORUNUR — bobin depoda kalmaya devam ediyor, yalnızca yeri
        değişiyor; bekleme süresi sıfırlanmaz. (``state.place`` bu alana dokunmaz.)
        """
        if coil.location is None:
            raise ValueError(f"Bobin depoda değil, taşınamaz: {coil.coil_id}")
        old_slot = coil.location
        distance = crane_distance(old_slot, new_slot)
        self.state.remove(coil)
        self.state.place(coil, new_slot)
        self.metrics.total_crane_distance_m += distance
        self.metrics.total_loading_time_min += crane_move_time(distance)
        return distance

    def advance_to_next_decision(self) -> None:
        """Yeni bir yerleştirme kararı gerekene kadar olayları işleyip zamanı ilerletir."""
        self._advance()

    def skip_pending(self) -> SteelCoil | None:
        """Bekleyen ilk bobini yerleştirmeden kuyruktan düşürür (geçerli konum yoksa).

        Depo dolduğunda veya bir bobin yerleştirilemediğinde çağrılır; çekirdeğin
        taşma davranışıdır. Düşürülen bobini döndürür (kuyruk boşsa None)."""
        if self._pending:
            return self._pending.popleft()
        return None

    def is_done(self) -> bool:
        """Simülasyon ufku doldu ve işlenecek bobin kalmadı mı?"""
        return self._done and not self._pending

    @property
    def clock(self) -> float:
        """Anlık simülasyon saati (saat cinsinden) — dashboard ve loglama için."""
        return self._clock

    @property
    def horizon(self) -> float:
        """Simülasyon ufku (saat) — gözlemde zaman oranı normalizasyonu için."""
        return self._horizon

    @property
    def pending_count(self) -> int:
        """Yerleştirilmeyi bekleyen bobin sayısı (gözlemde kuyruk uzunluğu için)."""
        return len(self._pending)

    # ---------------------------------------- bobin -> sipariş -> araç aramaları
    def order_of(self, coil: SteelCoil) -> Order | None:
        """Bir bobinin ait olduğu siparişi döndürür (yoksa None).

        Politikalar (affinity ve aciliyet için) bu metodu kullanır.
        """
        if coil.order_id is None:
            return None
        return self._order_by_id.get(coil.order_id)

    def vehicle_of(self, order: Order | None) -> Vehicle | None:
        """Bir siparişi karşılayan aracı döndürür (yoksa None).

        MLHeuristicPolicy bu araçtan gecikme tahmini için özellik çıkarır.
        """
        if order is None:
            return None
        return self._vehicles.get(order.vehicle_id)

    def planned_dispatch_time(self, order: Order | None) -> float | None:
        """Etkin bir siparişin PLANLANAN (gecikmesiz) sevkiyat anını (saat) döndürür.

        Politikalar aciliyeti buradan hesaplar: yakında sevk edilecek (kalan süresi
        az) bobin daha aciledir. Sipariş etkin değilse (ör. henüz etkinleşmemiş
        başlangıç bobini) None döner. ML politikası buna tahmini gecikmeyi ekler.
        """
        if order is None:
            return None
        entry = self._active.get(order.order_id)
        return entry["planned_dispatch"] if entry is not None else None

    # ------------------------------------------------------ politika-güdümlü mod
    def run(self, policy: PlacementPolicy, horizon_hours: float | None = None) -> SimulationMetrics:
        """Verilen politikayı simülasyon ufku boyunca uçtan uca koşturur.

        İç döngü: pending_coil -> policy.decide -> apply_placement. Her kararın
        hesaplama süresi ölçülüp ``decision_times_ms``e eklenir.
        """
        if horizon_hours is not None:
            self._horizon = horizon_hours
        self.reset()

        while True:
            coil = self.pending_coil()
            if coil is None:
                break
            valid = self.valid_actions()
            if not valid:
                # Depo dolu / bu bobin için geçerli konum yok -> yerleştirilemez, atla.
                self._pending.popleft()
                continue
            t0 = perf_counter()
            slot = policy.decide(coil, self)
            self.metrics.decision_times_ms.append((perf_counter() - t0) * 1000.0)
            self.apply_placement(slot)

        self.metrics.final_fill_ratio = self.state.fill_ratio()
        return self.metrics

    # ----------------------------------------------------------- iç yardımcılar
    def _push(self, time: float, kind: str, data: object) -> None:
        """Kuyruğa zamanlanmış bir öğe ekler (seq, eşit zamanlarda kararlı sıralama sağlar)."""
        heapq.heappush(self._queue, (time, self._seq, kind, data))
        self._seq += 1

    def _advance(self) -> tuple[int, float, list[Event]]:
        """Kuyruğu, yeni bir bobin bekleyene veya ufuk dolana kadar işler.

        Dönüş: bu ilerleme sırasında biriken (rehandling, sevkiyat mesafesi, olaylar).
        Metrikler ilgili yerlerde (yerleştirme/sevkiyat) ayrıca güncellenir; bu dönüş
        yalnızca StepResult raporlaması içindir.
        """
        rehandling_delta = 0
        dispatch_dist = 0.0
        events: list[Event] = []

        while not self._pending and self._queue and not self._done:
            time, _seq, kind, data = heapq.heappop(self._queue)
            if time > self._horizon:
                self._done = True
                break
            self._clock = time
            if kind == "DISPATCH":
                order_id = data
                # Gecikme ile ertelenmiş veya iptal edilmiş sevkiyatlar atlanır.
                if (
                    order_id in self._active
                    and self._active[order_id]["dispatch_time"] == time
                ):
                    result = self._do_dispatch(order_id)
                    rehandling_delta += result.rehandling_count
                    dispatch_dist += result.distance_m
            else:  # EVENT
                events.append(data)
                self._handle_event(data)

        # Kuyruk ve bekleyen tükendiyse simülasyon biter.
        if not self._queue and not self._pending:
            self._done = True

        return rehandling_delta, dispatch_dist, events

    def _do_dispatch(self, order_id: str):
        """Bir siparişin sevkiyatını yürütür: bobinleri al, rehandling/mesafe/süre ekle."""
        entry = self._active.pop(order_id)
        order: Order = entry["order"]
        result = dispatch_order(self.state, order)
        self.metrics.rehandling_count += result.rehandling_count
        self.metrics.total_crane_distance_m += result.distance_m
        self.metrics.total_loading_time_min += result.loading_time_min
        self.metrics.n_dispatches += 1
        order.status = OrderStatus.FULFILLED
        self._dispatched.add(order_id)
        return result

    def _handle_event(self, event: Event) -> None:
        """Dinamik olayı tipine göre işler (Katman 1 tepkisel yeniden planlama)."""
        et = event.event_type
        if et == EventType.NEW_ORDER:
            self._activate_next_order()
        elif et == EventType.VEHICLE_DELAY:
            self._apply_vehicle_delay()
        elif et == EventType.CANCEL_ORDER:
            self._cancel_active_order()
        elif et == EventType.PRIORITY_CHANGE:
            self._change_priority()
        # PEAK_LOAD Aşama 2'de durum mutasyonu gerektirmez (Aşama 7 dayanıklılık analizi).

    def _activate_next_order(self) -> None:
        """Sipariş havuzundan sıradaki açık siparişi etkinleştirir.

        Siparişin henüz yerleştirilmemiş bobinlerini bekleme kuyruğuna sokar ve
        ``dispatch_lead`` saat sonrasına bir sevkiyat zamanlar.
        """
        while self._pool_idx < len(self._orders):
            order = self._orders[self._pool_idx]
            self._pool_idx += 1
            if order.status == OrderStatus.CANCELLED:
                continue
            order.status = OrderStatus.IN_PROGRESS
            # Planlanan teslim: sipariş-başına saat-ölçekli bir teslim süresi sonrası.
            service_lead = float(self._rng.uniform(MIN_SERVICE_LEAD_H, MAX_SERVICE_LEAD_H))
            planned_dispatch = self._clock + service_lead
            # Gerçek varış = planlanan + aracın GERÇEK gecikmesi (geç kamyon geç sevk eder).
            vehicle = self._vehicles.get(order.vehicle_id)
            delay_hours = (vehicle.delay_minutes / 60.0) if vehicle is not None else 0.0
            actual_dispatch = planned_dispatch + delay_hours
            # planned_dispatch: politikaların gördüğü (gecikmesiz) referans;
            # dispatch_time: sevkiyatın gerçekten tetiklendiği (gecikmeli) an.
            self._active[order.order_id] = {
                "order": order,
                "dispatch_time": actual_dispatch,
                "planned_dispatch": planned_dispatch,
            }
            # Depoda olmayan (henüz yerleştirilmemiş) bobinleri yerleştirme kuyruğuna al.
            for coil_id in order.coil_ids:
                coil = self._coils.get(coil_id)
                if coil is not None and coil.location is None:
                    self._pending.append(coil)
            self._push(actual_dispatch, "DISPATCH", order.order_id)
            return

    def inject_peak(self, n_orders: int = 5) -> int:
        """Zirve/kriz senaryosu: birden çok siparişi ANINDA etkinleştirir (ani yük artışı).

        Dashboard'daki "Olay Tetikle" düğmesi bunu çağırır; bekleyen bobin kuyruğunu
        aniden doldurarak sistemin yoğunluk altındaki tepkisini görünür kılar.
        Dönüş: gerçekten etkinleştirilen sipariş sayısı (havuz biterse daha az olabilir).
        """
        activated = 0
        for _ in range(n_orders):
            before = self._pool_idx
            self._activate_next_order()
            if self._pool_idx == before:
                break  # sipariş havuzu tükendi
            activated += 1
        return activated

    def _apply_vehicle_delay(self) -> None:
        """Etkin bir siparişe ÖNGÖRÜLEMEZ EK gecikme ekler (sürpriz olay).

        Önemli ayrım: aracın TAHMİN EDİLEBİLİR gecikmesi (vehicle.delay_minutes)
        aktivasyonda gerçek sevkiyat zamanına zaten gömülüdür ve ML bunu tahmin eder.
        Bu olay ise onun ÜSTÜNE, hiçbir politikanın öngöremeyeceği ek bir gecikmedir
        (gerçek dünyadaki son-dakika aksaklık). Planlanan zamanı (politika referansı)
        değiştirmez; yalnızca gerçekleşen sevkiyatı öteler."""
        if not self._active:
            return
        order_id = self._pick_active()
        extra = float(self._rng.uniform(1.0, 5.0))
        new_time = self._active[order_id]["dispatch_time"] + extra
        self._active[order_id]["dispatch_time"] = new_time
        # Yeni zamanlı sevkiyat ekle; eski kuyruk öğesi zaman uyuşmazlığından atlanır.
        self._push(new_time, "DISPATCH", order_id)

    def _cancel_active_order(self) -> None:
        """Etkin bir siparişi iptal eder; sevkiyatı düşer (bobinleri depoda kalır)."""
        if not self._active:
            return
        order_id = self._pick_active()
        self._active[order_id]["order"].status = OrderStatus.CANCELLED
        del self._active[order_id]

    def set_reposition_policy(self, policy: "PlacementPolicy | None") -> None:
        """B3: aciliyet değişiminde yeniden konumlandırmayı yapacak politikayı takar.

        Yalnızca dashboard çağırır (opt-in görselleştirme özelliği). Değerlendirme
        hattı (``evaluate_policy`` → ``run``) bunu ASLA çağırmaz; böylece yeniden
        konumlandırma eval metriklerini değiştirmez (determinizm + rapor tutarlılığı).
        """
        self._reposition_policy = policy

    def _change_priority(self) -> None:
        """Etkin bir siparişin önceliğini yükseltir (aciliyet değişimi).

        B3 etkinse (``_reposition_policy`` takılı): bu siparişin YERLEŞMİŞ bobinlerini
        güncel (zaman-tabanlı) aciliyetlerine göre yeniden değerlendirir; politika
        gerekirse onları daha iyi konumlara taşır (canlı dinamik uyum görseli).
        """
        if not self._active:
            return
        order_id = self._pick_active()
        order: Order = self._active[order_id]["order"]
        order.priority = OrderPriority.URGENT
        if self._reposition_policy is not None:
            for coil_id in order.coil_ids:
                coil = self._coils.get(coil_id)
                if coil is not None and coil.location is not None:
                    self._reposition_policy.reposition_on_priority_change(coil, self)

    def _pick_active(self) -> str:
        """Etkin siparişlerden deterministik-rastgele birini seçer (sim rng ile)."""
        keys = list(self._active.keys())
        return keys[int(self._rng.integers(0, len(keys)))]
