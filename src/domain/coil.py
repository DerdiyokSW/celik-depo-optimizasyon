"""Çelik bobin modeli (``SteelCoil``) ve tipe bağlı fiziksel parametreleri.

docs/01 §2 ile birebir uyumludur. Ayrıca her bobin tipinin ağırlık/çap/genişlik
aralıklarını ve maksimum istif katını tek bir doğruluk kaynağı olarak burada
tanımlar (``COIL_TYPE_SPECS``); hem veri üretici bu aralıklardan örnekler, hem
doğrulama ve testler bu aralıklara karşı kontrol yapar. Böylece sözleşmedeki
sayılar kod içinde dağılıp birbirinden kopmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .enums import CoilStatus, CoilType, QualityClass
from .warehouse import SlotCoord


@dataclass(frozen=True)
class CoilTypeSpec:
    """Bir bobin tipinin fiziksel örnekleme aralıkları ve istif sınırı.

    ``frozen=True`` çünkü bu değerler sözleşmeden gelen sabit tanımlardır;
    çalışma zamanında asla değişmemelidir.
    """

    weight_min: float
    weight_max: float
    width_min: int
    width_max: int
    diameter_min: int
    diameter_max: int
    max_stack_layer: int


# docs/01 §2 tablosunun makinece okunabilir tek kaynağı. Üreticinin örneklediği
# ve testlerin doğruladığı aralıklar buradan gelir.
COIL_TYPE_SPECS: dict[CoilType, CoilTypeSpec] = {
    # sıcak hadde: en ağır, en geniş, sadece 2 kat istiflenebilir
    CoilType.HOT_ROLLED: CoilTypeSpec(20.0, 30.0, 1000, 2000, 1500, 2100, 2),
    # soğuk hadde: orta ağırlık, 3 kat
    CoilType.COLD_ROLLED: CoilTypeSpec(12.0, 22.0, 800, 1500, 1200, 1600, 3),
    # galvaniz: en hafif, 3 kat
    CoilType.GALVANIZED: CoilTypeSpec(10.0, 18.0, 700, 1300, 1000, 1500, 3),
}


def max_stack_layer_for(coil_type: CoilType) -> int:
    """Bobin tipinin izin verdiği maksimum istif katını döndürür.

    docs/01 doğrulama kuralı 8'in (tip–kat tutarlılığı) tek noktadan uygulaması:
    HOT_ROLLED için 2, diğerleri için 3.
    """
    return COIL_TYPE_SPECS[coil_type].max_stack_layer


@dataclass
class SteelCoil:
    """Sistemin depoya yerleştirdiği temel nesne — tek bir çelik bobin.

    Bu sınıf bobinin tüm yaşam döngüsünü taşır; ``status`` ve ``location``
    simülasyon ilerledikçe değişir, bu yüzden sınıf değiştirilebilir (frozen değil).

    Alanlar:
        coil_id: Benzersiz kimlik (ör. "COIL-000001").
        coil_type: Üretim tipi; fiziksel aralıkları belirler.
        weight_ton: Ağırlık (ton), tipe göre 10–30 bandında.
        width_mm: Genişlik (mm).
        diameter_mm: Dış çap (mm).
        quality_class: Yüzey hassasiyet sınıfı.
        max_stack_layer: Tipten türetilen maksimum istif katı.
        production_time: Üretim bandından çıkış anı.
        order_id: Ait olduğu sipariş; henüz atanmadıysa None.
        status: Yaşam döngüsü durumu.
        location: Depodaki konum; depoda değilse None.
        urgency_score: 0..1 sevkiyat aciliyeti; simülasyonda hesaplanır.
        rehandled: Sevkiyatta engelleyici olarak yer değiştirdiyse True. Çalışma-zamanı
            bayrağıdır (veri setine yazılmaz); görselleştirmede işaretlemek için kullanılır.
        stored_at: Bobinin depoya GİRDİĞİ simülasyon saati (saat). Yerleştirmede atanır,
            swap'ta KORUNUR (bobin depoda kalır, sadece yeri değişir). Görselleştirmede
            bekleme süresi (dwell time) = şu anki saat − stored_at olarak gösterilir.
            Çalışma-zamanı alanıdır; depoda değilse None.
        swap_reason: Bu bobin SWAP ile taşındıysa kararın gerekçesi (sözlük). Hangi acil
            bobin için yer açıldığı, taşıma/alternatif maliyeti vb. Görselleştirmede
            elmas hover'ında gösterilir; taşınmadıysa None. Çalışma-zamanı alanıdır.
    """

    coil_id: str
    coil_type: CoilType
    weight_ton: float
    width_mm: int
    diameter_mm: int
    quality_class: QualityClass
    max_stack_layer: int
    production_time: datetime
    order_id: str | None
    status: CoilStatus
    location: SlotCoord | None
    urgency_score: float
    rehandled: bool = False
    stored_at: float | None = None
    swap_reason: dict | None = None
