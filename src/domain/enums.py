"""Veri sözleşmesindeki (docs/01) tüm sabit kategorik tipler.

Tüm enum'lar tek bir dosyada toplanmıştır çünkü bazıları (özellikle
``LogisticsLine``) birden fazla model tarafından paylaşılır; bunları ayrı
dosyalara dağıtmak modüller arası döngüsel içe aktarma riskini doğurur.

Her enum ``str`` tabanlıdır: üyeler aynı zamanda kendi metinsel değerleridir.
Bunun sebebi, tabular verilerin ``pandas`` üzerinden okunabilir metin olarak
``parquet``e yazılması ve JSON'a serileştirmenin sorunsuz olmasıdır
(``CoilType.HOT_ROLLED`` -> "HOT_ROLLED"). Geri okurken ``CoilType("HOT_ROLLED")``
çağrısı üyeyi yeniden üretir.
"""

from __future__ import annotations

from enum import Enum


class CoilType(str, Enum):
    """Çelik bobinin üretim tipi. Fiziksel parametre aralıklarını belirler."""

    HOT_ROLLED = "HOT_ROLLED"      # sıcak hadde — en ağır, en az istiflenebilir
    COLD_ROLLED = "COLD_ROLLED"    # soğuk hadde — orta ağırlık
    GALVANIZED = "GALVANIZED"      # galvaniz — en hafif, yüzeyi hassas


class QualityClass(str, Enum):
    """Yüzey hassasiyet sınıfı. A daha hassas (genelde galvaniz), B standart."""

    A = "A"
    B = "B"


class CoilStatus(str, Enum):
    """Bobinin yaşam döngüsündeki anlık durumu."""

    IN_PRODUCTION = "IN_PRODUCTION"          # üretim bandında, henüz depoya gelmedi
    PENDING_PLACEMENT = "PENDING_PLACEMENT"  # depoya geldi, yerleştirilmeyi bekliyor
    STORED = "STORED"                        # depoda bir konuma yerleştirildi
    LOADED = "LOADED"                        # araca yüklendi
    DISPATCHED = "DISPATCHED"                # tesisten sevk edildi


class VehicleType(str, Enum):
    """Sevkiyat aracının türü. Kapasite ve gecikme oynaklığını etkiler."""

    TRUCK = "TRUCK"    # TIR — ~25 ton, en düşük kapasite
    TRAIN = "TRAIN"    # tren — ~60 ton, en stabil
    SHIP = "SHIP"      # gemi — ~120 ton, en oynak lojistik


class Weather(str, Enum):
    """Aracın varış günündeki hava durumu. Gecikmenin ana sürücülerinden."""

    CLEAR = "CLEAR"    # açık — gecikme etkisi yok
    RAIN = "RAIN"      # yağmur — orta gecikme
    SNOW = "SNOW"      # kar — yüksek gecikme


class OrderPriority(str, Enum):
    """Siparişin aciliyet düzeyi."""

    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"


class OrderStatus(str, Enum):
    """Siparişin işlem durumu."""

    OPEN = "OPEN"                # açık, henüz işleme alınmadı
    IN_PROGRESS = "IN_PROGRESS"  # yükleme süreci başladı
    FULFILLED = "FULFILLED"      # tamamlandı
    CANCELLED = "CANCELLED"      # iptal edildi


class LogisticsLine(str, Enum):
    """Sevkiyat hattı. Hangi zone'un hangi taşıma moduna hizmet ettiğini bağlar.

    Affinity (aynı hatta giden bobinleri aynı zone'a koyma) ödülünün anlamlı
    olabilmesi için araç ve zone bu enum üzerinden eşleştirilir.
    """

    SHIP_1 = "SHIP_1"
    SHIP_2 = "SHIP_2"
    TRAIN_A = "TRAIN_A"
    TRUCK_DOCK = "TRUCK_DOCK"


class EventType(str, Enum):
    """Simülasyon zaman ekseninde üretilen dinamik olay tipleri."""

    NEW_ORDER = "NEW_ORDER"              # kuyruğa yeni sipariş geldi
    CANCEL_ORDER = "CANCEL_ORDER"        # mevcut sipariş iptal edildi
    VEHICLE_DELAY = "VEHICLE_DELAY"      # araç planlanandan geç kalıyor
    PRIORITY_CHANGE = "PRIORITY_CHANGE"  # siparişin önceliği değişti
    PEAK_LOAD = "PEAK_LOAD"              # zirve/kriz senaryosu tetiklendi
