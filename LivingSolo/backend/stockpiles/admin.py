"""
   Stockpile, StockpileTransaction에 관한 어드민 패널입니다.
"""
from django.contrib import admin
from stockpiles.models import Stockpile, StockpileTransaction


@admin.register(Stockpile)
class StockpileAdmin(admin.ModelAdmin):
    """Stockpile admin definition"""

    list_display = ("name", "pk")


@admin.register(StockpileTransaction)
class StockpileTransactionAdmin(admin.ModelAdmin):
    """StockpileTransaction admin definition"""

    list_display = ("target", "delta_amount", "memo")
