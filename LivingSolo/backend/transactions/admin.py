"""
    Transaction, Trxn, 거래에 관한 어드민 패널입니다.
"""
from django.contrib import admin
from transactions.models import Transaction, PayMethod, Budget


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    """Transaction admin definition"""

    list_display = ("pk", "date", "amount", "memo")


@admin.register(PayMethod)
class PayMethodAdmin(admin.ModelAdmin):
    """PayMethod admin definition"""

    list_display = ("pk", "type", "name", "currency")


@admin.register(Budget)
class BudgetAdmin(admin.ModelAdmin):
    """Budget admin definition"""

    list_display = ("pk", "name", "amount", "tag_preset")
