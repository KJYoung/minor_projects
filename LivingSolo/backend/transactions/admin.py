"""
    Transaction, Trxn, 거래에 관한 어드민 패널입니다.
"""
from django.contrib import admin
from transactions.models import Transaction


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    """Transaction admin definition"""

    list_display = ("pk", "date", "amount", "memo")
