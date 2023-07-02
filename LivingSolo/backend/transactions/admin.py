from django.contrib import admin
from transactions.models import Transaction, TransactionType


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    """Transaction admin definition"""

    list_display = ("pk", "date", "type", "amount", "memo")


@admin.register(TransactionType)
class TransactionAdmin(admin.ModelAdmin):
    """Transaction admin definition"""

    list_display = ("pk", "created", "name")
