from django.contrib import admin
from transactions.models import Transaction


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    """Transaction admin definition"""

    list_display = ("pk", "date", "amount", "memo")
