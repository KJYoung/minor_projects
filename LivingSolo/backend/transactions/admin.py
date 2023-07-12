from django.contrib import admin
from transactions.models import Transaction, TransactionType, TransactionTypeClass


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    """Transaction admin definition"""

    list_display = ("pk", "date", "amount", "memo")

@admin.register(TransactionTypeClass)
class TransactionTypeClassAdmin(admin.ModelAdmin):
    """Transaction Type Class admin definition"""

    list_display = ("pk", "created", "name", "color")

@admin.register(TransactionType)
class TransactionTypeAdmin(admin.ModelAdmin):
    """Transaction Type admin definition"""

    list_display = ("pk", "created", "name", "color", "type_class")
