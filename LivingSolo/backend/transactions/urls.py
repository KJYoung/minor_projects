from django.urls import path
from transactions.views import (
    general_transaction,
    detail_transaction,
    general_trxn_type,
    general_trxn_type_class,
)

urlpatterns = [
    path('', general_transaction, name="trxn"),
    path('<int:trxn_id>/', detail_transaction, name="trxn_detail"),
    path('type_class/', general_trxn_type_class, name="trxn_type_class"),
    path('type/', general_trxn_type, name="trxn_type"),
]
