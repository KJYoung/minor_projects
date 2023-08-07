from django.urls import path
from transactions.views import (
    general_transaction,
    general_trxn_combined,
    detail_transaction,
)

urlpatterns = [
    path('', general_transaction, name="trxn"),
    # Params: keyword<String?>, year<Number?(Number if month is valid)>, month<Number?>
    path('combined/', general_trxn_combined, name="trxn_combined"),
    # Params: year<Number?(Number if month is valid)>, month<Number?>
    path('<int:trxn_id>/', detail_transaction, name="trxn_detail"),
]
